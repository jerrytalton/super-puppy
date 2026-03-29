#include <dlfcn.h>
#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <mach-o/dyld.h>

/*
 * Super Puppy app bundle launcher.
 *
 * Embeds Python via dlopen so this Mach-O is the process running
 * NSApplication. macOS reads CFBundleName for Cmd-Tab, screen
 * recording attribution, etc.
 *
 * Phase 1: `uv run menubar.py --python-info` → base_prefix, libpython,
 *          site-packages (ensures venv + deps exist).
 * Phase 2: dlopen libpython, set PYTHONHOME to base_prefix, add the
 *          venv site-packages to PYTHONPATH, run menubar.py.
 */

typedef void    (*Py_SetPythonHome_t)(const wchar_t *);
typedef void    (*Py_Initialize_t)(void);
typedef int     (*Py_FinalizeEx_t)(void);
typedef int     (*PyRun_SimpleFile_t)(FILE *, const char *);
typedef void    (*PySys_SetArgvEx_t)(int, wchar_t **, int);
typedef wchar_t*(*Py_DecodeLocale_t)(const char *, size_t *);

static char *next_line(char *s) {
    char *nl = strchr(s, '\n');
    if (!nl) return NULL;
    *nl = '\0';
    return nl + 1;
}

int main(int argc, char *argv[]) {
    char exe[PATH_MAX];
    uint32_t esize = sizeof(exe);
    if (_NSGetExecutablePath(exe, &esize) != 0) return 1;

    char resolved[PATH_MAX];
    if (!realpath(exe, resolved)) return 1;

    /* .app/Contents/MacOS/super-puppy → app/ */
    char rcopy[PATH_MAX];
    strlcpy(rcopy, resolved, sizeof(rcopy));
    char *dir = dirname(rcopy);     /* MacOS */
    dir = dirname(dir);             /* Contents */
    dir = dirname(dir);             /* .app */
    dir = dirname(dir);             /* app/ */

    char script[PATH_MAX];
    snprintf(script, sizeof(script), "%s/menubar.py", dir);

    const char *home = getenv("HOME");
    if (!home) { fprintf(stderr, "super-puppy: HOME not set\n"); return 1; }

    /* Ensure user-local paths are in PATH (launchd/open gives minimal PATH) */
    const char *old_path = getenv("PATH");
    char new_path[PATH_MAX * 4];
    snprintf(new_path, sizeof(new_path), "%s/.local/bin:%s/bin:%s",
             home, home, old_path ? old_path : "/usr/bin:/bin:/usr/sbin:/sbin");
    setenv("PATH", new_path, 1);

    char uv[PATH_MAX];
    snprintf(uv, sizeof(uv), "%s/.local/bin/uv", home);

    /* Phase 1 */
    int pipefd[2];
    if (pipe(pipefd) != 0) return 1;

    pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);
        execl(uv, "uv", "run", "--python", "3.12", script,
              "--python-info", NULL);
        _exit(1);
    }
    close(pipefd[1]);

    char buf[PATH_MAX * 4];
    ssize_t n = read(pipefd[0], buf, sizeof(buf) - 1);
    close(pipefd[0]);
    if (n <= 0) { fprintf(stderr, "super-puppy: uv failed to resolve Python\n"); return 1; }
    buf[n] = '\0';

    int status;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "super-puppy: uv exited with status %d\n",
                WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        return 1;
    }

    /* Parse: line 1 = base_prefix, line 2 = libpython, line 3 = site-packages */
    char *base_prefix = buf;
    char *libpath = next_line(base_prefix);
    if (!libpath) { fprintf(stderr, "super-puppy: unexpected uv output\n"); return 1; }
    char *sitepkgs = next_line(libpath);
    if (sitepkgs) next_line(sitepkgs);  /* strip trailing newline */

    /* Phase 2 */
    void *lib = dlopen(libpath, RTLD_LAZY | RTLD_GLOBAL);
    if (!lib) {
        fprintf(stderr, "super-puppy: dlopen(%s): %s\n", libpath, dlerror());
        return 1;
    }

    Py_SetPythonHome_t py_home = dlsym(lib, "Py_SetPythonHome");
    Py_Initialize_t    py_init = dlsym(lib, "Py_Initialize");
    Py_FinalizeEx_t    py_fin  = dlsym(lib, "Py_FinalizeEx");
    PyRun_SimpleFile_t py_run  = dlsym(lib, "PyRun_SimpleFile");
    PySys_SetArgvEx_t  py_argv = dlsym(lib, "PySys_SetArgvEx");
    Py_DecodeLocale_t  py_dec  = dlsym(lib, "Py_DecodeLocale");

    if (!py_home || !py_init || !py_run || !py_dec) {
        fprintf(stderr, "super-puppy: missing Python C API symbols\n");
        return 1;
    }

    /* Set PYTHONHOME to base prefix (has stdlib) */
    wchar_t *whome = py_dec(base_prefix, NULL);
    if (!whome) { fprintf(stderr, "super-puppy: Py_DecodeLocale failed for home\n"); return 1; }
    py_home(whome);

    /* Add venv site-packages so deps (rumps, pyobjc, etc.) are importable */
    if (sitepkgs && sitepkgs[0]) {
        setenv("PYTHONPATH", sitepkgs, 1);
    }

    py_init();

    wchar_t *wscript = py_dec(script, NULL);
    if (!wscript) { fprintf(stderr, "super-puppy: Py_DecodeLocale failed for script\n"); return 1; }
    if (py_argv) py_argv(1, &wscript, 0);

    FILE *fp = fopen(script, "r");
    if (!fp) { fprintf(stderr, "super-puppy: can't open %s\n", script); return 1; }
    int ret = py_run(fp, script);
    fclose(fp);

    if (py_fin) py_fin();

    /* PyMem_RawFree is the correct way to free Py_DecodeLocale results,
       but we're about to exit anyway — OS reclaims all process memory. */
    return ret;
}
