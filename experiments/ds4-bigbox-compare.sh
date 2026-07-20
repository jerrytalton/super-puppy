#!/usr/bin/env bash
#
# ds4-bigbox-compare.sh — DeepSeek V4 PRO (ds4) vs glm-5.2 (SP/MLX), head to head,
# on the 512GB M3 Ultra. This is the deciding test from the ds4 assessment
# (docs/superpowers/specs/2026-07-19-ds4-backend-integration-assessment.md):
# is a second single-lane big-model backend worth it over glm-5.2?
#
# Run it on the M3 Ultra when it's up:   experiments/ds4-bigbox-compare.sh
# Idempotent: skips build/download if already present. Cleans up the ds4 server
# on exit. Saves per-model outputs + a tok/s summary for you to judge quality.
#
# Env overrides: DS4_DIR, DS4_MODEL (download target), GLM_MODEL, MLX_URL,
#                MAX_TOKENS, PORT.
set -uo pipefail   # NOT -e: one model failing must not abort the whole run.

DS4_DIR="${DS4_DIR:-$HOME/experiments/ds4}"
DS4_MODEL="${DS4_MODEL:-pro-q2-imatrix}"      # 512GB PRO; ~430GB. Or q4-imatrix (~153GB).
PORT="${PORT:-8002}"                          # MLX owns 8000
MLX_URL="${MLX_URL:-http://localhost:8000}"   # SP's mlx-openai-server serves glm-5.2
GLM_MODEL="${GLM_MODEL:-glm-5.2}"
MAX_TOKENS="${MAX_TOKENS:-512}"
OUT="$HOME/experiments/ds4-compare-$(date +%Y%m%d-%H%M%S)"
DS4_URL="http://127.0.0.1:${PORT}"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }

DS4_PID=""
cleanup() {
    if [ -n "$DS4_PID" ] && kill -0 "$DS4_PID" 2>/dev/null; then
        say "Stopping ds4-server (freeing RAM)"
        kill "$DS4_PID" 2>/dev/null; sleep 3; kill -9 "$DS4_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ── Preflight ────────────────────────────────────────────────────────────
say "Preflight"
RAM_GB=$(sysctl -n hw.memsize | awk '{printf "%d", $1/1073741824}')
note "RAM: ${RAM_GB} GB"
if [ "$DS4_MODEL" = "pro-q2-imatrix" ] && [ "$RAM_GB" -lt 460 ]; then
    note "WARNING: pro-q2 wants ~512GB; this machine has ${RAM_GB}GB. Set DS4_MODEL=q4-imatrix for a smaller run, or run on the M3 Ultra."
fi
note "Disk free: $(df -h "$HOME" | awk 'NR==2{print $4}')"

# ── Build ds4 if needed ──────────────────────────────────────────────────
if [ ! -x "$DS4_DIR/ds4-server" ]; then
    say "Building ds4 (not found at $DS4_DIR)"
    [ -d "$DS4_DIR/.git" ] || git clone --depth 1 https://github.com/antirez/ds4 "$DS4_DIR"
    ( cd "$DS4_DIR" && make ) || { note "build failed"; exit 1; }
fi
note "ds4-server: $DS4_DIR/ds4-server"

# ── glm5.2 branch state (assessment open question Q1) ────────────────────
say "ds4 glm5.2 branch status"
if git -C "$DS4_DIR" ls-remote --exit-code origin glm5.2 >/dev/null 2>&1; then
    note "glm5.2 branch EXISTS on origin: $(git -C "$DS4_DIR" ls-remote origin glm5.2 | awk '{print substr($1,1,10)}')"
    note "-> worth investigating whether it could retire bin/apply-mlx-glm52-patch.sh"
else
    note "no glm5.2 branch on origin (or already merged/removed) — retire-the-patch idea likely dead"
fi

# ── Download the PRO model if needed ─────────────────────────────────────
if [ ! -e "$DS4_DIR/ds4flash.gguf" ] || [ ! -s "$(readlink "$DS4_DIR/ds4flash.gguf" 2>/dev/null || echo /nonexistent)" ]; then
    say "Model '$DS4_MODEL' not present"
    note "This downloads a LARGE model (pro-q2 ≈ 430GB, q4 ≈ 153GB) into $DS4_DIR."
    read -r -p "   Download '$DS4_MODEL' now? [y/N] " ans
    case "$ans" in
        [Yy]*) ( cd "$DS4_DIR" && ./download_model.sh "$DS4_MODEL" ) || { note "download failed"; exit 1; } ;;
        *) note "Skipped. Re-run after downloading, or point DS4_DIR at an existing model."; exit 0 ;;
    esac
fi
note "model: $(readlink "$DS4_DIR/ds4flash.gguf")"

# ── Start ds4-server (full residency; 512GB has room) ────────────────────
say "Starting ds4-server on :$PORT"
nohup "$DS4_DIR/ds4-server" --metal --port "$PORT" --ctx 32768 -m "$DS4_DIR/ds4flash.gguf" \
    > /tmp/ds4-compare-server.log 2>&1 &
DS4_PID=$!
note "pid $DS4_PID — waiting for load..."
for i in $(seq 1 120); do
    sleep 4
    curl -s -o /dev/null -w '%{http_code}' "$DS4_URL/v1/models" 2>/dev/null | grep -q 200 && { note "ready after ~$((i*4))s"; break; }
    kill -0 "$DS4_PID" 2>/dev/null || { note "ds4-server died — see /tmp/ds4-compare-server.log"; tail -8 /tmp/ds4-compare-server.log; exit 1; }
done

# ── Warm glm-5.2 (it may be on_demand on the MLX server) ─────────────────
say "Warming $GLM_MODEL on $MLX_URL (cold-load may take a while)"
curl -s -o /dev/null --max-time 300 "$MLX_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$GLM_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" || note "glm warmup returned non-200 (check the model name / MLX server)"

# ── Comparison prompts ───────────────────────────────────────────────────
mkdir -p "$OUT"
PROMPTS=(
"reasoning|Five people (A,B,C,D,E) sit in a row. A is not at either end. B is immediately left of C. D is at one end. E is not next to A. Give a valid seating and explain step by step."
"coding|Write a Python function that merges N sorted iterators into one sorted stream lazily, using a heap. Include docstring and edge cases."
"longform|Explain, in ~400 words, the tradeoffs between a log-structured merge tree and a B-tree for a write-heavy embedded database."
)

# measurement helper (written out to avoid shell-quoting pain)
cat > "$OUT/_measure.py" <<'PY'
import json, subprocess, sys
url, model, prompt, max_tokens, outfile = sys.argv[1:6]
body = json.dumps({"model": model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": int(max_tokens), "temperature": 0})
p = subprocess.run(["curl","-s","-w","\n__t=%{time_total}","--max-time","1200",
                    url+"/v1/chat/completions","-H","Content-Type: application/json","-d",body],
                   capture_output=True, text=True)
raw = p.stdout
try:
    t = float(raw.split("__t=")[1]); d = json.loads(raw.split("__t=")[0])
    u = d.get("usage", {}); ct = u.get("completion_tokens", 0); pt = u.get("prompt_tokens", 0)
    msg = d["choices"][0]["message"]
    # thinking models (glm-5.2, qwen3.6) may put text in reasoning_content, not content
    text = (msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning")
            or (json.dumps(msg.get("tool_calls")) if msg.get("tool_calls") else "")
            or json.dumps(msg))
    open(outfile, "w").write(text or "")
    print(f"{ct} {pt} {t:.1f} {ct/t if t else 0:.1f}")
except Exception as e:
    open(outfile, "w").write("ERROR: "+raw[:500])
    print(f"0 0 0 0  # ERROR {e}")
PY

say "Running comparison (outputs saved to $OUT)"
SUMMARY="$OUT/summary.md"
{
  echo "# ds4 (DeepSeek V4 $DS4_MODEL) vs $GLM_MODEL — $(date)"
  echo
  echo "| Prompt | Model | completion_tok | prompt_tok | wall_s | tok/s |"
  echo "|---|---|---|---|---|---|"
} > "$SUMMARY"

for entry in "${PROMPTS[@]}"; do
    name="${entry%%|*}"; prompt="${entry#*|}"
    for pair in "ds4:$DS4_URL:deepseek-v4-pro" "glm:$MLX_URL:$GLM_MODEL"; do
        tag="${pair%%:*}"; rest="${pair#*:}"; u="${rest%:*}"; m="${rest##*:}"
        note "[$name / $tag] generating..."
        stats=$(python3 "$OUT/_measure.py" "$u" "$m" "$prompt" "$MAX_TOKENS" "$OUT/${name}.${tag}.txt")
        set -- $stats
        echo "| $name | $tag ($m) | ${1:-?} | ${2:-?} | ${3:-?} | ${4:-?} |" >> "$SUMMARY"
    done
done

say "Done"
cat "$SUMMARY"
note ""
note "Read the side-by-side outputs to judge QUALITY (tok/s is only half the story):"
note "  $OUT/<prompt>.ds4.txt  vs  $OUT/<prompt>.glm.txt"
note ""
note "Decision (from the assessment): merge the ds4 integration ONLY if V4 PRO"
note "clearly beats glm-5.2 on quality here — otherwise it's a redundant second"
note "single-lane backend and we correctly skip it."
