const CACHE_VERSION = 'sp-v1';
const SHELL_ASSETS = [
  '/tools',
  '/manifest.json',
  '/pwa/icon-192.png',
  '/pwa/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION)
      .then(cache => cache.addAll(SHELL_ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_VERSION).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // API calls and streaming — always network, never cache
  if (url.pathname.startsWith('/api/')) {
    return;
  }

  // App shell — network first, fall back to cache
  event.respondWith(
    fetch(event.request)
      .then(resp => {
        if (resp.ok) {
          const clone = resp.clone();
          caches.open(CACHE_VERSION).then(c => c.put(event.request, clone));
        }
        return resp;
      })
      .catch(() => caches.match(event.request))
  );
});
