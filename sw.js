/* Offline cold-load for the static site (the phone's notes must be readable with no connection and no chat quota).
   Minimal, runtime-caching only — no precache list to keep in sync:
     - navigations: network-first (a deploy is picked up on the next online visit), cache fallback offline
     - immutable hashed assets: cache-first
     - feed JSON and other same-origin GETs: stale-while-revalidate
     - cross-origin (the chat Worker) and non-GET: passed straight through, never cached             */
const CACHE = 'citadel-v1'
const IMMUTABLE = /\/assets\/|.+\.(js|css|woff2?|glb|gltf|bin|ktx2|webp|png|jpg|mp3|ogg)$/

self.addEventListener('install', () => self.skipWaiting())
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      // claim at once, and put the document itself in the cache: the first visit's assets were fetched before this
      // worker controlled the page, so the SECOND navigation (online or offline) is the fully covered one
      .then(() => Promise.all([self.clients.claim(), caches.open(CACHE).then((c) => c.addAll(['/']).catch(() => {}))])),
  )
})

self.addEventListener('fetch', (event) => {
  const req = event.request
  if (req.method !== 'GET') return
  const url = new URL(req.url)
  if (url.origin !== location.origin) return // never intercept the chat Worker or any other origin

  if (req.mode === 'navigate') {
    event.respondWith(fetch(req).then((res) => { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); return res }).catch(() => caches.match(req).then((hit) => hit || caches.match('/'))))
    return
  }
  if (IMMUTABLE.test(url.pathname)) {
    event.respondWith(caches.match(req).then((hit) => hit || fetch(req).then((res) => { if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)) } return res })))
    return
  }
  event.respondWith(
    caches.match(req).then((hit) => {
      const refresh = fetch(req).then((res) => { if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)) } return res }).catch(() => hit)
      return hit || refresh
    }),
  )
})
