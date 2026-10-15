/* Offline cold-load for the static site (the phone's notes must be readable with no connection and no chat quota).
   Minimal, runtime-caching only — no precache list to keep in sync:
     - navigations and JSON (feed, build.json, scene data): network-first, cache fallback offline
     - Vite's content-hashed /assets/: cache-first (a new build has new names)
     - other same-origin GETs (public art, models): stale-while-revalidate — their names never change, so they
       must not be cache-first or an updated picture never reaches a returning visitor
     - cross-origin (the chat Worker) and non-GET: passed straight through, never cached             */
// bumping the name drops every entry cached by an older worker (v1 pinned public art and build.json forever)
const CACHE = 'citadel-v2'
const HASHED = /^\/assets\//

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

const store = (req, res) => { if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)) } return res }

self.addEventListener('fetch', (event) => {
  const req = event.request
  if (req.method !== 'GET') return
  const url = new URL(req.url)
  if (url.origin !== location.origin) return // never intercept the chat Worker or any other origin

  if (req.mode === 'navigate' || url.pathname.endsWith('.json')) {
    event.respondWith(fetch(req).then((res) => store(req, res)).catch(() => caches.match(req).then((hit) => hit || (req.mode === 'navigate' ? caches.match('/') : Response.error()))))
    return
  }
  if (HASHED.test(url.pathname)) {
    event.respondWith(caches.match(req).then((hit) => hit || fetch(req).then((res) => store(req, res))))
    return
  }
  event.respondWith(
    caches.match(req).then((hit) => {
      const refresh = fetch(req).then((res) => store(req, res)).catch(() => hit || Response.error())
      if (hit) event.waitUntil(refresh)
      return hit || refresh
    }),
  )
})
