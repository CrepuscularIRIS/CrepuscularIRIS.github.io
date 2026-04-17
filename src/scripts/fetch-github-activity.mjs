/**
 * Fetch GitHub activity from people I follow.
 *
 * Two-step approach per actions.md design:
 *   1. GET /user/following  → list of followed users
 *   2. GET /users/{login}/events/public → each user's public events
 *
 * Output: src/data/github-activity.json
 * Runs via blog-maintenance pipeline (hourly).
 */

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const OUTPUT_PATH = path.resolve(__dirname, '../data/github-activity.json')

const GH_TOKEN = process.env.GITHUB_TOKEN || process.env.GH_TOKEN
const GH_USER = process.env.GITHUB_USER || 'CrepuscularIRIS'

if (!GH_TOKEN) {
  console.error('Missing GITHUB_TOKEN env var')
  process.exit(1)
}

const BIG_ORGS = new Set(
  [
    'anthropics', 'anthropic', 'google', 'google-gemini', 'google-deepmind',
    'openai', 'microsoft', 'meta', 'meta-llama', 'nvidia', 'amd', 'apple',
    'huggingface', 'MiniMax-AI', 'deepseek-ai', 'mistralai',
    'tensorflow', 'pytorch', 'facebook', 'facebookresearch',
    'vercel', 'cloudflare', 'stripe', 'github',
  ].map((o) => o.toLowerCase()),
)

const BOT_LOGINS = new Set(['dependabot', 'renovate', 'codecov', 'pre-commit-ci'])
const isBot = (login) =>
  typeof login === 'string' && (login.endsWith('[bot]') || BOT_LOGINS.has(login.toLowerCase()))

const isBigOrg = (repoFullName) => {
  if (!repoFullName) return false
  const org = repoFullName.split('/')[0].toLowerCase()
  return BIG_ORGS.has(org)
}

async function gh(path, opts = {}) {
  const url = path.startsWith('http') ? path : `https://api.github.com${path}`
  const res = await fetch(url, {
    headers: {
      Authorization: `token ${GH_TOKEN}`,
      Accept: 'application/vnd.github+json',
      'User-Agent': `${GH_USER}-feed-digest`,
    },
    ...opts,
  })
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText} on ${url}`)
  }
  return res.json()
}

async function ghPaginate(path, maxPages = 5) {
  const all = []
  for (let page = 1; page <= maxPages; page++) {
    const data = await gh(`${path}${path.includes('?') ? '&' : '?'}per_page=100&page=${page}`)
    if (!Array.isArray(data) || data.length === 0) break
    all.push(...data)
    if (data.length < 100) break
  }
  return all
}

// Describe an event in one human-readable line.
function describeEvent(ev) {
  const repo = ev.repo?.name
  const payload = ev.payload || {}
  switch (ev.type) {
    case 'CreateEvent':
      if (payload.ref_type === 'repository')
        return { kind: 'new-repo', text: `created new repo ${repo}`, repo }
      if (payload.ref_type === 'branch')
        return { kind: 'branch', text: `pushed new branch ${payload.ref} to ${repo}`, repo }
      if (payload.ref_type === 'tag')
        return { kind: 'tag', text: `tagged ${payload.ref} on ${repo}`, repo }
      return null
    case 'ReleaseEvent':
      return {
        kind: 'release',
        text: `released ${payload.release?.tag_name || ''} of ${repo}`,
        repo,
        url: payload.release?.html_url,
      }
    case 'PublicEvent':
      return { kind: 'open-sourced', text: `made ${repo} public`, repo }
    case 'ForkEvent':
      return { kind: 'fork', text: `forked ${repo}`, repo }
    case 'WatchEvent':
      return { kind: 'star', text: `starred ${repo}`, repo }
    case 'PullRequestEvent':
      if (payload.action === 'opened')
        return {
          kind: 'pr-opened',
          text: `opened PR #${payload.pull_request?.number} on ${repo}`,
          repo,
          url: payload.pull_request?.html_url,
        }
      if (payload.action === 'closed' && payload.pull_request?.merged)
        return {
          kind: 'pr-merged',
          text: `merged PR #${payload.pull_request?.number} on ${repo}`,
          repo,
          url: payload.pull_request?.html_url,
        }
      return null
    case 'IssuesEvent':
      if (payload.action === 'opened')
        return {
          kind: 'issue',
          text: `opened issue #${payload.issue?.number} on ${repo}`,
          repo,
          url: payload.issue?.html_url,
        }
      return null
    case 'PushEvent':
      if ((payload.commits?.length ?? 0) > 0) {
        return {
          kind: 'push',
          text: `pushed ${payload.commits.length} commit(s) to ${repo}`,
          repo,
        }
      }
      return null
    default:
      return null
  }
}

const RANK = {
  'new-repo': 10,
  release: 9,
  'open-sourced': 9,
  'pr-merged': 8,
  'pr-opened': 7,
  fork: 6,
  issue: 5,
  star: 4,
  tag: 3,
  branch: 2,
  push: 1,
}

async function main() {
  const startedAt = new Date()
  console.log(`[feed-digest] fetching following list for ${GH_USER}`)
  const following = await ghPaginate(`/users/${GH_USER}/following`)
  console.log(`[feed-digest] ${following.length} follows`)

  const cutoffMs = Date.now() - 7 * 24 * 60 * 60 * 1000 // last 7 days
  const perUser = []
  const highlights = [] // personal headliners: new repo, release, fork/star of indie repos
  const repoTrending = new Map() // repo -> { count, starers: Set(login), bigOrg }

  // Limit concurrency
  const CONCURRENCY = 5
  for (let i = 0; i < following.length; i += CONCURRENCY) {
    const batch = following.slice(i, i + CONCURRENCY)
    const results = await Promise.all(
      batch.map(async (user) => {
        if (isBot(user.login)) return null
        try {
          const events = await gh(`/users/${user.login}/events/public?per_page=100`)
          return { user, events }
        } catch (err) {
          console.warn(`[feed-digest] failed for ${user.login}: ${err.message}`)
          return null
        }
      }),
    )
    for (const r of results) {
      if (!r) continue
      const { user, events } = r
      const userEvents = []
      for (const ev of events) {
        if (isBot(ev.actor?.login)) continue
        const ts = new Date(ev.created_at).getTime()
        if (ts < cutoffMs) continue
        const desc = describeEvent(ev)
        if (!desc) continue
        const big = isBigOrg(desc.repo)
        userEvents.push({ ...desc, at: ev.created_at, big })

        // Trending aggregation
        if (desc.repo) {
          const t = repoTrending.get(desc.repo) || {
            repo: desc.repo,
            count: 0,
            starers: new Set(),
            bigOrg: big,
          }
          t.count += 1
          if (desc.kind === 'star') t.starers.add(user.login)
          repoTrending.set(desc.repo, t)
        }

        // Personal highlights: avoid big-org star/fork noise
        if (['new-repo', 'release', 'open-sourced'].includes(desc.kind)) {
          highlights.push({ user: user.login, name: user.login, ...desc, at: ev.created_at })
        } else if (['fork', 'star'].includes(desc.kind) && !big) {
          highlights.push({ user: user.login, name: user.login, ...desc, at: ev.created_at })
        }
      }
      if (userEvents.length > 0) {
        perUser.push({
          login: user.login,
          avatar: user.avatar_url,
          profile: user.html_url,
          eventCount: userEvents.length,
          events: userEvents.sort((a, b) => new Date(b.at) - new Date(a.at)).slice(0, 10),
        })
      }
    }
  }

  perUser.sort((a, b) => b.eventCount - a.eventCount)
  highlights.sort((a, b) => (RANK[b.kind] ?? 0) - (RANK[a.kind] ?? 0) || new Date(b.at) - new Date(a.at))

  // Trending: split into indie (show names) vs big (just count)
  const trending = Array.from(repoTrending.values())
    .sort((a, b) => b.count - a.count)
    .slice(0, 20)
    .map((t) => ({
      repo: t.repo,
      count: t.count,
      bigOrg: t.bigOrg,
      starers: Array.from(t.starers).slice(0, 8),
    }))

  const out = {
    generatedAt: startedAt.toISOString(),
    user: GH_USER,
    followingCount: following.length,
    activeUserCount: perUser.length,
    eventCount: perUser.reduce((s, u) => s + u.eventCount, 0),
    highlights: highlights.slice(0, 25),
    trending,
    perUser: perUser.slice(0, 60),
  }

  fs.mkdirSync(path.dirname(OUTPUT_PATH), { recursive: true })
  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(out, null, 2))
  console.log(
    `[feed-digest] wrote ${OUTPUT_PATH}: ${out.activeUserCount} active, ${out.eventCount} events`,
  )
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
