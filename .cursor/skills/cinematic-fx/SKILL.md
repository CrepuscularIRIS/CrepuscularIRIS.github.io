---
name: cinematic-fx
description: Enforces hospital-night cinematic atmosphere for 《担保人》 Astro pages. Use when editing web/ UI, CSS, motion, grain, scanlines, glitch, or any visual of this story site. Prevents shipping static box layouts without DESIGN.md FX.
---

# Cinematic FX

This repo already has a visual brief. Do not invent a second aesthetic. Do not skip atmosphere to ship faster.

## Authority

1. `docs/spec/guarantor.md` is the in-repo product spec. If `E:\bug-film\DESIGN.md` exists locally, its tokens/bans win on hex values; they must not contradict the spec.
2. Vercel Web Interface Guidelines win on motion mechanics: `transform`/`opacity` only, never `transition: all`, honor `prefers-reduced-motion`, `:focus-visible`, `color-scheme`.
3. Anthropic `frontend-design` skill: ambient atmosphere is a required motion choice for this brief, not optional polish.

## Required FX (must exist in the live page)

| Token | Where |
|---|---|
| FilmGrain | fixed overlay, whole site, z-index 50, must be visible in a screenshot |
| Scanlines | dark sections, 3px, **over** the HUD (z 50), off on chat white |
| CRTFlicker | terminal panel |
| GlitchRGB | countdown digits, persistent chromatic offset |
| glow/pulse | live LEDs, ventilator frame |
| cam sweep | still/camera frames |
| vignette | dark sections |

Looping motion only inside `@media (prefers-reduced-motion: no-preference)`.

## Fail the page if

- A 1440×900 screenshot of a dark section looks like flat CSS boxes with no grain/scan/vignette
- Grain is the only FX and is invisible (0.06 overlay on black is not enough by itself)
- Scanlines sit at z-index 10 while panels sit at 20
- You skipped FX because Karpathy/lazy/minimum-code

## Bans (still)

Purple gradient, Inter, glassmorphism, Scroll down cue, em-dash, Google Fonts, character sprites.
