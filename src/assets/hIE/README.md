# hIE Visual Library

This directory holds illustrations for the blog's 5-hIE taxonomy. Every blog post is mapped to one of the five Beatless hIE characters; the post's hero image is selected from that character's directory.

## Mapping (canonical: `~/claw/plan/blog-taxonomy-hIE.md`)

| Directory | hIE (Beatless type) | Vibe | Blog category |
|---|---|---|---|
| `snowdrop/` | hIE-002 Snowdrop | counterfactual, breakthrough, snow | `paper-spotlight` |
| `kouka/` | hIE-001 Kouka | high-pressure, deadline, red | `signal` (news) |
| `saturnus/` | hIE-003 Saturnus | governance, audit, gold | `meta` (audits) |
| `methode/` | hIE-004 Methode | engineering, automation, tool | `engineering` |
| `lacia/` | hIE-005 Lacia | narrative, synthesis, black | `bundle` / `digest` |
| `_shared/` | (multi-hIE / generic Beatless) | overview / index | taxonomy hero, multi-character posts |

## Sources

| Source | URL pattern | Files here |
|---|---|---|
| AlphaCoders | `images*.alphacoders.com/<3>/<id>.jpg` | `_shared/group-5hIE-arato-alphacoders-896444.jpg`, `_shared/group-kengo-lacia-922740.jpg`, `_shared/cybernetic-922741.jpg`, `lacia/lacia-*.jpg` |
| yande.re (filenames preserve original artist tags: redjuice, fhilippedu, ogawa_akane, …) | `files.yande.re/image/<hash>/yande.re%20<id>%20...` | `snowdrop/`, `kouka/`, `methode/`, `_shared/280186-*` |

## License & attribution stance

All images here are fan-art / anime stills / wallpapers from the Beatless light novel & anime (© KEI / 長谷敏司 / 講談社 / Beatless production committee). Original artists (most prominently **redjuice / Kuwashima Rei**) are credited in filenames where the source preserved that.

**Use stance**: personal blog, non-commercial, transformative-curation context (using these as section markers tied to a custom 5-way taxonomy, not as standalone reproductions). Any blog post embedding one of these images **must include the source URL and original artist credit in the frontmatter** (`sources:` field) or as a caption.

When in doubt, use a `_shared/` group shot rather than a single-character keyvis.

## The Saturnus gap

`saturnus/` is **empty** as of 2026-04-25.

Reason: Saturnus has very limited public single-character imagery. yande.re's Beatless tag has 280+ posts but **0** tagged `saturnus`. AlphaCoders' Beatless category is dominated by Lacia.

**Options to fill `saturnus/`**:
1. Crop Saturnus's region from `_shared/group-5hIE-arato-alphacoders-896444.jpg` (5-hIE same-frame, 1920×1080) — she's clearly visible and croppable. Recommended first step.
2. Search Pixiv / Twitter for `#セラタス` / `#Saturnus_Beatless` (requires login the cron can't do).
3. Use a mood-substitute: gold-tone abstract, archival imagery, structured-pattern photo (CC-BY from Unsplash). Tag as `mood-substitute: true` in frontmatter so future you can swap when canon art is found.

Until `saturnus/` is filled, audit-meta posts default to `_shared/group-5hIE-arato-alphacoders-896444.jpg` (Saturnus is in that frame).

## Adding more images

1. Find a candidate URL (yande.re, AlphaCoders, Wallhaven if your network can reach it).
2. Download to the matching `<hIE>/` directory.
3. Filename convention: `<source-id>-<artist>-<descriptors>.jpg` — keep tags from source URL when present (helps future filter/dedup).
4. If the image features multiple hIEs, put it in `_shared/` and prefix `multi-<characters>-`.
5. Update this README's per-character count if it materially changes.

## Current inventory (2026-04-25)

```
_shared/   4 files  3.4 MB   ← group + 5-hIE taxonomy hero
snowdrop/  3 files  3.6 MB   ← redjuice originals
kouka/     3 files  2.7 MB   ← redjuice + nyaa
saturnus/  0 files  0.0 MB   ← needs cropping or manual sourcing
methode/   2 files  1.0 MB   ← fhilippedu + group
lacia/     2 files  2.0 MB   ← AlphaCoders 1920×1200 + futuristic glow
                  ─────────
total:    14 files  13 MB
```
