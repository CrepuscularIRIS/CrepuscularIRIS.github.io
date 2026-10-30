# Rocketbox ambient residents

These two files are unmodified copies from Tidewater revision
`4811ba48d795197de5621985f404e765c0b7c0ef`, `public/models/characters/`.
Upstream repository: https://github.com/microsoft/Microsoft-Rocketbox

Tidewater's `CREDITS.md` identifies the original avatars as `Wood_Male_01`
(Joe) and `Female_Adult_04` (Marta), with their idle, talk, wave and shrug
animations from the Microsoft Rocketbox Avatar Library, converted using
Tidewater's `tools/characters/` (textures resized; animations retargeted and baked).
Copyright (c) 2020 Microsoft. MIT license: [LICENSE-Rocketbox.md](LICENSE-Rocketbox.md).

| File | SHA256 | Bytes |
| --- | --- | ---: |
| joe.glb | da3efce5a452d3c17f40b8d33bf449f565c0a6bef93602dc3e74364fb64081ca | 4743204 |
| marta.glb | 2b5eb048c75d37a6815cfc2770ca87dcaba1729b72779f1557c4f4d13b3ba75a | 5507076 |

Paper: M. Gonzalez-Franco et al., “The Rocketbox Library and the Utility of
Freely Available Rigged Avatars”, Frontiers in Virtual Reality, 2020.

Citadel adapts material handling at runtime only: native PBR channels are kept,
optional roster tint multiplies the native colour, and Marta's transparent hair
uses an alpha cutout of 0.35 in both colour and skinned normal/depth passes.
The files contain seven standing clips; they have no walking or sitting clips.
Do not use these casual outfits as uniformed officers or medical staff.

## Transferred activities

`joe-activities.json` and `marta-activities.json` are derived animation-only
bundles: walk, sitting idle, folded arms and leaning on a rail. The donor is
`Repos/island/public/assets/character/hero.glb` at Island revision
`3e9dfe7d191d28b4119605a661ac3dc09b160af3`, SHA256
`a91aa68cc3ace081e2ff8afe5d734dc627f3cbf9eab45458f5d46838c504e0c2`.
Island's `CREDITS.md` identifies the Universal Animation Library (Standard)
and Universal Animation Library 2 (Standard), by Quaternius, as **CC0 1.0**:
https://quaternius.itch.io/universal-animation-library and
https://quaternius.itch.io/universal-animation-library-2 .

The world-space/rest-direction transfer method is adapted from Island's
`tools/retarget.ts`, copyright (c) 2026 Prasenjit, MIT; the complete notice is
[LICENSE-Island.md](LICENSE-Island.md). Citadel adapts that method to two Biped
rigs, restores channels for transitions, scales by leg length, and corrects
root height from actual skinned sole vertices. Only animation data is derived;
the donor hero's mesh, skin and textures are not copied into this package.

Reproduce with `node tools/people/retarget-native.mjs /path/to/island/public/assets/character/hero.glb`.
The original Joe/Marta GLBs and their hashes above remain unchanged.
