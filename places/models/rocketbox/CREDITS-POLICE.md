# Rocketbox professional officer

`police-male-03.glb` is derived from Microsoft's **Police_Male_03**, the
non-facial FBX under `Assets/Avatars/Professions/Police_Male_03/Export/`.
Official source: https://github.com/microsoft/Microsoft-Rocketbox , pinned revision
`0943055db6ec570bcef9f2c8b41c9e5467c808f9`.
Copyright (c) 2020 Microsoft; MIT: [LICENSE-Rocketbox.md](LICENSE-Rocketbox.md).
The license was fetched at this revision and is byte-identical to the notice here.

[police-male-03-source.json](police-male-03-source.json) records every source path,
SHA256, byte count, conversion choices and derived GLB hash. These are source
FBX/TGA hashes, not merely hashes of downloaded previews. Unused `m156` maps
also present in the official directory were not included. The mesh actually
references `m172`, including the misspelled `m172_equpiment` material.

Conversion preserves the native 80-bone skin and four material groups:
body, head, equipment, pistol. 1024px JPEG colour and normal maps are retained;
roughness is approximated from the original specular texture (`.92 - .6 × s`),
metallic is zero. This is a realtime PBR conversion, not physically measured
material data. No geometry is synthesized and no costume is painted over a
casual body. The official uniform includes a holstered sidearm.

Seven original Microsoft animations from `Assets/Animations/all_animations_max_motextr_static/`
are retargeted and baked: `m_idle_neutral_01.max.fbx`, `m_idle_breathe_01.max.fbx`,
`m_idle_look_around_01.max.fbx`, `m_gestic_talk_neutral_01.max.fbx`,
`m_gestic_talk_relaxed_01.max.fbx`, `m_gestic_shrug_01.max.fbx`, `m_wave_01.max.fbx`.
The conversion method and script are adapted from Tidewater's
`tools/characters/convert.py`, revision `4811ba48d795197de5621985f404e765c0b7c0ef`,
Copyright (c) 2026 DRG Software Solutions LLC, MIT:
[LICENSE-Tidewater.md](LICENSE-Tidewater.md).

`police-male-03-activities.json` separately transfers walk, sit, folded and lean
from the Island donor already documented in [CREDITS.md](CREDITS.md#transferred-activities):
Island transfer implementation MIT; original Quaternius animations CC0 1.0.
The donor mesh and textures are not copied. Transfer is calibrated to this
specific target using rest directions, leg length and actual skinned soles.

The character has a fixed face, no lip sync and no facial expression controls.
A cap and dark uniform suit a patrol officer; this asset does not claim to match
any particular jurisdiction, the existing female desk sergeant, medical staff,
or every professional role. Source female Police_Female_01 was reviewed and
rejected for this office use because it wears riot armor and a full helmet.
