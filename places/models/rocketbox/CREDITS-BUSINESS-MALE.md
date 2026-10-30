# Rocketbox professional man — Business_Male_04

`business-male-04.glb` is converted from Microsoft's non-facial
`Assets/Avatars/Professions/Business_Male_04/Export/Business_Male_04.fbx`
and its six `m015_{body,head}_{color,normal,specular}.tga` maps.
Official repository: https://github.com/microsoft/Microsoft-Rocketbox .
Pinned revision: `0943055db6ec570bcef9f2c8b41c9e5467c808f9`.
Copyright (c) 2020 Microsoft. MIT: [LICENSE-Rocketbox.md](LICENSE-Rocketbox.md).
Every shipped source file has its byte count and SHA256 in
[business-male-04-source.json](business-male-04-source.json).

The seven official Business_Male preview images were reviewed. 04 was chosen
for its original grey shirt/dark waistcoat, without a hat or uniform insignia.
No costume was painted over and no geometry was generated. It is a civilian
professional suitable for a director's bartender interpretation; no claim is
made that Microsoft labeled the original asset as a bartender.

The original 80-bone skeleton, 7,326 triangles, skin weights and proportions
remain. The body/head maps are reduced to 1024px JPEG. Normal maps remain
normal data; roughness is approximated as `.92 - .6 * source specular` and
metalness is zero. This is a conversion from a specular library, not a measured
metallic-roughness scan. No facial morphs or speech lip sync are supplied.

## Original motions and limited editing

All nine motion sources are **Microsoft male FBXs** under
`Assets/Animations/all_animations_max_motextr_static/m_<name>.max.fbx`.
They are baked into the target's own native Biped rig in world orientation:

- idle_neutral_01, idle_breathe_01, idle_look_around_01;
- gestic_talk_neutral_01, gestic_talk_relaxed_01, gestic_shrug_01, wave_01;
- sit_chair_idle_neutral_01;
- idle_neutral_08, from which `folded_native_loop` is edited.

The folded loop uses source seconds **24.6–27.8**, at 30 Hz. Its final 0.6 seconds
smoothly crossfade back to the first source pose, preserving a continuous
crossed-arm pose. This is an edited loop, not a separate captured performance.
The seated clip raises only the pelvis **12 mm** to correct source shoe-floor
penetration. It does not stretch limbs or move the chair. Its measured underside
is about 0.451 m; a fixture seat top of 0.45 m at local z=-0.04 was reviewed.
Scene-specific chair placement and transitions still need scene verification.

No Quaternius/Island activities are used by this profile. Walk, lean, phone,
seated talk and path/attend choreography are deliberately unsupported and cause
the existing explicit legacy fallback. Do not give this stationary profile a
path or attend configuration and describe it as native locomotion.

The FBX converter is adapted from Tidewater's MIT implementation:
[LICENSE-Tidewater.md](LICENSE-Tidewater.md), copyright 2026 DRG Software
Solutions LLC. `pack-business-male.mjs` performs the documented clip selection,
loop editing and pelvis correction. Existing shared converter behavior is unchanged.

## Reproduction

From repository root, with Blender 4.5, Python Pillow and project Node dependencies:

```sh
python tools/people/fetch-rocketbox-source.py public/places/models/rocketbox/business-male-04-source.json /tmp/business-male-source
python tools/people/prepare-rocketbox-textures.py /tmp/business-male-source /tmp/business-male-textures m015
blender -b --python tools/people/convert-rocketbox.py -- /tmp/business-male-source/Business_Male_04.fbx /tmp/business-male-textures m015 /tmp/business-male-all.glb /tmp/business-male-source/m_*.fbx
node tools/people/pack-business-male.mjs /tmp/business-male-all.glb public/places/models/rocketbox/business-male-04.glb
node --test src/world/people/profiles.test.js
```

`tools/people/business-male-preview.html` uses actual addCast and the room Post
pipeline. `?direct` is only a raw converted-clip diagnostic. Runtime photographs
and measurements are recorded separately from source previews in the reconstruction
run's `professional-men` evidence directory; they do not establish Bar scene acceptance.
