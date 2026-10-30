# Rocketbox professional woman

`business-female-01.glb` derives from the non-facial **Business_Female_01** FBX,
`Assets/Avatars/Professions/Business_Female_01/Export/Business_Female_01.fbx`,
and the seven `f014_*` TGA maps in that avatar's Textures directory.
Official source: https://github.com/microsoft/Microsoft-Rocketbox , revision
`0943055db6ec570bcef9f2c8b41c9e5467c808f9`. Copyright (c) 2020 Microsoft;
MIT notice: [LICENSE-Rocketbox.md](LICENSE-Rocketbox.md).

Full source paths, SHA256 and byte counts:
[business-female-01-source.json](business-female-01-source.json).
Source texture dimensions are reduced to 1024px. Colour/normal are JPEG;
roughness is approximated as `.92 - .6 * source specular`, metalness is zero;
hair alpha remains PNG and uses the runtime's shared .35 cutout. The original
80-bone rig and body/head/hair material separation are preserved. There is no
synthesized costume, and this suit has no police or jurisdiction insignia.
It is suitable for a professional civilian/office interpretation chosen by the
director, not a claim that a business outfit is a statutory police uniform.

Seven original female animations are baked from
`Assets/Animations/all_animations_max_motextr_static/f_<clip>.max.fbx`:
idle_neutral_01, idle_breathe_01, idle_look_around_01, gestic_talk_neutral_01,
gestic_talk_relaxed_01, gestic_shrug_01, wave_01.
The converter is adapted from Tidewater's MIT implementation as credited in
[CREDITS-POLICE.md](CREDITS-POLICE.md); complete notice
[LICENSE-Tidewater.md](LICENSE-Tidewater.md).

`business-female-01-activities.json` contains the independently retargeted
walk/sit/folded/lean, with source attribution described in
[CREDITS.md](CREDITS.md#transferred-activities): Island implementation MIT,
original Quaternius motions CC0. Its additional **sitTalk is a composition**,
not a newly sourced seated recording: the measured seated lower body is kept,
while this target's own Microsoft `gestic_talk_neutral_01` drives the Spine1
subtree. Both thighs are children of Spine in this Biped rig; Spine itself
must remain seated. World orientation is compensated at Spine1. The full
27.3667-second gesture sequence is sampled at 15Hz; no source gesture is
replaced by generated motion. The bundle records the target GLB SHA256.

Business_Female_02/03/04 and Security_Female_01 previews were reviewed as
alternatives. Only 01 is included. In particular SECURITY markings were not
changed to pretend to be police. This release does not include audio lip sync
or facial morph controls, and retains visible layering in the source hair cards.
