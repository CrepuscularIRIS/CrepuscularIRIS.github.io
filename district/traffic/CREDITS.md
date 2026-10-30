# District traffic: attribution and changes

These four models and their textures are **CC BY 4.0**, not MIT or CC0.
Attribution is retained from [Neon Zenith CREDITS.md](https://github.com/billpwchan/neon-zenith/blob/27ad04ef04aa9e1e1366fccc63e15a379b999f01/CREDITS.md), revision `27ad04ef04aa9e1e1366fccc63e15a379b999f01`.

| File | Original model and author | License |
| --- | --- | --- |
| taxi.glb | [Toyota Crown Comfort (Hong Kong taxi)](https://sketchfab.com/models/a8f3ad22901643809ca72450ac3cc5f6), zxgod118 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| sedan.glb | [Generic passenger car pack](https://sketchfab.com/models/20f9af9b8a404d5cb022ac6fe87f21f5), comrade1280 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| minibus.glb | [toyota coaster 2017](https://sketchfab.com/models/b59e858adf6944ecba44bd4ce88acfa5), Emmetv96 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| bus.glb | [Hong Kong KMB Bus Game Ready](https://sketchfab.com/models/1e0d21d8fb1d426c80f65e11c11abdd7), softmind | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Neon Zenith re-exported/compressed these models for the web, baked custom material atlases, provided two LODs, and split the passenger pack into body styles. This import uses those derived GLBs, not the original authoring files. Source model and brand names belong to their owners; no endorsement is implied.

Citadel changes (2026-10-29):

- Relocated source albedo alpha (glass mask) to metallic/roughness atlas alpha; relocated source emission alpha (actual opacity) to albedo alpha. RGB pixels are unchanged, verified against per-channel source hashes in `manifest.json`.
- Re-encoded those two atlases as lossless PNG and re-exported GLB. Geometry triangle counts are unchanged.
- Runtime applies source node transforms, widens quantized geometry and grounds each LOD at Y=0. No scale retargeting.
- Adapted the source material's custom paint, glass, roughness, metalness and lamp logic to Three.js r170 WebGL. Near clearcoat, far standard shading. Source emission is multiplied by its authored `emax`.
- Cabin emission is night-only at 20% of the original value to avoid daylit white windows. Host environment/lights replace the original city-specific neon lightmap. No full traffic simulation, neon lightmap, far light-dot layer or collision code is included.

The adapted material code derives from [Neon Zenith traffic.js](https://github.com/billpwchan/neon-zenith/blob/27ad04ef04aa9e1e1366fccc63e15a379b999f01/src/life/traffic.js), MIT, Copyright (c) 2026 Bill Chan. Its complete license is copied in `LICENSE-neon-zenith.txt`. This code license does not relicense the models.

Redistribution must retain attribution, license links and an indication of changes. When integrating a public scene, also surface this attribution in the application's asset credits. Source and derivative SHA-256 values are recorded in `manifest.json`.
