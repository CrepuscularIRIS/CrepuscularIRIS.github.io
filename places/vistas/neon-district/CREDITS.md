# Neon Zenith district transfer

Upstream: [billpwchan/neon-zenith](https://github.com/billpwchan/neon-zenith), commit `27ad04ef04aa9e1e1366fccc63e15a379b999f01`. Source code: MIT, copyright 2026 Bill Chan; full notice in LICENSE-Neon-Zenith.txt. Full original asset ledger is preserved in CREDITS-source.md.

This subset contains the authored Temple Street district geometry and HDR diffuse light atlas, 14 Poly Haven building surface sets at 1K, the street irradiance field, road and sidewalk material maps, and the Shanghai Bund HDR environment. It additionally contains the eight genuine Asian Shop Pack storefront modules and their original material texture arrays, plus a pruned CC0 prop subset. It does **not** contain vehicles, vending machines, full font programs, or the unfiltered source props GLB. Geometry, lightmaps and 1K building textures are copied without modification. Street maps were resized to 1024px; their RAO channels were repacked to AO/roughness/zero-metalness. The file manifest records every input and output SHA-256.

## Asset attribution

Poly Haven surfaces and Shanghai Bund HDRI are CC0 public domain. Building sets: concrete wall 009, dirty concrete, dirty tiles, grey tiles, painted metal shutter, painted plaster wall, peeling painted wall, precast concrete wall, rectangular facade tiles 01 and 02, rusty metal 02, rusty metal shutter, square tiled wall, worn corrugated iron. Road and pavement materials: asphalt 02 and checkered pavement tiles. Props: cardboard box 01, plastic crate 01, trash bag, exterior aircon unit and its rusted variant. Only middle LOD1 and coarse LOD2 are retained; high LOD0 and all non-CC0 vending content are removed and unused buffers/textures pruned. [Poly Haven](https://polyhaven.com).

For the upstream district composition and any derived facade elements, preserve the original facade-pack credits and modifications:

- [Procedural Hong Kong building](https://sketchfab.com/models/528a732e84c44fd49c4726f341014a23), udayjeet, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), upstream sliced cells/captured facade textures.
- [Full Gameready City Buildings III HongKong](https://sketchfab.com/models/03a5255bb9624d4e82cb8e15e0070b70), abhayexe, CC BY 4.0, upstream sliced cells/captured facade textures.
- [Asian Shop Pack Free Gameready](https://sketchfab.com/models/2f603ad8d5c640f7b9794f6916f9fbdd), abhayexe, CC BY 4.0, upstream sliced modules. Eight shop modules and material texture arrays are included. The WebGL adapter retains source material-layer indices, distinct UV sets, roughness/metalness, and open-shop emission.

No exclusive ownership of these assets is claimed. Redistribution must retain applicable attribution and license notices. Neon sign and typeface attributions remain in CREDITS-source.md; no standalone font or sign atlas is included here.

Basis transcoder files are copied from Three r170's distribution. Its embedded license notice and basis/README.md are retained; Three's MIT notice is LICENSE-three.txt.

## Runtime adaptation

Citadel's adapter preserves quantization transforms and PBR channels, ports the nighttime baked-diffuse separation to WebGL standard materials, adds local-coordinate road/pavement geometry, and switches to host scene lighting during daytime. The shader and road assembly are adaptations of the MIT source. HDR lightmaps are linear EXR-derived radiance; no old logarithmic PNG decode is applied. Runtime reflection highlight clamping follows upstream.
