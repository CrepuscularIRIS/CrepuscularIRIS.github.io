# Street rebuild assets

Official Poly Haven CC0 sources, fetched from API glTF 2K on 2026-10-29; license https://polyhaven.com/license. Full input URLs, MD5 and SHA256 hashes and output transformations are recorded in manifest.json.

- aircon: [exterior_aircon_unit](https://polyhaven.com/a/exterior_aircon_unit), Monsta3D. 2k color; 1k normal and ARM.
- carton: [cardboard_box_01](https://polyhaven.com/a/cardboard_box_01), Rahul Chaudhary. 2k color; 1k normal and ARM.
- bag: [trashbag](https://polyhaven.com/a/trashbag), Benny Weimer. 2k color; 1k normal and ARM.
- crate: [plastic_crate_03](https://polyhaven.com/a/plastic_crate_03), Pacific Penguin. 2k color; 1k normal and ARM.
- utility: [utility_box_01](https://polyhaven.com/a/utility_box_01), James Ray Cock. 2k color; 1k normal and ARM.
- shelves: [steel_frame_shelves_01](https://polyhaven.com/a/steel_frame_shelves_01), James Ray Cock. 2k color; 1k normal and ARM.
- toolbox: [metal_tool_chest](https://polyhaven.com/a/metal_tool_chest), Yann Kervran, John Hutcheson. 2k color; 1k normal and ARM.

Native metre bounds and anchor corrections are recorded per asset; shelves glTF was 10x the official 2.1 m height and is uniformly corrected by 0.1. Carton/crate use conservative meshoptimizer simplification; realized triangle counts are recorded. The open toolbox keeps all original articulated parts in their authored pose. No downloaded model is represented as our original work.

Neon facade module: source Neon Zenith (Bill Chan, MIT) commit 27ad04ef04aa9e1e1366fccc63e15a379b999f01, derived from [Procedural Hong Kong building](https://sketchfab.com/models/528a732e84c44fd49c4726f341014a23) by udayjeet, CC BY 4.0. This package retains geometry, quantization transforms and native material attributes; WebGL shader uses host lighting, omitting source irradiance and simulated rooms. Source file hashes are in neon/manifest.json.

Optional shops reuse the already shipped public/places/vistas/neon-district/details/ payload and its CREDITS.md: Asian Shop Pack by abhayexe, CC BY 4.0. Do not redistribute it without those notices.
