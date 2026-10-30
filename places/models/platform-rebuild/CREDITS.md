# Platform station assets

New full-PBR models from the official Poly Haven CC0 library, downloaded 2026-10-29:

- `pf-bench.glb`: [Modular Street Seating](https://polyhaven.com/a/modular_street_seating), Stuart Attenborrow. Complete straight bench retained; loose curved connectors and extension bench omitted. Source support positions, 0.45m seat height, timber, metal and armrest materials retained. 2.452m wide, 0.670m deep, 0.867m high.
- `pf-phone.glb`: [Korean Public Payphone 01](https://polyhaven.com/a/korean_public_payphone_01), UM JOORIN. Original text, worn finish, handset and cord retained. Native 0.740m height; no centimetre conversion or language replacement.
- `pf-cabinet.glb`: [Drawer Cabinet](https://polyhaven.com/a/drawer_cabinet), Ulan Cabanilla. Native 1.881m height and full assembly retained.

[Poly Haven asset license](https://polyhaven.com/license): CC0-1.0. This import retains original geometry without simplification, source transforms, 2K base color, 1K normal/packed ARM, AO from ARM red, and lossless noncolor WebP. Input URLs, MD5/SHA256 and output hashes, metre bounds and transformations are in `manifest.json`. Import with `node scripts/platform-rebuild/import.mjs`; verify with `node scripts/platform-rebuild/validate.mjs`. Preview images and Poly Haven website text are not shipped.

The scene catalog references existing assets without copying them:

- Office desk / articulated lamp / classic laptop / binder: `../office-rebuild/`, CC0. See that package's CREDITS.md and manifest for respective creators and retained files.
- Door and roller shutter: `../street-rebuild/`, CC0. See that package's credits and manifest.
- Native facade cells: Neon Zenith, Bill Chan, source commit `27ad04ef04aa9e1e1366fccc63e15a379b999f01`, derived from [Procedural Hong Kong building](https://sketchfab.com/models/528a732e84c44fd49c4726f341014a23) by udayjeet, CC BY 4.0. See `../street-rebuild/neon/manifest.json`. Source geometry and native PBR used; simulated rooms and city baked light omitted. These are visible street scenery, not colliders.
- Existing ambientCG CC0 maps: Tiles141, Tiles139, Marble012, Bricks060, PaintedPlaster017, Concrete034, Concrete048 and Metal012; per-set provenance in `public/places/pbr/LEDGER.json`. Metric UV projection is local Platform code.

The atlas train-platform still supplied architectural depth/rhythm inspiration only; no pixels or meshes are redistributed. Gate geometry remains the project's original authored model; a licensed high-quality native turnstile source has not been found. This package does not claim the gates or characters have been replaced by scanned assets.
