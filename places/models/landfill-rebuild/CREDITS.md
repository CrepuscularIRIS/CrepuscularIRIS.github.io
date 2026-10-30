# Landfill native assets

All new models are Poly Haven CC0 1.0 assets; no third-party redistribution/EULA was accepted. Original files were downloaded from the official API/CDN. License: https://polyhaven.com/license

- **lf-cart**: [industrial_storage_cart](https://polyhaven.com/a/industrial_storage_cart); authors: Jule Bielitz. Exact triangle/byte counts are recorded in manifest.json.
- **lf-compressor**: [old_military_compressor](https://polyhaven.com/a/old_military_compressor); authors: Brian Speight. Exact triangle/byte counts are recorded in manifest.json.
- **lf-stove**: [electric_stove](https://polyhaven.com/a/electric_stove); authors: Kuutti Siitonen. Exact triangle/byte counts are recorded in manifest.json.
- **lf-crane**: [overhead_crane](https://polyhaven.com/a/overhead_crane); authors: Timothy3D. Exact triangle/byte counts are recorded in manifest.json.
- **lf-tyre**: [old_tyre](https://polyhaven.com/a/old_tyre); authors: MP. Exact triangle/byte counts are recorded in manifest.json.
- **lf-rim**: [rusted_wheel_rim_01](https://polyhaven.com/a/rusted_wheel_rim_01); authors: John Hutcheson. Exact triangle/byte counts are recorded in manifest.json.
- **lf-generator**: [portable_generator](https://polyhaven.com/a/portable_generator); authors: James Ray Cock. Exact triangle/byte counts are recorded in manifest.json.
- **lf-pillows**: [throw_pillows_01](https://polyhaven.com/a/throw_pillows_01); authors: Serhii Khromov. Exact triangle/byte counts are recorded in manifest.json.
- **lf-sack**: [cement_bag](https://polyhaven.com/a/cement_bag); authors: PierreB3D. Exact triangle/byte counts are recorded in manifest.json.

`manifest.json` records original URLs, verified MD5, SHA256 of every input/output, exact metre bounds and transformations. Reproduce with `node scripts/landfill-rebuild/import.mjs`; verify with `node scripts/landfill-rebuild/validate.mjs`.

Original authored PBR is retained. The distant crane uses a 0.35 geometry reduction ratio and the small rim 0.30 (error tolerance 0.001); other geometry is retained. Color maps: 2K WebP Q95. Normal/ARM maps: lossless WebP 1K. The wheel rim’s authored dent morph at weight 1 is frozen before runtime instancing. Meshopt compression and floor-centred translation are applied. The cushion albedo is washed to muted grey (linear .30 +150 on grayscale) to match discarded cloth; original weave, normal and roughness are retained. No object is anisotropically stretched. Runtime instances own their textures and remain outside baked collision geometry.

Existing bag/carton/crate and a1-drum are referenced through the scene catalog; attribution stays in street-rebuild and alley1-rebuild. Site-built crane columns/footings/longitudinal supports use the existing CC0 Metal012/Concrete034 materials and are original connecting geometry, not part of the imported crane.

Rejected: Innerscene CC0 dump truck 8fbb9f8c was downloaded and rendered for evaluation, but its near-field material/detail quality did not meet this rebuild. It is not shipped. Original story utility truck and handler remain a documented quality gap.
