# Restaurant native rebuild

## Dining chair

- **Dining Chair 02**, James Ray Cock, [Poly Haven](https://polyhaven.com/a/dining_chair_02), [CC0](https://polyhaven.com/license).
- Official 2K glTF + original binary topology; retrieved 2026-10-29. 22,013 triangles, full topology retained. Original PBR diffuse/normal/ARM converted to embedded 1K WebP quality95; original ARM also supplies occlusion. Floor-centered metric anchor only; no mesh simplification. Source URLs/MD5/SHA256 and output SHA256 in manifest.json.
- 0.433571 × 0.973378 × 0.576420 m. Backrest toward -Z, front toward +Z. Seat top at center ray (x0,z.03) is **.452844 m**, distinct from backrest height .973378 m. Measure with tools/restaurant-rebuild/measure-chair.mjs.

## Referenced upstream assets, no duplicated payload

- `wine-bordeaux` / `wine-burgundy`: Bar rebuild CC0 Poly Haven wine_bottles_01, Rico Cilliers (model), Jurita Burger (graphics). `../bar-rebuild/` source notice and manifest retain transformations and raw file hashes; dark glass approximation is inherited.
- `rest-plant`: botanical rebuild CC0 Poly Haven potted_plant_02, Rico Cilliers. Original leaf alpha/PBR retained; hidden soil cleanup as documented upstream. `../botanical-rebuild/` source notice and manifest.
- Native lake-side district is loaded separately from `env.vista`; upstream `public/places/vistas/neon-district/CREDITS.md`, MIT code and CC BY 4.0 architecture notices remain authoritative. No geometry scaled to fake a far skyline.
- Room entrance wood and quay stone use existing credited AmbientCG Wood051 / Bricks100 surface sets; no new texture copies.
- Existing restaurant-hero scanned croissant/tart/carved plate remain in their original bundle with existing ledger. Existing tart also replaces four low-poly dessert pies.

## References and limits

Actually viewed Repos/three-gpu-pathtracer/docs/kitchen.webp (Country Kitchen by Jay-Artist via Bitterli) and atlas gold-accents-matte-mass/01.jpg (Athenaion). Visual mechanisms only; no third-party screenshot or kitchen mesh copied into the game. Native authored geometry and real room screenshots are distinct evidence.

Eleven tables / five service states are unchanged. Cooked fish and steak remain the old faceted assets because no suitable licensed scan was verified; current delivery does not claim their reconstruction. Soup vessels/liquid are smooth authored metric geometry, not photogrammetry. Root follow-up replaces people with credited Rocketbox profiles and flowers with native EZ-Tree models.

## Root integration additions

- Table flower arrangements: white flower model from Daniel Greenheck's EZ-Tree demo, MIT (LICENSE-EZ-TREE.txt). Original topology, embedded textures and materials retained, stem-base pivot centered, uniformly scaled to18% for table arrangements. Raw and converted SHA256 in manifest. Reproduction: tools/restaurant-rebuild/import-flowers.mjs <ez-tree checkout>.
- Waterfront end walls:164 existing native facade cells from street-rebuild/neon, using the same CC BY4.0 notices in street-rebuild/CREDITS.md. Measured front wall bands fitted with3.1m cells, preserving district street and tower silhouettes; added as a separate runtime kit.
- Occupied chairs uniformly resized to measured .417/.430/.451m seated contacts; floor shadows and collision bounds use the same per-chair scale.
