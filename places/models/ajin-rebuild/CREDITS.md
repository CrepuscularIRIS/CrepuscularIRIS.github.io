# Ajin kitchen and street sources

New native assets downloaded from official Poly Haven on 2026-10-29, CC0 1.0 (https://polyhaven.com/license):

- **Folding Wooden Stool**, Ulan Cabanilla — https://polyhaven.com/a/folding_wooden_stool . Original 6,324 triangles, wood grain and hardware retained; source also referenced by MIT `Repos/island` (CREDITS.md). This package uses the official 2K glTF, not Island's smaller derivative.
- **Pot Enamel 01**, Kuutti Siitonen — https://polyhaven.com/a/pot_enamel_01 . Original 11,416 triangles and authored patterned enamel retained. It is a small pot, not a stockpot; scene uniformly scales it 1.25×.

Only transforms and texture packaging changed: floor-centered native metre bounds, packed ARM AO enabled, WebP quality 95 at 1K, unchanged geometry. Exact official URLs, MD5 checks, SHA256 values, output hashes and dimensions are in manifest.json. Added payload is 1,853,840 bytes; assets load only when Ajin's streetKit catalog is requested.

The combined catalog refers to existing assets without copying them: aircon by Monsta3D; cardboard box by Rahul Chaudhary; trash bag by Benny Weimer; plastic crate by Pacific Penguin; utility box by James Ray Cock. These remain Poly Haven CC0 with complete source records in `../street-rebuild/CREDITS.md` and `manifest.json`.

Six near facade cells reuse Neon Zenith's procedural kit: Bill Chan (MIT adapter), udayjeet's **Procedural Hong Kong building** (CC BY 4.0), https://sketchfab.com/models/528a732e84c44fd49c4726f341014a23 . Exact source revision and file hashes are retained in `../street-rebuild/neon/manifest.json`; source license in `../street-rebuild/LICENSE-Neon-Zenith.txt`. Cells are unscaled, 3.1m modules, placed on two existing Ajin facades. No complete street, baked city lighting or simulated rooms copied.

Bespoke counter joinery, sliding door, cloth folds, bowls, vending construction and layout are authored for this project. Existing CC0 ambientCG surfaces follow `lab/assets/cc0/LEDGER.json`; paper lanterns reuse the existing Izakaya helper. They are not represented as scanned assets.
