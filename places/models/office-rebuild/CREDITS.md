# Office reconstruction assets

All four model sources are Poly Haven assets licensed **CC0 1.0 Universal**. Their geometry and texture maps may be reused and redistributed under that dedication. Verified against each asset page and [Poly Haven's official asset license](https://polyhaven.com/license); license text: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en).

| Local model | Original asset | Artist | Intended use |
|---|---|---|---|
| desk.glb | [Metal Office Desk](https://polyhaven.com/a/metal_office_desk) | Ulan Cabanilla | Worn steel municipal-office desk, real drawer handles and edge details |
| lamp.glb | [Desk Lamp Arm 01](https://polyhaven.com/a/desk_lamp_arm_01) | Kuutti Siitonen (model/textures), Yann Kervran (rig) | Orange articulated clamp lamp, springs and visible hardware |
| laptop.glb | [Classic Laptop](https://polyhaven.com/a/classic_laptop) | Arrangemonk | Optional retro desk computer; unsuitable as a modern slim computer replacement |
| binder.glb | [Binder Notebook](https://polyhaven.com/a/binder_notebook) | DaDrood | Open leather notes binder with modeled paper, pen and rings |

Changes: downloaded official2K glTF set; preserved native color, normal and metallic/roughness texture response; recovered the AO channel in the supplied ARM images; 2K color images compressed to WebP quality95, normal/ARM resized to1K and stored as lossless WebP; geometry meshopt-quantized/compressed without simplification; declared anchor translations; binder's duplicate closed variant removed. Source lamp's natural clamp anchor retained. The import does not add light sources or replace the laptop's blank screen with invented source content.

`manifest.json` records each source file's exact official download URL, MD5 verification, SHA-256, output SHA-256, authors, modifications, metre bounds and triangle count. These are actual native-source models, not extractions from the project's aggressively reduced dressing bundles. Their existing reduced versions remain untouched.

The source art uses an intentionally aged office vocabulary. The loader preserves it; choosing which assets fit the room, replacing old visual proxies, providing practical light and preserving interaction/collision anchors belongs to scene integration. No claim of final cinematic scene acceptance is made by this asset package.
