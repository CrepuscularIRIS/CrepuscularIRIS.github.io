# Botanical props — sources and changes

Both models and all their texture maps are original Poly Haven assets released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), verified against the [official asset license](https://polyhaven.com/license) on 2026-10-29.

| Local file | Original asset | Artist |
| --- | --- | --- |
| potted_plant_02.glb | [Potted Plant 02](https://polyhaven.com/a/potted_plant_02) | Rico Cilliers |
| potted_plant_04.glb | [Potted Plant 04](https://polyhaven.com/a/potted_plant_04) | James Ray Cock |

These are complete potted ornamental plants. Neither asset is presented as common ivy or as a climbing vine. Model 04 is described by its source as a zebra haworthia succulent. Model 02 is kept under the author's generic asset name; no unsupported species identification is added.

Changes made for Citadel:

- Downloaded the official 2K glTF, binary geometry and referenced maps. Source URLs, advertised MD5s and SHA-256 receipts are recorded in `manifest.json`.
- Model 02's source material is `MASK`, but its base-color JPG has no alpha. The official separate `leaves_alpha` PNG was joined into the albedo alpha channel. Exported alpha bytes are verified against the original 2048×2048 single-channel mask.
- Preserved all leaf, stem and pot geometry. Only model 02's dense soil/gravel mesh was simplified with meshoptimizer (`ratio .12`, `error .001`, locked borders). The conservative error bound stopped at 47,666 soil triangles, from 53,930; total model triangles became 63,542 from 69,806. Geometry hashes verify the other parts stayed unchanged.
- Model 04 retains all 8,929 actual glTF triangles. The website's 6K/6,097 summary differs from the actual triangulated glTF; runtime budgets use the file count.
- Retained native base color, OpenGL normal and roughness maps. Model 04's authored packed ARM map also supplies ambient occlusion.
- Kept model 02 leaf albedo/normal at 2K; reduced its pot/roughness maps and model 04 maps to 1K, then encoded WebP quality 95. RGB compression is lossy; the alpha mask is unchanged. No generated painted leaves or material tinting.
- Runtime places the pot bottom on the requested surface while retaining its native pot centre and authored local transforms. The whole plant can be uniformly scaled/rotated by the caller.

The importer uses Poly Haven's public API only offline, with the identifying user-agent `CitadelBotanicalImporter/1.0`; deployed scenes load these self-hosted GLBs and make no live API calls. Attribution is appreciated by Poly Haven and retained here, although CC0 does not require it.
