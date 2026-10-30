# Bar Cosmos native assets

- [Wine Bottles 01](https://polyhaven.com/a/wine_bottles_01): Rico Cilliers (modeling), Jurita Burger (graphic design), Poly Haven, **CC0 1.0**. Four authored bottles, original label, foil and outer-glass geometry. Split into separate bottle files and translated to floor-centred metre anchors; embedded maps compressed to 1024px WebP. Filled-bottle WebGL approximation: hidden inner liquid primitives removed; native outer shell, labels, foil and maps retained with opaque dark bottle-glass tints. This avoids nested single-buffer transmission washing the wine into cyan/cream under cabinet lights; it does not reproduce transparent liquid optics. Labels belong to the source artwork, not endorsements.
- [Bar Chair Round 01](https://polyhaven.com/a/bar_chair_round_01): Dairon Sanchez, Poly Haven, **CC0 1.0**. Native worn wooden stool, full geometry retained, floor-centred metre anchor, 1024px WebP PBR maps. Central seat surface measures 0.689m above the floor; total height including the back guard is 0.75055m.
- Near facade modules: udayjeet, Procedural Hong Kong building, **CC BY 4.0**, as supplied by neon-zenith (Bill Chan). See [street kit credits](../street-rebuild/CREDITS.md) for original URLs, attribution and modifications.
- Ground-floor doors/shutters reuse existing files: abhayexe, Asian Shop Pack, **CC BY 4.0**, and Poly Haven contributors, **CC0 1.0**. See [alley kit credits](../alley1-rebuild/CREDITS.md). No duplicate source asset files are shipped.

Source download URLs, verified MD5/SHA256, authors, dimensions, output hashes and geometry counts are recorded in manifest.json. Rebuild: `python3 tools/bar-rebuild/download.py CACHE` then `node tools/bar-rebuild/import.mjs CACHE`.

Atlas goblet and robot-workshop images were inspected as material/light references only. Their media are not shipped in this asset package. No offline cinematic render quality or physically exact glass caustics is claimed for the live web scene.
