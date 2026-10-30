# Alley1 native assets

Official Poly Haven CC0 models, downloaded 2026-10-29 using public glTF API. [License](https://polyhaven.com/license). Asset-specific original URLs/MD5/SHA256, output hashes, metre bounds and transformations are in `manifest.json`.

- [Rollershutter Door](https://polyhaven.com/a/rollershutter_door), MP. Only plain variant retained. **Single door is 1.08 × 2.40 m**; the website's 3.08 m width encloses two side-by-side variants. Two unscaled doors are used in Alley1. 2K colour, normal and ARM maps.
- [Painted Wooden Cabinet 02](https://polyhaven.com/a/painted_wooden_cabinet_02), Kirill Sannikov. Full authored geometry, 2K colour and 1K normal/ARM.
- [Worn Metal Rack](https://polyhaven.com/a/worn_metal_rack), Luca B. Full authored geometry, 2K colour and 1K normal/ARM.
- [Barrel 03](https://polyhaven.com/a/barrel_03), Serhii Khromov. Full authored geometry, 2K colour and 1K normal/ARM.

`a1-door.glb` derives from [Asian Shop Pack Free Gameready](https://sketchfab.com/models/2f603ad8d5c640f7b9794f6916f9fbdd), abhayexe, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), via Neon Zenith's `S_door`, source commit `27ad04ef04aa9e1e1366fccc63e15a379b999f01`. Lower doorway whole triangles below 2.35 m are retained, source projecting sign-kind triangles excluded; the inset Chinese shop lettering on the native doorway remains. Native width, depth and UV sets remain; source texture-array layers are repackaged as standard glTF PBR, with normal green inversion matching the source shader. Source irradiance/emission is not applied. It retains the source 1K layer resolution and is not claimed to be a new high-resolution scan. The surrounding scene wall covers the selection boundary. All source hashes and modification details are in the manifest. Upstream notices remain in `../street-rebuild/LICENSE-Neon-Zenith.txt` and `../../vistas/neon-district/CREDITS-source.md`.

The other native AC, toolbox, cabinet, cartons/crates/bags and facade cells are reused from `../street-rebuild/`; their credits remain with that payload. No Repos source was modified.

Regenerate: `node scripts/alley1-rebuild/import.mjs` then `node scripts/alley1-rebuild/import-door.mjs`. Each script adds only its own IDs to the shared street manifest. `import.mjs` refreshes the local asset manifest before the door extraction adds its own record.
