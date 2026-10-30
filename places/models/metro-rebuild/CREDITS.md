# Metro reconstruction sources

## Native mesh integration

- **korean_fire_extinguisher_01**, UM JOORIN, Poly Haven. CC0 1.0.
  - https://polyhaven.com/a/korean_fire_extinguisher_01
  - https://polyhaven.com/license
  - API: https://api.polyhaven.com/files/korean_fire_extinguisher_01 and `/info/…`.
  - Official 2K glTF geometry retained: 9,913 triangles; three native material groups; original body, labels, gauge, hose and metal stand. Retains source Korean text; not relabeled as a local-language replica.
  - Offline conversion embeds 1K WebP maps, preserves base-color / normal / metallic-roughness and connects the source ARM red channel as AO. Floor-centered metre anchor, height 0.65925 m. Two shared instances; 2,143,632 bytes once per scene.
  - Every input URL, source MD5, input SHA256 and output SHA256 is in manifest.json. File paths are relative to `places/models/street-rebuild/`, the existing loader base. No runtime source downloads.

## Existing CC0 material integration

Existing project AmbientCG originals: `Tiles141` (floor and lower wall), `Concrete048` (upper wall and tunnel), `Concrete034` (wall skirting), `Gravel043` (track ballast), `Metal012` (cab skin), `Leather037` (driver seat). Their exact source/download/license records remain in `public/places/pbr/LEDGER.json`; the relevant JPEG maps are reused without duplicate copies. Tiles141 comprises six tiles across: floor repeat 3.6 m gives 0.6 m tiles; wall repeat 1.8 m gives 0.3 m tiles. Concrete and ballast UVs use metres.

## Reconstruction and visual reference boundary

The full train is **not** a newly sourced native model. Its existing shell is retained, with an open cab face, transparent glazing, driver controls/chair/partition and reallocated front passenger bay. The three perforated bent-metal station seats and extended tunnel are authored scene geometry. They are not attributed to Poly Haven.

Actually viewed: baseline door/ref/reverse/bench pictures; atlas `colonnade-rhythm/shots/01.jpg`, `mirror-beside-roughness/shots/05.jpg`; `references/grok/2026-10/space-train-platform-t08.jpg`, `space-train-vault-t14.jpg`, and `edolus-compute-core.jpg`. These inform repeated depth frames, localized roughness/reflection, and contrast hierarchy. These images are reference only; no image or geometry is copied into the product. Source-only inspection of MIT `Repos/train-track-builder` and `Repos/edolus-replica` did not establish a native modern subway asset; their promotional descriptions are not quality evidence.

Rejected after viewing official previews: Poly Haven painted_wooden_bench (antique wood geometry mismatches the established station seating/poses), metal_trash_can (excessive exterior corrosion mismatches the managed platform). Other generic low-poly/unverified-license train results were not imported. A fully detailed licensed modern subway car remains an explicit gap.
