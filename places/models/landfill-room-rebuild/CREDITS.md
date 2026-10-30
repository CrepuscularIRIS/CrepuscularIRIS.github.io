# Landfill workshop rebuild

Native CC0 geometry and PBR texture sources from Poly Haven. The original metre geometry and UVs are retained, floor-centered by a wrapper node. Embedded maps are 1K WebP quality 95, ARM red is wired to occlusion. No asset generation service or external unlicensed game content is used.

| Local ID | Original source | Artist | Change |
|---|---|---|---|
| desk | [metal_office_desk](https://polyhaven.com/a/metal_office_desk) | Ulan Cabanilla | Floor anchor, embedded maps; uniform runtime 1.142857 scale places tabletop at .900 m |
| drill | [old_drill_press](https://polyhaven.com/a/old_drill_press) | Harsh Vardhan Singh | Floor anchor, embedded maps; .85 uniform runtime scale |
| tool-case | [metal_toolbox](https://polyhaven.com/a/metal_toolbox) | Mateusz Sadek | Original open-lid pose, floor anchor, embedded maps |
| carton | [cardboard_box_01](https://polyhaven.com/a/cardboard_box_01) | Rahul Chaudhary | References existing street-rebuild file; no duplicate payload |
| crate | [plastic_crate_03](https://polyhaven.com/a/plastic_crate_03) | PacificPenguin | References existing street-rebuild file; no duplicate payload |
| lf-compressor | [old_military_compressor](https://polyhaven.com/a/old_military_compressor) | Brian Speight | References exterior agent's landfill-rebuild file, original PBR; no duplicate payload |

[Poly Haven CC0 license](https://polyhaven.com/license). `manifest.json` retains source URLs, source MD5/SHA-256, artist records, native counts/bounds, and output hashes. Its `file` paths resolve against the street-kit loader base `places/models/street-rebuild/`, not against this directory.

Dependencies: native street-kit runtime and catalog parameter, `public/places/models/street-rebuild/{carton,crate}.glb`, and `public/places/models/landfill-rebuild/lf-compressor.glb` from the Landfill exterior reconstruction. Only referenced assets load. No facades or shops are used.

Reproduce: `python tools/landfill-room-rebuild/download.py /tmp/citadel-landfill-room-sources`, then `node tools/landfill-room-rebuild/import.mjs /tmp/citadel-landfill-room-sources`. The exterior catalog must be installed to reproduce its reference. Test with `node --test tools/landfill-room-rebuild/assets.test.mjs`.

Existing AmbientCG Planks014 floor and concrete wall maps retain their ledger in `lab/assets/cc0/LEDGER.json`. New authored folded cabinet geometry, drawer labels and rack braces are project code. Atlas cards `weathering-follows-form`, `networks-link-industrial-masses`, `framed-distant-view` informed edge wear, articulated machinery and the retained window frame. Offshore Tower was visually inspected only: no license was found, so none of its code/assets is copied.

Limits: the cabinet wall and radio remain authored project geometry; this is a bounded workshop upgrade, not a claim of fully photogrammetric or film-quality reconstruction. The narrow frosted casements still constrain exterior detail. Original far heap/crane silhouettes remain outside.
