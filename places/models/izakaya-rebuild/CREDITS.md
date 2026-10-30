# Nomiya / Izakaya native kitchen assets

The four files in this directory originate from **Poly Haven, CC0 1.0**:

| Source | Author | Use and transformation |
|---|---|---|
| [Wooden Cutting Board](https://polyhaven.com/a/wooden_cutting_board) | Kuutti Siitonen | Full 3,520-triangle native board; 45 × 24.7 × 4.13 cm. Preparation surface and night serving board. |
| [Fish Knife](https://polyhaven.com/a/fish_knife) | Mateusz Sadek | Full native blade/handle topology. Rotated 90° about X to lie flat on board; 21.6 cm length. |
| [Food Ginger 01](https://polyhaven.com/a/food_ginger_01) | Oliver Harries | Full scanned topology. Rotated 90° about X and uniformly scaled 0.3 from the 57 cm source to a 17.2 cm rhizome. |
| [Wooden Bowl 02](https://polyhaven.com/a/wooden_bowl_02) | Kuutti Siitonen | Native 13 cm burl bowl, only limited prep storage. Not substituted for porcelain Japanese tableware. |

Each is translated to a centred horizontal contact anchor at y=0 after orientation. Color, normal and roughness maps are embedded as 1024px WebP (quality95). ARM red is also used as AO for the three source ARM assets. Ginger has its original roughness map, no fabricated AO. No decimation or geometry reconstruction. All input MD5s, download URLs, input SHA256, authors, dimensions, triangle counts and output hashes are in manifest.json.

Reused without copying:
- Dairon Sanchez / Poly Haven [Bar Chair Round 01](https://polyhaven.com/a/bar_chair_round_01), CC0, `../bar-rebuild/bar-stool.glb`. See upstream CREDITS. Four counter stools at scale 0.65 / 0.689, retaining a 0.65 m central seat and 0.8 m spacing. The back guard faces the aisle, and feet remain on the floor.
- Poly Haven enamel pot, `../ajin-rebuild/pot_enamel_01.glb`; see Ajin credits and copied source ledger.
- Poly Haven plastic produce crate and external air conditioner, `../street-rebuild/`; see street-kit credits.
- udayjeet, Procedural Hong Kong Building, **CC BY 4.0**, supplied through Bill Chan's Neon Zenith. Five native relief facade cells, original geometry and source channel decoding via existing street-kit. See [street credits](../street-rebuild/CREDITS.md) for source URL, author, license and adapter changes. A bounded opposite frontage is authored specifically for this doorway, not the entire Neon city.
- AmbientCG `Wood051`, `Wood066`, `Leather037`, `Tiles107`, asphalt and plaster are existing CC0 project surfaces. Wood UVs are in metres; leather albedo is replaced by original green upholstery color while retaining scanned micro-normal and roughness. Their attribution lives in the project CC0 ledger.

Rebuild from the worktree root:

```
python tools/izakaya-rebuild/download.py /tmp/citadel-izakaya-sources
node tools/izakaya-rebuild/import.mjs /tmp/citadel-izakaya-sources
node --test tools/izakaya-rebuild/assets.test.mjs
```

Source references observed: atlas `cold-world-warm-refuge`, `framed-distant-view`, and Neon Zenith preview. Reference media are not shipped. Existing Kenney cooked food and procedural sake bottles/chairs remain fidelity gaps; no realistic sushi scan is claimed. Day preparation keeps empty active story supports, with day wording integration owned by the main task. No timetable change.
