# Port of Barcelona twin: sources

Built by `envs/berth_planning/tools/twin/build_twin.py` (`fetch`, then `build`). Frame: UTM zone 31N metres minus
the origin in `twin.json.gz` (`frame`).

| File | What | Source and licence |
|---|---|---|
| `twin.json.gz` | coastline / land, quay walls, breakwaters, land cover, roads, 9,247 buildings with heights, 582 tanks, towers, BEST's 34 stacking blocks, APM's yard slabs, rail, the two quay lines | © OpenStreetMap contributors, [ODbL 1.0](https://www.openstreetmap.org/copyright) (Overpass API, data of July 2026) |
| `terrain.png` | ground height on a 40 m grid (Montjuïc, the city, Collserola) | [Terrain Tiles on AWS](https://registry.opendata.aws/terrain-tiles/) (Mapzen terrarium, z13; SRTM, EU-DEM and others, see their attribution page) |
| `cover.png` | land-cover class per 20 m cell, rasterised from the OSM land cover above | © OpenStreetMap contributors, ODbL 1.0 |

The quay lines come from the OSM container-crane positions snapped to the coastline (BEST 1,526 m at bearing 37.6°,
APM's south-east face 1,162 m at 27.0°). Ship routes and the anchorage were read off a Copernicus Sentinel-2 L2A scene
(17 June 2026, tile 31TDF, contains modified Copernicus Sentinel data 2026); no imagery is shipped. Crane, yard and
livery colours come from published terminal specifications and photographs (see the project README).

## Surrounding city and street detail

`scenery.json.gz` supplements the original twin with **37,666** additional mapped building footprints,
**24,945** road/path polylines (replacing the original major-road-only layer), and **290** surface-parking
polygons. Source: © OpenStreetMap contributors, [ODbL 1.0](https://www.openstreetmap.org/copyright),
Overpass API snapshots **6 October 2026**, 08:24:49 UTC (buildings) and 08:26:58 UTC (streets/parking).
Building heights use OSM height/level tags where present and the original twin's type-based estimates otherwise.
Duplicate and overlapping relation/way building footprints are removed during generation.

`surface.webp` is a 4096 × 4096 georeferenced **generated cartographic material**, built from those mapped
footprints, roads, the original OSM land cover and the existing Terrain Tiles elevation raster. It is not
satellite imagery. It retains ground detail beyond the 3D geometry's draw distance. Unclassified uplands receive
a height-based vegetation tint. Roadside trees, parking-bay paint and parked vehicles are illustrative details;
their placement excludes mapped buildings, carriageways and water.

Rebuild from the worktree (the ignored `.cache/barcelona-scenery` under `07-simulation-environments/portsim-v1` stores raw responses):

```sh
uv run --with numpy --with pillow --with shapely --with pyproj python envs/berth_planning/tools/twin/build_scenery.py
```

The original coastline, quay frames, terrain raster, navigation masks and RL task data are unchanged.
