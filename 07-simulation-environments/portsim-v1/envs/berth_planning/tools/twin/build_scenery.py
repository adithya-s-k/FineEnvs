"""Supplement the port twin with mapped city buildings, local streets and parking.

Run with: uv run --with numpy --with pillow --with shapely --with pyproj build_scenery.py
Raw Overpass responses are cached under this worktree's ignored .cache directory.
The original shoreline, terrain, quay frames and RL data are never changed.
"""
from __future__ import annotations
import gzip
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, LineString
from shapely.strtree import STRtree
from build_twin import OUT, _height, _rings, _q, geom_xy, BUILDING_KIND, ROAD_W

CACHE = Path(__file__).resolve().parents[4] / '.cache' / 'barcelona-scenery'
QUERIES = {
    'city': '''[out:json][timeout:150];(way[building](41.350,2.085,41.415,2.230);relation[building][type=multipolygon](41.350,2.085,41.415,2.230););out tags geom qt;''',
    'streets': '''[out:json][timeout:150];(way[highway~"^(motorway|trunk|primary|secondary|tertiary|residential|living_street|unclassified|service|pedestrian|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link)$"](41.285,2.095,41.415,2.230);way[highway~"^(footway|path|cycleway)$"](41.350,2.140,41.378,2.177);way[amenity=parking][parking!=underground][parking!="multi-storey"](41.285,2.095,41.390,2.215););out tags geom qt;''',
}
COLORS = {
    'urban': '#ACA89B', 'port': '#A5A49B', 'industrial': '#B3B2A7', 'residential': '#B8AC98',
    'commercial': '#ADA89C', 'rail': '#807A6E', 'dirt': '#BBA580', 'grass': '#7F8955',
    'park': '#798E57', 'forest': '#536844', 'scrub': '#7C8150', 'sand': '#D8C49B',
    'farm': '#A6A177', 'cemetery': '#A4A28B', 'apron': '#AEABA2', 'runway': '#727673',
}


def fetch(name, query):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f'{name}.json'
    if path.exists():
        return json.loads(path.read_text())
    for host in ['https://overpass-api.de/api/interpreter', 'https://overpass.kumi.systems/api/interpreter']:
        try:
            req = urllib.request.Request(host, data=urllib.parse.urlencode({'data': query}).encode(), headers={'User-Agent': 'BerthPlanningScenery/1.0'})
            with urllib.request.urlopen(req, timeout=180) as response:
                data = response.read()
            obj = json.loads(data)
            if obj.get('remark') or not obj.get('elements'):
                raise ValueError(obj.get('remark', 'Empty map response'))
            path.write_bytes(data)
            print(name, len(obj['elements']), 'mapped elements', flush=True)
            return obj
        except Exception as error:
            print(host, str(error), flush=True)
    raise RuntimeError(f'Could not fetch {name}; cached assets have not been changed')


def main():
    data = json.loads(gzip.decompress((OUT / 'twin.json.gz').read_bytes()))
    f = data['frame']
    terrain = np.asarray(Image.open(OUT / 'terrain.png')).astype(float)
    heights = (terrain[:, :, 0] * 256 + terrain[:, :, 1]) / 10 - 50
    def ground(x, y):
        j = np.clip((x - f['extent'][0]) / f['grid'], 0, heights.shape[1] - 1.001)
        i = np.clip((f['extent'][3] - y) / f['grid'], 0, heights.shape[0] - 1.001)
        ii, jj = int(i), int(j)
        u, v = j - jj, i - ii
        return max(0, heights[ii, jj] * (1-u)*(1-v) + heights[ii, jj+1]*u*(1-v) + heights[ii+1,jj]*(1-u)*v + heights[ii+1,jj+1]*u*v)

    responses = {key: fetch(key, query) for key, query in QUERIES.items()}
    old_polys = [Polygon(np.array(b[3:]).reshape(-1, 2)).buffer(0) for b in data['buildings']]
    old_tree = STRtree(old_polys)
    added, extra_polys, seen = [], [], set()
    for element in responses['city']['elements']:
        tags = element.get('tags', {})
        if tags.get('building') in ('no', 'construction', 'ruins') or tags.get('man_made') == 'storage_tank':
            continue
        for ring in _rings(element):
            poly = Polygon(ring).buffer(0).simplify(0.6)
            if poly.geom_type != 'Polygon' or poly.area < 30:
                continue
            signature = tuple(np.round(poly.bounds, 0))
            if signature in seen:
                continue
            seen.add(signature)
            if any(poly.intersection(old_polys[idx]).area > min(poly.area, old_polys[idx].area) * 0.6 for idx in old_tree.query(poly)):
                continue
            # OSM relations and their outer ways can describe the same building.
            # Bounds deduplication above handles identical footprints.
            ct = poly.centroid
            kind = BUILDING_KIND.get(tags.get('building'), 0)
            added.append([round(_height(tags), 1), round(ground(ct.x, ct.y), 1), kind] + _q(np.array(poly.exterior.coords)[:-1]))
            extra_polys.append(poly)

    # Remove overlapping relation/way footprints as well as identical bounds.
    extra_tree = STRtree(extra_polys)
    remove = set()
    for i, a in enumerate(extra_polys):
        for j in extra_tree.query(a):
            if j <= i:
                continue
            b = extra_polys[j]
            if a.intersection(b).area > min(a.area, b.area) * 0.7:
                remove.add(i if a.area < b.area else int(j))
    added = [b for i, b in enumerate(added) if i not in remove]

    roads, parking = [], []
    widths = {**ROAD_W, 'tertiary': 9, 'residential': 7, 'living_street': 5, 'unclassified': 7, 'service': 5,
              'pedestrian': 5, 'primary_link': 7, 'secondary_link': 6, 'tertiary_link': 5, 'footway': 2.2, 'path': 2, 'cycleway': 2.8}
    for element in responses['streets']['elements']:
        tags = element.get('tags', {})
        if tags.get('amenity') == 'parking':
            for ring in _rings(element):
                poly = Polygon(ring).buffer(0)
                if poly.geom_type == 'Polygon' and poly.area > 300:
                    parking.append(_q(np.array(poly.exterior.coords)[:-1]))
            continue
        if tags.get('tunnel') in ('yes', 'building_passage') or tags.get('covered') == 'yes' or str(tags.get('layer', '0')).startswith('-'):
            continue
        pts = geom_xy(element.get('geometry', []))
        if len(pts) < 2:
            continue
        w = widths.get(tags.get('highway'), 5)
        layer = int(tags.get('bridge') == 'yes' or tags.get('layer') in ('1', '2'))
        roads.append([w, layer] + _q(LineString(pts).simplify(1.2).coords))

    result = {'buildings': added, 'roads': roads, 'parking': parking,
              'source': {'provider': 'OpenStreetMap contributors', 'licence': 'ODbL-1.0',
                         'timestamps': {k: v.get('osm3s', {}).get('timestamp_osm_base') for k, v in responses.items()}}}
    (OUT / 'scenery.json.gz').write_bytes(gzip.compress(json.dumps(result, separators=(',', ':')).encode(), mtime=0))
    bake_surface(data, result)
    print(json.dumps({'added_buildings': len(added), 'total_buildings': len(data['buildings']) + len(added), 'roads': len(roads), 'parking': len(parking), 'source': result['source']}, indent=2), flush=True)


def bake_surface(data, scenery):
    """Georeferenced context at ~4 m/px, from the same footprints as the 3D geometry.

    This retains streets and roof grain beyond the detailed geometry's draw distance.
    It is generated cartographic material, not satellite imagery.
    """
    f = data['frame']; x0, y0, x1, y1 = f['extent']
    size = 4096
    # Retain the original twin's upland scrub/wood tint where detailed cover is
    # absent. City roofs, streets and mapped land-use polygons are painted over it.
    terrain = np.asarray(Image.open(OUT / 'terrain.png'), dtype=np.float32)
    heights = (terrain[..., 0] * 256 + terrain[..., 1]) / 10 - 50
    elevation = np.asarray(Image.fromarray(heights).resize((size, size), Image.Resampling.BILINEAR))
    wooded = np.clip((elevation - 90) / 140, 0, 1)[..., None]
    base = np.array([172, 168, 155]) * (1 - wooded) + np.array([102, 119, 79]) * wooded
    im = Image.fromarray(base.astype(np.uint8)); draw = ImageDraw.Draw(im)
    def pts(flat):
        return [((flat[i]-x0)/(x1-x0)*size, (y1-flat[i+1])/(y1-y0)*size) for i in range(0,len(flat),2)]
    def poly(ring, color):
        p = pts(ring)
        if len(p) > 2: draw.polygon(p, fill=color)
    for cls, polys in data['cover'].items():
        for rings in polys:
            poly(rings[0], COLORS.get(cls, COLORS['urban']))
    for ring in scenery['parking']:
        poly(ring, '#898C89')
    for road in scenery['roads']:
        width = max(1, round(road[0] / (x1-x0) * size))
        path = pts(road[2:])
        draw.line(path, fill='#C1BEB1', width=width+2, joint='curve')
        draw.line(path, fill='#697171' if road[0] > 3 else '#BCAB85', width=width, joint='curve')
    rng = np.random.default_rng(411)
    roofs = ['#B6A895', '#C6B6A0', '#A89F91', '#C2BCB0', '#B39A81']
    for b in data['buildings'] + scenery['buildings']:
        poly(b[3:], roofs[int(rng.integers(0, len(roofs)))])
    # Metre-scale mineral grain and broad weathering break up large slabs and hills.
    arr = np.asarray(im).astype(np.float32)
    noise = rng.normal(0, 1.5, (size, size, 1)).astype(np.float32)
    coarse = Image.fromarray(rng.integers(94, 106, (128,128), dtype=np.uint8)).resize((size,size), Image.Resampling.BICUBIC)
    arr = np.clip(arr * (np.asarray(coarse, dtype=np.float32)[...,None]/100) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(OUT / 'surface.webp', quality=88, method=6)


if __name__ == '__main__':
    main()
