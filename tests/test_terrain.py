"""Рельеф на синтетических сетках: уклон, экспозиция, горы, духи, жила."""

import math

import numpy as np
import pytest
from affine import Affine

from geosite import geo, terrain
from geosite.dem import DemGrid


def flat_grid(value: float = 100.0, n: int = 201, res: float = 100.0) -> DemGrid:
    data = np.full((n, n), value, dtype="float32")
    return DemGrid(data, Affine(res, 0, 0, 0, -res, n * res), "EPSG:32630", res)


def grid_from_height(fn, n: int = 201, res: float = 100.0) -> DemGrid:
    xs = (np.arange(n) + 0.5) * res
    ys = (n - np.arange(n) - 0.5) * res
    xx, yy = np.meshgrid(xs, ys)
    return DemGrid(fn(xx, yy).astype("float32"), Affine(res, 0, 0, 0, -res, n * res), "EPSG:32630", res)


def test_slope_and_aspect_on_plane_falling_south():
    # высота падает к югу (y уменьшается) → склон смотрит на юг, 180°
    g = grid_from_height(lambda x, y: 300 + 0.2 * y, n=41, res=100.0)
    slope, aspect = terrain.gradient(g)
    mid = (20, 20)
    assert slope[mid] == pytest.approx(math.degrees(math.atan(0.2)), abs=0.5)
    assert aspect[mid] == pytest.approx(180.0, abs=1.0)


def test_sector_picks_single_hill():
    res, n = 100.0, 401
    cx, cy = n * res / 2, n * res / 2
    g = grid_from_height(lambda x, y: 100 + 60 * np.exp(-((x - cx) ** 2 + (y - cy - 3000) ** 2) / (2 * 800 ** 2)),
                         n=n, res=res)
    frame = terrain.polar_frame(g, cx, cy)
    z0 = g.point_value(cx, cy)
    sectors = {s.name: s for s in terrain.mountain_sectors(g, frame, z0)}
    north = sectors["子"].rings[2]
    south = sectors["午"].rings[2]
    assert north["max_rel_elev_m"] > 50
    assert south["max_rel_elev_m"] < 2


def test_four_spirits_follow_axis_not_cardinal():
    res, n = 200.0, 401
    cx, cy = n * res / 2, n * res / 2
    # холм на востоке от точки
    g = grid_from_height(lambda x, y: 100 + 80 * np.exp(-((x - cx - 3000) ** 2 + (y - cy) ** 2) / (2 * 1000 ** 2)),
                         n=n, res=res)
    frame = terrain.polar_frame(g, cx, cy)
    z0 = g.point_value(cx, cy)
    # фасад на север: слева (запад) — дракон, справа (восток) — тигр
    sp = terrain.four_spirits(g, frame, z0, facing_deg=0.0, meso_km=8.0)
    assert sp["tiger"]["rel_elev_m"] > 50      # тигр справа = восток
    assert sp["dragon"]["rel_elev_m"] < 5      # дракон слева = запад, там плоско
    # фасад на юг: слева уже восток → дракон высокий
    sp2 = terrain.four_spirits(g, frame, z0, facing_deg=180.0, meso_km=8.0)
    assert sp2["dragon"]["rel_elev_m"] > 50
    assert sp2["dragon_vs_tiger"] > 40


def test_dragon_vein_climbs_cone_to_peak():
    res, n = 200.0, 401
    cx, cy = n * res / 2, n * res / 2
    peak_x, peak_y = cx + 6000, cy + 6000
    g = grid_from_height(lambda x, y: 100 + 150 * np.exp(-((x - peak_x) ** 2 + (y - peak_y) ** 2) / (2 * 3000 ** 2)),
                         n=n, res=res)
    z0 = g.point_value(cx, cy)
    vein = terrain.dragon_vein(g, cx, cy, z0, max_km=30)
    assert vein.length_km > 3
    assert vein.drop_m > 50
    px, py = vein.path[-1]
    assert math.hypot(px - peak_x, py - peak_y) < 3000     # пришёл в окрестность вершины


def test_dragon_vein_does_not_wander_on_flat():
    g = flat_grid(50.0, n=201, res=100.0)
    vein = terrain.dragon_vein(g, 10_000, 10_000, 50.0, max_km=10)
    assert vein.length_km <= 2.0          # допустимы 1–4 плоских шага, не 10 км
    assert vein.drop_m == 0.0


def test_point_description_shapes():
    g = grid_from_height(lambda x, y: 100 + 0.0005 * x, n=201, res=100.0)
    d = terrain.point_description(g, 10_000, 10_000)
    assert d["slope_deg"] is not None
    assert d["shape"] in {"склон", "гребень", "низина", "плоско / уступ"}


def test_sectors_are_deterministic():
    g = grid_from_height(lambda x, y: 100 + 40 * np.sin(x / 2000) * np.cos(y / 3000), n=401, res=100.0)
    cx = cy = 20_000
    runs = []
    for _ in range(2):
        frame = terrain.polar_frame(g, cx, cy)
        z0 = g.point_value(cx, cy)
        runs.append([(s.name, r["max_rel_elev_m"]) for s in terrain.mountain_sectors(g, frame, z0) for r in s.rings])
    assert runs[0] == runs[1]
