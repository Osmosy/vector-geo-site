"""Покров ESA WorldCover (W2): тайлы, доли групп, сверка воды с OSM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from geosite import geo, landcover

FIXTURE = Path(__file__).parent / "golden" / "london-city" / "fixtures" / "worldcover.tif"


def _tif(tmp_path: Path, lat: float, lon: float, *, half_km: float = 2.0,
         classes: dict[int, float] | None = None, default: int = 80) -> Path:
    """Синтетическая вырезка: классы по квадратам (доля площади, класс)."""
    import rasterio

    res = 10.0
    size = int(half_km * 1000 / res) * 2
    array = np.full((size, size), default, dtype="uint8")
    if classes:
        # {класс: доля} — раскладываем по пикселям с фиксированным зерном: доля не зависит
        # от формы кольца, а именно её и проверяем
        rng = np.random.default_rng(7)
        codes = np.array(list(classes), dtype="uint8")
        weights = np.array(list(classes.values()), dtype=float)
        flat = rng.choice(codes, size=size * size, p=weights / weights.sum())
        array = flat.reshape(size, size).astype("uint8")
    # пиксель в градусах: тайлы WorldCover лежат в EPSG:4326, а не в метрах.
    # По долготе шаг крупнее — иначе квадрат в градусах не квадрат в метрах.
    step_y = res / landcover.M_PER_DEG_LAT
    step_x = res / (landcover.M_PER_DEG_LAT * max(0.05, np.cos(np.radians(lat))))
    transform = Affine(step_x, 0, lon - size / 2.0 * step_x, 0, -step_y, lat + size / 2.0 * step_y)
    path = tmp_path / "wc.tif"
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="uint8", crs="EPSG:4326", transform=transform) as dst:
        dst.write(array, 1)
    return path


def _site(lat: float, lon: float):
    epsg = geo.utm_epsg(lat, lon)
    site = np.array([0.0, 0.0])
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    site = np.array(tr.transform(lon, lat))
    return epsg, site


def test_tile_names_follow_three_degree_grid():
    """Имя тайла — юго-западный угол 3°: широта 2 знака, долгота 3."""
    assert landcover.tile_name(51.5138, -0.0984) == "N51W003"
    assert landcover.tile_name(-33.8568, 151.2153) == "S36E150"
    assert landcover.tile_name(-18.1416, 178.4419) == "S21E177"
    assert landcover.tile_name(64.1466, -21.9426) == "N63W024"
    assert landcover.tile_name(0.5, 0.5) == "N00E000"


def test_all_water_gives_share_one(tmp_path):
    path = _tif(tmp_path, 51.5138, -0.0984, default=80)
    epsg, site = _site(51.5138, -0.0984)
    data = landcover.measure(51.5138, -0.0984, epsg=epsg, site_xy=site, path=path,
                             radius_m=2000.0, rings=((0.0, 2000.0),))
    assert data is not None
    assert data["rings"]["0-2"]["share"] == pytest.approx(1.0, abs=0.01)
    assert data["groups"]["water"] == pytest.approx(1.0, abs=0.02)


def test_wetland_is_not_open_water(tmp_path):
    """Класс 90 (болото) и 95 (мангры) — не открытая вода: иначе расхождение с OSM
    объяснялось бы своей же ошибкой."""
    path = _tif(tmp_path, 51.5138, -0.0984, default=90)
    epsg, site = _site(51.5138, -0.0984)
    data = landcover.measure(51.5138, -0.0984, epsg=epsg, site_xy=site, path=path,
                             radius_m=2000.0, rings=((0.0, 2000.0),))
    assert data["groups"]["water"] == 0.0
    assert data["rings"]["0-2"]["share"] == 0.0


def test_groups_split_by_class(tmp_path):
    """Застройка, дерево и открытое считаются по своим классам, а не «всё не вода»."""
    path = _tif(tmp_path, 51.5138, -0.0984, classes={50: 0.5, 10: 0.3, 30: 0.2})
    epsg, site = _site(51.5138, -0.0984)
    data = landcover.measure(51.5138, -0.0984, epsg=epsg, site_xy=site, path=path,
                             radius_m=2000.0, rings=((0.0, 2000.0),))
    assert data["groups"]["built"] == pytest.approx(0.5, abs=0.03)
    assert data["groups"]["tree"] == pytest.approx(0.3, abs=0.03)
    assert data["groups"]["open"] == pytest.approx(0.2, abs=0.03)
    assert data["groups"]["water"] == 0.0
    assert sum(data["groups"].values()) == pytest.approx(1.0, abs=0.03)


def test_offline_without_cache_returns_none(tmp_path, monkeypatch):
    """Нет кеша и нет сети — None, а не нули: «не смотрели» не равно «воды нет»."""
    monkeypatch.setattr(landcover, "CACHE_DIR", tmp_path / "cache")
    epsg, site = _site(51.5138, -0.0984)
    assert landcover.window(51.5138, -0.0984, 8000.0, offline=True) is None
    assert landcover.measure(51.5138, -0.0984, epsg=epsg, site_xy=site,
                             offline=True) is None


@pytest.mark.skipif(not FIXTURE.exists(), reason="нет вырезки WorldCover для Лондона")
def test_london_fixture_matches_satellite():
    """Лондон по снимкам: застройка доминирует, воды в кольце 0.5–2 км около 7 %."""
    epsg, site = _site(51.5138, -0.0984)
    data = landcover.measure(51.5138, -0.0984, epsg=epsg, site_xy=site, path=FIXTURE)
    assert data["source"] == landcover.SOURCE
    assert data["groups"]["built"] > 0.55
    assert 0.02 < data["groups"]["water"] < 0.12, data["groups"]
    assert data["groups"]["tree"] > 0.02
    # вода по снимкам и по OSM расходятся меньше чем на 2 п.п. в кольце целиком
    assert abs(data["rings"]["0.5-2"]["share"] - 0.0788) < 0.02


@pytest.mark.skipif(not FIXTURE.exists(), reason="нет вырезки WorldCover для Лондона")
def test_river_sector_seen_by_both_sources():
    """Южные сектора у Темзы видны обоими источниками: и OSM, и снимки дают десятки процентов."""
    epsg, site = _site(51.5138, -0.0984)
    data = landcover.measure(51.5138, -0.0984, epsg=epsg, site_xy=site, path=FIXTURE)
    south = data["rings"]["0.5-2"]["by_mountain"]
    assert south["申"] > 0.3 and south["辰"] > 0.2, south
