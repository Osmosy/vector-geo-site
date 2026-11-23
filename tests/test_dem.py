"""Рельеф: тайлы, мозаика, запасной источник."""

from __future__ import annotations

import pytest


def test_read_hgt_decodes_and_marks_nodata(tmp_path):
    """HGT — int16 big-endian; -32768 означает «нет данных», а не глубину."""
    import gzip

    import numpy as np

    from geosite import dem

    side = 4
    data = np.array([[10, 20, 30, 40],
                     [-32768, 25, 35, 45],
                     [11, 21, 31, 41],
                     [12, 22, 32, 42]], dtype=">i2")
    path = tmp_path / "N51E000.hgt.gz"
    with gzip.open(path, "wb") as fh:
        fh.write(data.tobytes())
    grid, transform = dem.read_hgt(path)
    assert grid.shape == (side, side)
    assert grid[0, 0] == 10 and grid[1, 1] == 25
    assert np.isnan(grid[1, 0])
    assert abs(transform.c - 0.0) < 1e-12 and abs(transform.f - 52.0) < 1e-12


def _stub_grid(source: str, label: str, epsg: int):
    import numpy as np

    from affine import Affine

    from geosite import dem

    grid = dem.DemGrid(np.zeros((3, 3), dtype="float32"), Affine(30.0, 0.0, 0.0, 0.0, -30.0, 0.0),
                       f"EPSG:{epsg}", 30.0)
    grid.source, grid.label = source, label
    return grid


def test_load_dem_falls_back_to_srtm(monkeypatch):
    """Основной источник упал — рельеф берётся из запасного, и это видно по source.

    Подменяем загрузчики: порядок перебора проверяется без сети (реальная загрузка — в
    `test_load_dem_srtm_real_tile` под маркером network).
    """
    from geosite import dem

    def broken(*a, **k):
        raise RuntimeError("Copernicus: 503")

    monkeypatch.setattr(dem, "_load_copernicus", broken)
    monkeypatch.setattr(dem, "_load_srtm",
                        lambda *a, **k: _stub_grid("srtm", dem.SRTM_LABEL, k.get("epsg", 32630)))
    monkeypatch.delenv("GEOSITE_DEM", raising=False)
    grid = dem.load_dem(51.5138, -0.0984, 2.0, epsg=32630, resolution_m=90.0)
    assert grid.source == "srtm"
    assert "SRTM" in grid.label


@pytest.mark.network
def test_load_dem_srtm_real_tile():
    """Настоящий тайл SRTM: скачивание и чтение HGT (около 5 МБ, нужен интернет)."""
    from geosite import dem

    grid = dem.load_dem(51.5138, -0.0984, 5.0, epsg=32630, resolution_m=30.0, source="srtm")
    assert grid.source == "srtm"
    assert grid.data.shape[0] > 100


def test_load_dem_reports_both_sources_when_none_work(monkeypatch):
    from geosite import dem

    monkeypatch.setattr(dem, "_load_copernicus", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("нет сети")))
    monkeypatch.setattr(dem, "_load_srtm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("нет кеша")))
    monkeypatch.delenv("GEOSITE_DEM", raising=False)
    try:
        dem.load_dem(51.5, -0.1, 1.0, epsg=32630, resolution_m=90.0)
    except RuntimeError as exc:
        assert "copernicus" in str(exc) and "srtm" in str(exc)
    else:
        raise AssertionError("ожидали RuntimeError с обоими источниками")
