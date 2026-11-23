"""Рельеф: тайлы, мозаика, запасной источник."""

from __future__ import annotations


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


def test_load_dem_falls_back_to_srtm(monkeypatch):
    """Основной источник упал — рельеф берётся из запасного и это видно по source."""
    from geosite import dem

    def broken(*a, **k):
        raise RuntimeError("Copernicus: 503")

    monkeypatch.setattr(dem, "_load_copernicus", broken)
    monkeypatch.delenv("GEOSITE_DEM", raising=False)
    grid = dem.load_dem(51.5138, -0.0984, 2.0, epsg=32630, resolution_m=90.0)
    assert grid.source == "srtm"
    assert "SRTM" in grid.label


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
