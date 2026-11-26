"""JRC Global Surface Water: имя плитки, доли, сверка с OSM (W2).

Сеть в тестах запрещена: всё, что читает тайлы, проверяется на синтетических массивах,
а `metrics` — с подменённым `_read`.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosite import gsw


def test_tile_name_uses_west_and_north_edges():
    """Плитка 10°×10° называется по западной и северной границам."""
    assert gsw.tile_name(51.5138, -0.0984) == "10W_60N"
    assert gsw.tile_name(-18.1416, 178.4419) == "170E_10S"
    assert gsw.tile_name(0.3, 0.3) == "0E_10N"
    assert gsw.tile_name(84.30, 15.00) == "10E_90N"


def test_tile_name_boundary_belongs_to_upper_tile():
    """Точка ровно на 50° лежит в плитке 60N: 50N покрывает 40…50 и её уже не содержит."""
    assert gsw.tile_name(50.0, -10.0) == "10W_60N"
    assert gsw.tile_name(50.0, 10.0) == "10E_60N"


def test_tile_url_points_at_published_tile():
    url = gsw.tile_url("occurrence", "10W_60N")
    assert url == ("https://storage.googleapis.com/global-surface-water/downloads2021/"
                   "occurrence/occurrence_10W_60Nv1_4_2021.tif")


def test_metrics_counts_only_pixels_inside_circle(monkeypatch):
    """Квадрат и круг дают разные доли: вне круга пиксели не считаются вовсе."""
    # 4×4 пикселя по 0.001° (~111 м), центр — середина сетки; вода только в дальнем углу
    occ = np.zeros((4, 4), dtype="uint8")
    occ[0, 0] = 100
    occ[1, 1] = 100
    transform = (0.001, 0.0, -0.002, 0.0, -0.001, 0.002)

    def fake_read(layer, lat, lon, radius_km, *, allow_network=True, timeout=120):
        if layer == "occurrence":
            return {"array": occ, "transform": transform}
        return {"array": np.full((4, 4), 12, dtype="uint8"), "transform": transform}

    monkeypatch.setattr(gsw, "_read", fake_read)
    got = gsw.metrics(0.0, 0.0, radius_km=0.15)          # ~150 м: углы 0.001° отрезаны кругом
    assert got is not None
    assert got["pixels"] == 4                             # из 16 пикселей внутри круга четыре
    assert got["share_permanent"] == 0.25                 # вода в одном из них


def test_metrics_ignores_nodata(monkeypatch):
    """255 — «вне данных» (океанская часть плитки), а не «сухо»."""
    occ = np.full((4, 4), 255, dtype="uint8")
    occ[1, 1] = 80

    def fake_read(layer, lat, lon, radius_km, *, allow_network=True, timeout=120):
        return {"array": occ, "transform": (0.001, 0.0, -0.002, 0.0, -0.001, 0.002)}

    monkeypatch.setattr(gsw, "_read", fake_read)
    got = gsw.metrics(0.0, 0.0, radius_km=0.15)
    assert got["share_permanent"] == 1.0                  # единственный измеренный пиксель — вода
    assert got["pixels"] == 1


def test_metrics_without_data_returns_none(monkeypatch):
    monkeypatch.setattr(gsw, "_read", lambda *a, **k: None)
    assert gsw.metrics(51.5, -0.1, radius_km=8.0) is None


def test_compare_keeps_osm_as_base():
    """Снимок — второе мнение: расхождение фиксируется, но основа остаётся за OSM."""
    metrics = {"share_permanent": 0.0288}
    same = gsw.compare(0.0300, metrics)
    assert same["status"] == "ok" and abs(same["delta"]) < 0.03
    far = gsw.compare(0.0788, metrics)
    assert far["status"] == "differs"
    assert "за OSM" in far["note"]
    assert gsw.compare(0.05, None)["status"] == "unknown"


def test_offline_mode_does_not_go_to_network(monkeypatch):
    """В офлайне метрики — `None`, а не исключение: пакет собирается без сети."""
    monkeypatch.setenv("GEOSITE_OFFLINE", "1")
    monkeypatch.setattr(gsw, "CACHE_DIR", gsw.CACHE_DIR / "нет-такого-каталога")
    assert gsw.metrics(51.5138, -0.0984, radius_km=8.0) is None
