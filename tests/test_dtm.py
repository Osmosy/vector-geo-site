"""DTM (D3): реестр, цепочка источников, режимы и паспорт слоя."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geosite import dtm


def test_registry_is_readable_and_has_both_tiers():
    items = dtm.registry()
    ids = {item["id"] for item in items}
    assert {"ahn4", "ea-lidar", "gedtm30"} <= ids
    for item in items:
        # у каждой записи есть то, без чего нельзя подключать: покрытие, лицензия, разрешение
        assert item["bbox"] and item["license"] and item["license_class"] in {"open", "closed"}
        assert item["resolution_m"] > 0 and item["verified"]


def test_no_closed_license_source_is_used():
    """Приёмка раздела 4: источников с запретом коммерческого использования нет в цепочке."""
    assert all(item["license_class"] == "open" for item in dtm.registry())


def test_lidar_lookup_by_bbox():
    assert dtm.lidar_for(52.3731, 4.8922)["id"] == "ahn4"          # Амстердам
    assert dtm.lidar_for(48.8566, 2.3522) is None                  # Париж: покрытия нет


def test_ea_lidar_is_active_after_verified_response():
    """EA LIDAR проверен через VPN (GB) и стоит первым для Англии; Париж по-прежнему вне покрытия."""
    assert dtm.lidar_for(51.5138, -0.0984)["id"] == "ea-lidar"
    assert dtm.lidar_for(48.8566, 2.3522) is None                    # Париж: покрытия нет


def test_wcs2_window_refuses_window_over_limit():
    """Окно 20 км при сетке 2 м — 1.6 ГБ перекачки; источник обязан отказаться, а не тянуть."""
    source = next(s for s in dtm.registry() if s["id"] == "ea-lidar")
    assert source["protocol"] == "wcs-2.0.1"
    with pytest.raises(RuntimeError) as exc:
        dtm._wcs2_window(source, 51.5138, -0.0984, 20.0, 90.0)
    assert "лимита источника" in str(exc.value)


def test_mode_switch(monkeypatch):
    monkeypatch.setenv("GEOSITE_DTM", "none")
    assert dtm.mode() == "none"
    monkeypatch.setenv("GEOSITE_DTM", "чушь")
    assert dtm.mode() == "auto"                    # неизвестное значение — не повод падать


def test_none_mode_gives_no_dtm_and_reason(monkeypatch):
    monkeypatch.setenv("GEOSITE_DTM", "none")
    grid, info = dtm.grid(52.3731, 4.8922, crs=32631, radius_km=8.0)
    assert grid is None and "GEOSITE_DTM=none" in info["reason"]


def test_offline_without_cache_is_none_not_zero(monkeypatch, tmp_path):
    """Нет сети и нет кеша — None с причиной: «не смотрели» не равно «земля на нуле»."""
    monkeypatch.setattr(dtm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("GEOSITE_OFFLINE", "1")
    monkeypatch.setenv("GEOSITE_DTM", "lidar")
    grid, info = dtm.grid(52.3731, 4.8922, crs=32631, radius_km=8.0)
    assert grid is None
    assert info["reason"] and "ahn4" in " ".join(info["tried"])


def test_cache_roundtrip_avoids_network(monkeypatch, tmp_path):
    """Вырезка DTM кешируется: повторный запуск офлайн даёт тот же грид."""
    monkeypatch.setattr(dtm, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir(parents=True)
    size = 8
    array = np.full((size, size), 2.5, dtype="float32")
    transform = (30.0, 0.0, 0.0, 0.0, -30.0, size * 30.0)
    cache = (tmp_path / "cache" / "ahn4_52.3731_4.8922_8_30.npz")
    np.savez_compressed(cache, array=array, transform=np.array(transform), crs="EPSG:4326")

    monkeypatch.setenv("GEOSITE_OFFLINE", "1")
    monkeypatch.setenv("GEOSITE_DTM", "lidar")
    window = dtm._wcs_window(dtm.lidar_for(52.3731, 4.8922), 52.3731, 4.8922, 8.0, 30.0)
    assert window is not None and window["array"].shape == (size, size)


def test_nodata_is_masked():
    """У AHN nodata = 3.4e38: без отсечки среднее по участку становится 2.4e37."""
    array = np.array([[1.0, 3.4e38], [-2.0, np.nan]], dtype="float32")
    masked = dtm._mask_heights(array)
    assert np.isfinite(masked).sum() == 2
    assert float(np.nanmean(masked)) == pytest.approx(-0.5)


def test_file_block_matches_dem_passport():
    block = dtm.file_block({"source": "ahn4", "resolution_m": 30.0, "source_resolution_m": 0.5,
                            "scope": "all", "license": "CC BY 4.0", "vertical_datum": "NAP",
                            "datum_shift_m": None})
    assert block == {"source": "ahn4", "kind": "dtm", "resolution_m": 30.0,
                     "source_resolution_m": 0.5, "scope": "all",
                     "license": "CC BY 4.0", "vertical_datum": "NAP", "datum_shift_m": None}
    # паспорт без области по умолчанию — весь участок, а не «неизвестно»
    assert dtm.file_block({"source": "x"})["scope"] == "all"
    assert dtm.file_block({}) is None


@pytest.mark.skipif(not Path(dtm.REGISTRY).exists(), reason="нет реестра DTM")
def test_registry_json_is_valid():
    data = json.loads(Path(dtm.REGISTRY).read_text(encoding="utf-8"))
    assert data["version"] and data["sources"]
