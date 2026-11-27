"""GEDTM30 через OpenTopography: ключ, предел радиуса, офлайн, разбор ответа.

Сеть в тестах запрещена: ответ сервиса подменяется готовым GeoTIFF в памяти,
а `urlopen` — заглушкой. Так проверяется разбор, а не чужой аптайм.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosite import dtm


def _geotiff(values: np.ndarray, *, nodata: float = 3.4028234663852886e+38) -> bytes:
    """Маленький GeoTIFF 4326 в памяти — то, что вернул бы OpenTopography."""
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=values.shape[0], width=values.shape[1], count=1,
                      dtype="float32", crs="EPSG:4326", nodata=nodata,
                      transform=from_origin(-0.2, 51.6, 1 / 3600, 1 / 3600)) as ds:
            ds.write(values.astype("float32"), 1)
        return mem.read()


SOURCE = {"id": "gedtm30", "name": "GEDTM30", "kind": "opentopography",
          "url": "https://portal.opentopography.org/API/globaldem", "demtype": "GEDTM30",
          "key_env": "OPENTOPOGRAPHY_API_KEY", "resolution_m": 30.0, "license": "CC BY 4.0"}


@pytest.fixture(autouse=True)
def _no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(dtm, "CACHE_DIR", tmp_path / "dtm")
    monkeypatch.setattr(dtm, "KEY_FILE", tmp_path / "нет-ключа.key")   # ключ из ~/.config не подтекает
    monkeypatch.delenv("GEOSITE_OFFLINE", raising=False)
    monkeypatch.delenv("GEOSITE_GEDTM30_URL", raising=False)
    monkeypatch.setenv("OPENTOPOGRAPHY_API_KEY", "test-key")


def test_window_reads_geotiff_and_masks_nodata(monkeypatch):
    """Nodata сервиса (3.4e38) уходит в NaN: пустота не должна стать высотой."""
    values = np.full((36, 36), 42.0)
    values[0, 0] = 3.4028234663852886e+38
    seen = {}

    def fake_urlopen(url, timeout=None):
        seen["url"] = url

        class R:
            def read(self):
                return _geotiff(values)
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        return R()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    got = dtm._gedtm30_window(SOURCE, 51.5138, -0.0984, 8.0, 30.0)
    assert got is not None
    assert got["crs"] == "EPSG:4326"
    assert np.isnan(got["array"][0, 0])
    assert np.isfinite(got["array"][1, 1])
    assert "demtype=GEDTM30" in seen["url"] and "API_Key=test-key" in seen["url"]


def test_window_asks_bbox_by_diagonal(monkeypatch):
    """Окно шире радиуса: сетка пакета квадратная и в UTM, углам нужен запас."""
    seen = {}

    def fake_urlopen(url, timeout=None):
        seen["url"] = url

        class R:
            def read(self):
                return _geotiff(np.full((36, 36), 10.0))
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        return R()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    dtm._gedtm30_window(SOURCE, 51.5138, -0.0984, 8.0, 30.0)
    span_lat = 51.5138 - (51.5138 - 8.0 * 1.45 / 111.32)     # половина окна в градусах
    assert span_lat > 8.0 / 111.32                            # запас по диагонали есть


def test_window_without_key_explains_what_is_missing(monkeypatch):
    monkeypatch.delenv("OPENTOPOGRAPHY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ключа OpenTopography"):
        dtm._gedtm30_window(SOURCE, 51.5138, -0.0984, 8.0, 30.0)


def test_window_refuses_regional_radius():
    """32 км — предел: это рельеф ближнего кольца, а не региональная мозаика."""
    with pytest.raises(RuntimeError, match="предела 32 км"):
        dtm._gedtm30_window(SOURCE, 51.5138, -0.0984, 40.0, 30.0)


def test_window_offline_does_not_touch_network(monkeypatch):
    monkeypatch.setenv("GEOSITE_OFFLINE", "1")

    def boom(*a, **k):
        raise AssertionError("сеть в офлайне трогать нельзя")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(RuntimeError, match="offline"):
        dtm._gedtm30_window(SOURCE, 51.5138, -0.0984, 8.0, 30.0)


def test_window_survives_error_page(monkeypatch):
    """Сервис вместо растра вернул XML — это ошибка с внятным текстом, а не «нет данных»."""

    class R:
        def read(self):
            return b'<?xml version="1.0"?><error>bad key</error>'
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=None: R())
    with pytest.raises(RuntimeError, match="вместо GeoTIFF"):
        dtm._gedtm30_window(SOURCE, 51.5138, -0.0984, 8.0, 30.0)


def test_registry_gedtm30_is_active_and_asks_for_key():
    source = next(s for s in dtm.registry() if s["id"] == "gedtm30")
    assert source["active"] is True
    assert source["kind"] == "opentopography"
    assert source["key_env"] == "OPENTOPOGRAPHY_API_KEY"
    assert source["license_class"] == "open"
    assert source["resolution_m"] == 30.0
