"""Офлайн-пересборка воды из сохранённого ответа Overpass (`--osm-file`)."""

from __future__ import annotations

from pathlib import Path

from pyproj import Transformer

from geosite import cli

FIXTURE = Path(__file__).parent / "golden" / "london-city" / "fixtures" / "osm-water.json"


class _Resolved:
    lat = 51.5138
    lon = -0.0984
    precision = "district"


def test_payload_file_is_read_and_validated(tmp_path):
    data = cli._load_osm_file(str(FIXTURE))
    assert len(data["elements"]) > 500

    bad = tmp_path / "bad.json"
    bad.write_text('{"nope": 1}', encoding="utf-8")
    try:
        cli._load_osm_file(str(bad))
    except SystemExit as exc:
        assert "elements" in str(exc)
    else:  # pragma: no cover - защита от регрессии
        raise AssertionError("файл без elements должен отвергаться")


def test_water_from_payload_needs_no_network():
    """Сеть не нужна: ответ Overpass пришёл файлом, участок — координатами."""
    payload = cli._load_osm_file(str(FIXTURE))
    x0, y0 = Transformer.from_crs("EPSG:4326", "EPSG:32630",
                                  always_xy=True).transform(_Resolved.lon, _Resolved.lat)
    measured = {"x0": x0, "y0": y0, "grid_macro": None}
    out = cli._measure_water(_Resolved(), measured, 32630, payload=payload)
    assert out.get("water_error") is None
    kinds = {o.kind for o in out["water"]}
    assert "culvert" in kinds
    thames = [o for o in out["water"] if o.name == "River Thames"]
    assert thames and 150 <= thames[0].nearest_m <= 900

def test_measure_water_uses_payload_not_cache(monkeypatch, tmp_path):
    """T1: `--osm-file` доходит до метрик. Если код снова полезет в кеш или сеть — тест упадёт."""
    from geosite import water

    def boom(*args, **kwargs):
        raise AssertionError("payload передан — обращений к кешу и сети быть не должно")

    monkeypatch.setattr(water, "fetch_raw", boom)
    monkeypatch.setenv("GEOSITE_CACHE", str(tmp_path))

    payload = cli._load_osm_file(str(FIXTURE))
    x0, y0 = Transformer.from_crs("EPSG:4326", "EPSG:32630",
                                  always_xy=True).transform(_Resolved.lon, _Resolved.lat)
    measured = {"x0": x0, "y0": y0, "grid_macro": None}
    out = cli._measure_water(_Resolved(), measured, 32630, payload=payload)
    assert out.get("water_error") is None
    assert len(out["water"]) > 30
    thames = [o for o in out["water"] if o.name == "River Thames"]
    assert thames and 300 <= thames[0].nearest_m <= 600


def test_water_command_offline_from_file(monkeypatch, tmp_path, capsys):
    """T1: `geosite water --osm-file` считает воду из файла, не обращаясь к сети.

    DEM здесь не нужен: подменяем только его, чтобы проверка не зависела от кеша тайлов.
    """
    from geosite import water

    def boom(*a, **k):
        raise AssertionError("сеть: water.fetch_raw не должен вызываться при --osm-file")

    monkeypatch.setattr(water, "fetch_raw", boom)
    monkeypatch.setenv("GEOSITE_CACHE", str(tmp_path))
    x0, y0 = Transformer.from_crs("EPSG:4326", "EPSG:32630",
                                  always_xy=True).transform(_Resolved.lon, _Resolved.lat)
    monkeypatch.setattr(cli.terrain, "measure_site",
                        lambda *a, **k: {"x0": x0, "y0": y0, "grid_macro": None})
    rc = cli.main(["water", "51.5138, -0.0984", "--osm-file", str(FIXTURE), "--offline"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Thames" in out and '"error": null' in out
