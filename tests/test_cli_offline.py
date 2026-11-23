"""Офлайн-пересборка воды из сохранённого ответа Overpass (`--osm-file`)."""

from __future__ import annotations

from pathlib import Path

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
    from pyproj import Transformer

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
