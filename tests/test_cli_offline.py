"""Офлайн-пересборка воды из сохранённого ответа Overpass (`--osm-file`)."""

from __future__ import annotations

from pathlib import Path

from pyproj import Transformer

from geosite import cli

FIXTURE = Path(__file__).parent / "golden" / "london-city" / "fixtures" / "osm-water.json.gz"


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


def test_load_osm_file_accepts_gzip_fixture():
    """Широкая фикстура в репозитории лежит сжатой: формат виден по магии, а не по расширению."""
    gz = (Path(__file__).parent / "golden" / "london-city" / "fixtures" / "osm-water-wide.json.gz")
    data = cli._load_osm_file(str(gz))
    assert len(data["elements"]) > 100


def _osm_way(osm_id: int, name: str, points: list[tuple[float, float]], *, tidal: bool = False):
    tags = {"name": name, "waterway": "river"}
    if tidal:
        tags["tidal"] = "yes"
    return {"type": "way", "id": osm_id, "tags": tags,
            "geometry": [{"lat": la, "lon": lo} for la, lo in points]}


def _measure(args, payload):
    from types import SimpleNamespace

    from geosite import cli as cli_mod

    measured = {"x0": 0.0, "y0": 0.0, "grid_macro": None}
    measured.update(_site_xy())
    return cli_mod._measure_water(_Resolved(), measured, 32630, payload=payload,
                                  args=SimpleNamespace(**args))


def _site_xy():
    x0, y0 = Transformer.from_crs("EPSG:4326", "EPSG:32630",
                                  always_xy=True).transform(_Resolved.lon, _Resolved.lat)
    return {"x0": x0, "y0": y0}


def test_tidal_resolved_from_wide_payload(tmp_path, monkeypatch):
    """Река из таблицы приливов без тега `tidal`: решает широкая выгрузка, а не догадка.

    Лондонская фикстура такого случая не содержит (у Темзы есть тег), поэтому он собран
    синтетически: русло нарисовано от границы прилива (Теддингтон) к участку, то есть по
    течению. Кеш OSM отводится в пустой каталог — иначе широкая выгрузка нашлась бы в кеше
    и тест проверял бы не то.
    """
    import gzip
    import json

    from geosite import water as water_mod

    # CACHE_DIR вычисляется при импорте, поэтому подменяем сам атрибут, а не переменную среды
    monkeypatch.setattr(water_mod, "CACHE_DIR", tmp_path / "cache")
    limit = (51.431, -0.308)                       # Teddington Lock из tidal_limits.json
    site = (51.5138, -0.0984)
    between = [(limit[0] + (site[0] - limit[0]) * k / 6, limit[1] + (site[1] - limit[1]) * k / 6)
               for k in range(1, 6)]
    narrow = {"elements": [_osm_way(1, "River Thames", between[3:] + [site])]}
    wide = {"elements": [_osm_way(1, "River Thames", [limit] + between + [site])]}
    wide_path = tmp_path / "water-wide.json.gz"
    wide_path.write_bytes(gzip.compress(json.dumps(wide).encode("utf-8")))

    without = _measure({"osm_file_wide": None, "offline": True}, narrow)["water"]
    thames = [o for o in without if o.name == "River Thames"][0]
    assert thames.tidal is None and thames.tidal_reason == "no_wide_payload"

    with_wide = _measure({"osm_file_wide": str(wide_path), "offline": True}, narrow)["water"]
    thames = [o for o in with_wide if o.name == "River Thames"][0]
    assert thames.tidal is True, "участок ниже границы прилива — вода приливная"
    assert thames.tidal_reason is None
    assert thames.tidal_source == "known_limit"      # решено по таблице, а не тегом
