"""Превращение ввода в точку: координаты, адрес, населённый пункт.

Nominatim: ≤ 1 запрос/с, свой User-Agent, кеш ответов в ~/.cache/vector-geo-site.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import geo

ALGO_VERSION = "1.0.0"

USER_AGENT = "vector-geo-site/0.1 (fengshui landform measurement; https://github.com/Osmosy/vector-geo-site)"
NOMINATIM = "https://nominatim.openstreetmap.org"
CACHE_DIR = Path(os.environ.get("GEOSITE_CACHE", Path.home() / ".cache" / "vector-geo-site"))
MIN_INTERVAL_S = 1.1

_last_call = 0.0

_DECIMAL = re.compile(r"^\s*(-?\d{1,3}(?:[.,]\d+)?)\s*[,;\s]\s*(-?\d{1,3}(?:[.,]\d+)?)\s*$")
_DMS = re.compile(r"(\d{1,3})[°d]\s*(\d{1,2})?['′m]?\s*(\d{1,2}(?:\.\d+)?)?[\"″s]?\s*([NnSs])", re.I)
_DMS_LON = re.compile(r"(\d{1,3})[°d]\s*(\d{1,2})?['′m]?\s*(\d{1,2}(?:\.\d+)?)?[\"″s]?\s*([EeWw])", re.I)


@dataclass
class Resolved:
    input: str
    lat: float
    lon: float
    kind: str
    precision: str
    display_name: str = ""
    resolved_by: str = ""
    boundary_geojson: dict | None = None
    alternatives: list[dict] = field(default_factory=list)
    bbox: list[float] | None = None


class Ambiguous(Exception):
    def __init__(self, options: list[dict]):
        super().__init__("неоднозначный ввод")
        self.options = options


def parse_coordinates(text: str) -> tuple[float, float] | None:
    """Десятичные координаты или градусы-минуты-секунды."""
    m = _DECIMAL.match(text)
    if m:
        lat, lon = (float(x.replace(",", ".")) for x in m.groups())
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    lat_m, lon_m = _DMS.search(text), _DMS_LON.search(text)
    if lat_m and lon_m:
        def conv(mo: re.Match) -> float:
            d, mi, se, hemi = mo.groups()
            val = float(d) + float(mi or 0) / 60 + float(se or 0) / 3600
            return -val if hemi.upper() in "SW" else val
        return conv(lat_m), conv(lon_m)
    return None


def _get(url: str, params: dict) -> dict:
    global _last_call
    if os.environ.get("GEOSITE_OFFLINE") == "1":
        raise ConnectionError("режим --offline: сетевой запрос запрещён, а кеша нет")
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{q}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=40) as resp:
        _last_call = time.monotonic()
        return json.loads(resp.read().decode())


def _cached(url: str, params: dict, ttl_days: int = 30) -> dict | list:
    key = hashlib.sha256((url + urllib.parse.urlencode(sorted(params.items()))).encode()).hexdigest()[:24]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"nominatim-{key}.json"
    if path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days < ttl_days:
            return json.loads(path.read_text())
    data = _get(url, params)
    path.write_text(json.dumps(data, ensure_ascii=False))
    return data


def resolve(text: str, *, ask_threshold: float = 0.55) -> Resolved:
    """Ввод → точка. При неоднозначности поднимает Ambiguous с вариантами."""
    coords = parse_coordinates(text)
    if coords:
        lat, lon = coords
        return Resolved(text, lat, lon, "point", "point", f"{lat:.5f}, {lon:.5f}", "input", None, [], None)

    results = _cached(f"{NOMINATIM}/search",
                      {"q": text, "format": "jsonv2", "limit": 5, "addressdetails": 1, "extratags": 1})
    if not results:
        raise LookupError(f"ничего не найдено: {text!r}")

    def rank(r: dict) -> float:
        imp = float(r.get("importance", 0) or 0)
        cls = r.get("class", "")
        boost = 0.15 if cls in {"place", "boundary"} else 0.05 if cls == "building" else 0.0
        return imp + boost

    results = sorted(results, key=rank, reverse=True)
    best = results[0]
    top_two_gap = rank(best) - (rank(results[1]) if len(results) > 1 else 0.0)
    options = [{"name": r.get("display_name", ""), "lat": float(r["lat"]), "lon": float(r["lon"]),
                "type": r.get("type", ""), "class": r.get("class", ""),
                "importance": r.get("importance", 0)} for r in results[:3]]
    if top_two_gap < ask_threshold and len(results) > 1:
        raise Ambiguous(options)

    cls, typ = best.get("class", ""), best.get("type", "")
    lat, lon = float(best["lat"]), float(best["lon"])
    if typ == "house" or cls == "building":
        kind, precision = "address", "building"
    elif cls in {"boundary", "place"} and typ in {"city", "town", "village", "suburb", "administrative", "municipality", "district"}:
        kind, precision = "settlement", "district"
    else:
        kind, precision = "point", "point"

    boundary = None
    if best.get("osm_type") and best.get("osm_id"):
        boundary = _fetch_boundary(best["osm_type"], int(best["osm_id"]))

    return Resolved(text, lat, lon, kind, precision, best.get("display_name", ""),
                    "nominatim", boundary, options, [float(x) for x in best.get("boundingbox", [])] if best.get("boundingbox") else None)


def _fetch_boundary(osm_type: str, osm_id: int) -> dict | None:
    prefix = {"node": "N", "way": "W", "relation": "R"}.get(osm_type)
    if not prefix:
        return None
    try:
        data = _cached(f"{NOMINATIM}/lookup",
                       {"osm_ids": f"{prefix}{osm_id}", "format": "jsonv2", "polygon_geojson": 1}, ttl_days=90)
    except Exception:
        return None
    if isinstance(data, list) and data and data[0].get("geojson"):
        return data[0]["geojson"]
    return None
