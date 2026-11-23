"""Порядок реки и расход — HydroRIVERS v1.0 (WWF/HydroSHEDS, CC-BY 4.0).

Зачем отдельный источник: OSM знает имя и форму русла, но не знает, какая это река по
величине. Без порядка реки «ручей в 3 м» и «Волга» в пакете выглядят одинаково, и
толкователь не может отличить 來水-ручей от большой воды.

Как устроено: 938 544 участка на всю Европу в одном shapefile (150 МБ). Читать его на каждый
участок нельзя, поэтому один раз строится индекс по bbox (npy, 19 МБ) в кеше, а на участок
из него берутся кандидаты — десятки линий, среди них ищется ближайшая по расстоянию до
отрезков в UTM. Кеш per-site не нужен: индекс общий, выборка мгновенная.

Атрибуция HydroRIVERS: Lehner, B., Grill G. (2013). Global river hydrography and network
routing: baseline and new approaches to quantify river discharge. Hydrological Processes.
Данные: HydroSHEDS v1.0, CC-BY 4.0.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ALGO_VERSION = "1.0.0"

SOURCE = "HydroRIVERS v1.0 (WWF/HydroSHEDS, CC-BY 4.0)"
EU_SHAPE = "HydroRIVERS_v10_eu_shp/HydroRIVERS_v10_eu.shp"
ARCHIVE_URL = "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_eu_shp.zip"

# поля, которые попадают в пакет; остальные не нужны и только раздувают выборку
META_FIELDS = ("HYRIV_ID", "MAIN_RIV", "ORD_STRA", "ORD_CLAS", "DIS_AV_CMS",
               "LENGTH_KM", "DIST_DN_KM", "UPLAND_SKM")


def cache_dir() -> Path:
    path = Path.home() / ".cache" / "vector-geo-site" / "hydrorivers"
    path.mkdir(parents=True, exist_ok=True)
    return path


def shape_path() -> Path:
    return cache_dir() / EU_SHAPE


def index_path() -> Path:
    return cache_dir() / "index-eu.npy"


def _reader():
    """pyshp — необязательная зависимость: без неё участок просто остаётся без порядка."""
    try:
        import shapefile  # noqa: F401
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise RuntimeError("нет pyshp: uv pip install pyshp") from exc
    import shapefile
    path = shape_path()
    if not path.exists():
        raise FileNotFoundError(f"нет HydroRIVERS: {path} (скачать {ARCHIVE_URL})")
    return shapefile.Reader(str(path))


def build_index(rebuild: bool = False) -> np.ndarray:
    """Индекс bbox всех участков: (id, xmin, ymin, xmax, ymax). Строится один раз, ~40 с."""
    path = index_path()
    if path.exists() and not rebuild:
        return np.load(path)
    reader = _reader()
    rows = np.empty((len(reader), 5), dtype=np.float32)
    for i, shape in enumerate(reader.iterShapes()):
        xmin, ymin, xmax, ymax = shape.bbox
        rows[i] = (i, xmin, ymin, xmax, ymax)
    np.save(path, rows)
    return rows


def _distance_m(lat0: float, lon0: float, points: list[tuple[float, float]]) -> float:
    """Расстояние от точки до ломаной: в метрах, по равнопромежуточной проекции.

    На масштабе 10 км ошибка проекции меньше ширины русла, поэтому UTM не нужен и модуль
    остаётся без pyproj.
    """
    kx = 111_320.0 * math.cos(math.radians(lat0))
    ky = 110_540.0
    best = float("inf")
    px, py = 0.0, 0.0
    prev: tuple[float, float] | None = None
    for lon, lat in points:
        x = (lon - lon0) * kx
        y = (lat - lat0) * ky
        if prev is None:
            best = min(best, math.hypot(x, y))
        else:
            ax, ay = prev
            bx, by = x, y
            dx, dy = bx - ax, by - ay
            seg = dx * dx + dy * dy
            t = 0.0 if seg <= 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / seg))
            best = min(best, math.hypot(ax + t * dx, ay + t * dy))
        prev = (x, y)
        px, py = x, y
    return best


def nearest(lat: float, lon: float, *, max_km: float = 10.0, reader=None,
            index: np.ndarray | None = None) -> dict | None:
    """Ближайший участок HydroRIVERS к точке: bbox-кандидаты, затем точное расстояние."""
    reader = reader or _reader()
    index = build_index() if index is None else index
    dlat = max_km / 110.54
    dlon = max_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    box = index[(index[:, 1] <= lon + dlon) & (index[:, 3] >= lon - dlon) &
                (index[:, 2] <= lat + dlat) & (index[:, 4] >= lat - dlat)]
    if not len(box):
        return None
    best: dict | None = None
    for row in box:
        idx = int(row[0])
        record = reader.record(idx).as_dict()
        shape = reader.shape(idx)
        points = [(x, y) for x, y in shape.points]
        dist = _distance_m(lat, lon, points)
        if best is None or dist < best["distance_m"]:
            best = {"distance_m": round(dist, 1), **{k: record.get(k) for k in META_FIELDS}}
    if best is None or best["distance_m"] > max_km * 1000.0:
        return None
    best["source"] = SOURCE
    return best


def reaches_within(lat: float, lon: float, *, max_km: float = 2.0, reader=None,
                   index: np.ndarray | None = None,
                   with_geometry: bool = False) -> list[dict]:
    """Все участки HydroRIVERS в радиусе: расстояние от точки до ломаной."""
    reader = reader or _reader()
    index = build_index() if index is None else index
    dlat = max_km / 110.54
    dlon = max_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    box = index[(index[:, 1] <= lon + dlon) & (index[:, 3] >= lon - dlon) &
                (index[:, 2] <= lat + dlat) & (index[:, 4] >= lat - dlat)]
    out: list[dict] = []
    for row in box:
        idx = int(row[0])
        points = [(x, y) for x, y in reader.shape(idx).points]
        dist = _distance_m(lat, lon, points)
        if dist > max_km * 1000.0:
            continue
        record = reader.record(idx).as_dict()
        item = {"distance_m": round(dist, 1), **{k: record.get(k) for k in META_FIELDS}}
        if with_geometry:
            item["_points"] = points
        out.append(item)
    return out


def site_reach(lat: float, lon: float, *, max_km: float = 2.0, reader=None,
               index: np.ndarray | None = None, reaches: list[dict] | None = None) -> dict | None:
    """Главный участок у точки: наибольший порядок, затем наибольший расход.

    «Ближайший» не годится: у HydroRIVERS участки короткие (1–7 км) и в плоском эстуарии
    смещены от русла OSM, поэтому ближайшим к точке в Лондоне оказывается ручей с расходом
    0.18 м³/с. Главный участок в радиусе — это то, что человек называет «река здесь».
    """
    items = reaches if reaches is not None else reaches_within(
        lat, lon, max_km=max_km, reader=reader, index=index, with_geometry=True)
    if not items:
        return None
    best = max(items, key=lambda r: (r.get("ORD_STRA") or 0, r.get("DIS_AV_CMS") or 0.0))
    return {
        "_points": best.get("_points", []),
        "order": best.get("ORD_STRA"),
        "order_class": best.get("ORD_CLAS"),
        "discharge_m3s": best.get("DIS_AV_CMS"),
        "distance_m": best.get("distance_m"),
        "channels_within_km": len(items),
        "hyriv_id": best.get("HYRIV_ID"),
        "source": SOURCE,
    }


def assign(metrics, reach: dict | None, *, max_site_m: float = 1000.0,
           agree_m: float = 400.0, min_share: float = 0.6) -> list[str]:
    """Порядок и расход — только реке, с чьим руслом совпал участок HydroRIVERS.

    Совпадение считается от участка к реке, а не наоборот: участки HydroRIVERS короткие
    (1–7 км, 3–10 точек), и доля точек 12-километрового русла OSM рядом с таким участком
    всегда мала — даже когда это одна и та же река.

    Если совпадения нет, порядок остаётся `None`: подставить его «ближайшей реке» значило бы
    выдать Москве-реке порядок канала, а засыпанному Флиту — расход Темзы. Крупная вода у
    участка всё равно видна потребителю в `site.hydro` с расстоянием.
    """
    warnings: list[str] = []
    if reach is None:
        warnings.append("hydro: в радиусе 2 км нет участка HydroRIVERS — порядок и расход не определены")
        return warnings
    points = reach.get("_points") or []
    rivers = [m for m in metrics
              if m.kind in ("river", "canal") and m.nearest_m is not None
              and m.nearest_m <= max_site_m]
    if not points or not rivers:
        warnings.append("hydro: участок есть, но совпадения с руслом проверить нечем — порядок не проставлен")
        return warnings
    best: tuple[float, object] | None = None
    for metric in rivers:
        coords = getattr(metric, "coords", None) or []
        if len(coords) < 2:
            continue
        share = sum(1 for pt in points if _distance_m(pt[1], pt[0], coords) <= agree_m) / len(points)
        if share >= min_share and (best is None or share > best[0]):
            best = (share, metric)
    if best is None:
        warnings.append(
            f"hydro: порядок {reach.get('order')} и расход {reach.get('discharge_m3s')} м3/с "
            f"остались только в site.hydro — русло у участка с участком HydroRIVERS не совпало")
        return warnings
    share, metric = best
    metric.order = reach.get("order")
    metric.discharge_m3s = reach.get("discharge_m3s")
    metric.hydro_distance_m = reach.get("distance_m")
    metric.hydro_agree = round(share, 2)
    return warnings
