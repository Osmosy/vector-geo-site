"""Рельеф участка: 24 горы, четыре духа, драконья жила, описание точки.

Модуль только измеряет: высоты, углы, расстояния, доли. Оценок «хорошо / плохо» здесь нет.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import geo
from .dem import DemGrid, gradient, smooth

ALGO_VERSION = "2.0.0"

EYE_M = 2.0          # высота глаз для угла горизонта
RINGS_KM = [(0.0, 0.5), (0.5, 2.0), (2.0, 8.0), (8.0, 50.0)]
HILL_THRESHOLD_M = 5.0   # превышение над точкой, с которого считаем «возвышенность»


@dataclass
class PolarFrame:
    """Полярные координаты пикселей относительно участка в UTM."""
    dist: np.ndarray
    bearing: np.ndarray
    valid: np.ndarray


def polar_frame(grid: DemGrid, x0: float, y0: float) -> PolarFrame:
    h, w = grid.data.shape
    rows, cols = np.mgrid[0:h, 0:w]
    xs = grid.transform.c + (cols + 0.5) * grid.transform.a
    ys = grid.transform.f + (rows + 0.5) * grid.transform.e
    dx, dy = xs - x0, ys - y0
    dist = np.hypot(dx, dy)
    bearing = (np.degrees(np.arctan2(dx, dy))) % 360.0
    valid = ~np.isnan(grid.data)
    return PolarFrame(dist, bearing, valid)


def sector_slice(bearing: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Маска сектора с учётом перехода через 360°."""
    lo, hi = geo.normalize_deg(lo), geo.normalize_deg(hi)
    if lo <= hi:
        return (bearing >= lo) & (bearing < hi)
    return (bearing >= lo) | (bearing < hi)


@dataclass
class MountainMetrics:
    name: str
    rings: list[dict] = field(default_factory=list)


def mountain_sectors(grid: DemGrid, frame: PolarFrame, z0: float,
                     rings: list[tuple[float, float]] = RINGS_KM) -> list[MountainMetrics]:
    """24 горы × кольца: превышение, угол горизонта, доля возвышенностей."""
    rel = grid.data - z0
    out: list[MountainMetrics] = []
    for idx, name in enumerate(geo.MOUNTAINS):
        center = idx * geo.SECTOR_DEG
        mask_sector = sector_slice(frame.bearing, center - geo.HALF_SECTOR, center + geo.HALF_SECTOR)
        rings_out = []
        for r0, r1 in rings:
            mask = mask_sector & (frame.dist >= r0 * 1000.0) & (frame.dist < r1 * 1000.0) & frame.valid
            if not mask.any():
                rings_out.append({"r_km": [r0, r1], "max_rel_elev_m": 0.0, "mean_rel_elev_m": 0.0,
                                  "horizon_deg": 0.0, "horizon_azimuth": None, "hill_share": 0.0,
                                  "peaks": []})
                continue
            vals = rel[mask]
            dists = frame.dist[mask]
            angles = np.degrees(np.arctan2(vals - EYE_M, np.maximum(dists, 1.0)))
            imax = int(np.argmax(angles))
            rings_out.append({
                "r_km": [r0, r1],
                "max_rel_elev_m": round(float(np.nanmax(vals)), 1),
                "mean_rel_elev_m": round(float(np.nanmean(vals)), 1),
                "horizon_deg": round(float(angles[imax]), 2),
                "horizon_azimuth": round(float(frame.bearing[mask][imax]), 1),
                "hill_share": round(float(np.mean(vals > HILL_THRESHOLD_M)), 3),
                "peaks": [],
            })
        out.append(MountainMetrics(name, rings_out))
    return out


@dataclass
class SpiritMetrics:
    rel_elev_m: float
    dist_km: float
    continuity: float
    note: str = ""


def _sector_stats(frame: PolarFrame, rel: np.ndarray, center_deg: float, half_width: float,
                  r_max_km: float, azimuth_bins: int = 24, r_min_km: float = 1.0) -> SpiritMetrics:
    mask = (sector_slice(frame.bearing, center_deg - half_width, center_deg + half_width)
            & (frame.dist <= r_max_km * 1000.0) & (frame.dist >= r_min_km * 1000.0) & frame.valid)
    if not mask.any():
        return SpiritMetrics(0.0, 0.0, 0.0, "нет данных в секторе")
    vals, dists, brg = rel[mask], frame.dist[mask], frame.bearing[mask]
    keep = vals > HILL_THRESHOLD_M
    rel_max = float(vals.max())
    dist_km = float(dists[int(np.argmax(vals))] / 1000.0)   # расстояние до самой высокой точки сектора
    # непрерывность: доля азимутальных корзин, где есть возвышенность в пределах r_max
    lo = geo.normalize_deg(center_deg - half_width)
    bins = ((brg[keep] - lo) % 360.0 / (2 * half_width) * azimuth_bins).astype(int) if keep.any() else np.array([], dtype=int)
    continuity = len(np.unique(bins[bins < azimuth_bins])) / azimuth_bins if keep.any() else 0.0
    return SpiritMetrics(round(rel_max, 1), round(dist_km, 2), round(float(continuity), 2))


def four_spirits(grid: DemGrid, frame: PolarFrame, z0: float, facing_deg: float,
                 *, meso_km: float = 8.0) -> dict:
    """Четыре духа считаются от оси участка, а не от сторон света."""
    rel = grid.data - z0
    behind = geo.normalize_deg(facing_deg + 180.0)
    tortoise = _sector_stats(frame, rel, behind, 45.0, meso_km)
    bird = _sector_stats(frame, rel, facing_deg, 45.0, meso_km)
    dragon = _sector_stats(frame, rel, facing_deg - 90.0, 45.0, meso_km)   # слева от смотрящего
    tiger = _sector_stats(frame, rel, facing_deg + 90.0, 45.0, meso_km)    # справа
    return {
        "tortoise": {"rel_elev_m": tortoise.rel_elev_m, "dist_km": tortoise.dist_km, "continuity": tortoise.continuity},
        "bird": {"rel_elev_m": bird.rel_elev_m, "dist_km": bird.dist_km, "continuity": bird.continuity,
                 "open_plain_km": round(bird.dist_km if bird.dist_km else meso_km, 2), "table_hill": None},
        "dragon": {"rel_elev_m": dragon.rel_elev_m, "dist_km": dragon.dist_km, "continuity": dragon.continuity},
        "tiger": {"rel_elev_m": tiger.rel_elev_m, "dist_km": tiger.dist_km, "continuity": tiger.continuity},
        "dragon_vs_tiger": round(dragon.rel_elev_m - tiger.rel_elev_m, 1),
    }


@dataclass
class VeinResult:
    length_km: float
    drop_m: float
    bends: int
    saddles: int
    peak_xy: tuple[float, float] | None
    peak_elev: float
    path: list[tuple[float, float]]


def dragon_vein(grid: DemGrid, x0: float, y0: float, z0: float, *,
                steps_m: tuple[float, ...] = (500.0, 1000.0, 2000.0),
                max_km: float = 50.0, min_climb_m: float = 0.3, plateau_limit: int = 4) -> VeinResult:
    """Хребет вверх от участка до выполаживания.

    Из точки перебираются восемь направлений на трёх дальностях (0.5 / 1 / 2 км): берётся
    самое высокое. Если подъёма нет, разрешено несколько «плоских» шагов — иначе на
    городской равнине путь обрывается через километр. Длина пути считается по фактическим
    шагам, изгибы — с гистерезисом 60°.
    """
    path = [(x0, y0)]
    x, y, z_prev = x0, y0, z0
    bends, turn_accum = 0, 0.0
    prev_dir: float | None = None
    saddles, falling, min_since_fall = 0, False, z0
    plateau = 0
    total_len = 0.0
    last_step = steps_m[0]
    for _ in range(int(max_km * 1000 / min(steps_m))):
        best = None
        for step in steps_m:
            for a in range(0, 360, 45):
                dx = math.cos(math.radians(a)) * step
                dy = math.sin(math.radians(a)) * step
                z = grid.point_value(x + dx, y + dy)
                if math.isnan(z):
                    continue
                if best is None or z > best[2] + 1e-6:
                    best = (dx, dy, z, step)
        if best is None:
            break
        dx, dy, z_new, step = best
        if z_new < z_prev + min_climb_m:
            if plateau >= plateau_limit or z_new < z_prev - 1.0:
                break
            plateau += 1
        else:
            plateau = 0
        direction = geo.normalize_deg(math.degrees(math.atan2(dx, dy)))
        if prev_dir is not None:
            turn_accum += geo.angle_distance(direction, prev_dir)
            if turn_accum >= 60.0:
                bends += 1
                turn_accum = 0.0
        prev_dir = direction
        x, y = x + dx, y + dy
        total_len += step
        last_step = step
        path.append((x, y))
        if z_new < z_prev:
            if not falling:
                falling, min_since_fall = True, z_new
            else:
                min_since_fall = min(min_since_fall, z_new)
        elif falling and z_new >= min_since_fall + 5.0:
            saddles += 1
            falling = False
        z_prev = z_new
        if total_len > max_km * 1000:
            break
    peak_z = grid.point_value(*path[-1])
    return VeinResult(round(total_len / 1000.0, 2),
                      round(float(peak_z - z0), 1) if not math.isnan(peak_z) else 0.0,
                      bends, saddles, (path[-1][0], path[-1][1]),
                      round(float(peak_z), 1) if not math.isnan(peak_z) else 0.0, path)


def point_description(grid: DemGrid, x0: float, y0: float) -> dict:
    """Уклон, экспозиция, форма места (чаша / уступ / гребень / низина)."""
    slope, aspect = gradient(grid)
    col, row = ~grid.transform @ (x0, y0)
    r, c = int(row), int(col)
    z0 = grid.point_value(x0, y0)
    if not (0 <= r < grid.data.shape[0] and 0 <= c < grid.data.shape[1]):
        return {"slope_deg": None, "aspect_deg": None, "shape": "вне сетки", "elev_m": z0}
    local_slope = float(slope[r, c])
    local_aspect = float(aspect[r, c])
    k = max(3, int(round(200 / grid.resolution_m)) | 1)
    around = grid.data[max(0, r - k):r + k + 1, max(0, c - k):c + k + 1]
    lifted = grid.data[r, c] - np.nanmean(around)
    if lifted > 8:
        shape = "гребень"
    elif lifted < -8:
        shape = "низина"
    elif local_slope < 2:
        shape = "плоско / уступ"
    else:
        shape = "склон"
    return {"slope_deg": round(local_slope, 1), "aspect_deg": round(local_aspect, 1),
            "shape": shape, "elev_m": round(float(z0), 1),
            "lift_over_200m_m": round(float(lifted), 1)}


def measure_site(lat: float, lon: float, *, epsg: int, resolution_m: float,
                 macro_km: float = 20.0, meso_km: float = 8.0,
                 facing_deg: float | None = None, smooth_m: float = 300.0,
                 point_smooth_m: float = 90.0):
    """Полный замер рельефа: макро- и мезо-охваты в UTM."""
    from pyproj import Transformer

    from .dem import load_dem

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x0, y0 = tr.transform(lon, lat)

    macro = load_dem(lat, lon, macro_km, epsg=epsg, resolution_m=max(resolution_m, 90.0))
    macro_s = smooth(macro, smooth_m)
    z0 = macro_s.point_value(x0, y0)
    meso = load_dem(lat, lon, meso_km, epsg=epsg, resolution_m=resolution_m)
    meso_s = smooth(meso, min(smooth_m, 120.0))
    z0_meso = meso_s.point_value(x0, y0)
    if math.isnan(z0):
        z0 = z0_meso

    frame_macro = polar_frame(macro_s, x0, y0)
    sectors = mountain_sectors(macro_s, frame_macro, z0)

    result = {
        "elev_m": None if math.isnan(z0) else round(float(z0), 1),
        "elev_meso_m": None if math.isnan(z0_meso) else round(float(z0_meso), 1),
        "sectors": [{"mountain": m.name, "rings": m.rings} for m in sectors],
        "frame": frame_macro,
        "grid_macro": macro_s,
        "grid_meso": meso_s,
        # Точку считаем по сглаженному слою: на 30-метровом DSM крыши дают уклон 20° и «гребень»
        "point": point_description(smooth(meso, max(point_smooth_m, 9 * resolution_m)), x0, y0),
        "x0": x0, "y0": y0,
    }
    if facing_deg is not None:
        result["four_spirits"] = four_spirits(macro_s, frame_macro, z0, facing_deg, meso_km=meso_km)
    vein = dragon_vein(macro_s, x0, y0, z0)
    result["dragon_vein"] = {
        "to_peak": None,
        "length_km": vein.length_km,
        "drop_m": vein.drop_m,
        "bends": vein.bends,
        "saddles": vein.saddles,
        # «живой» дракон — извилистый: много изгибов на длине; прямой луч — «мёртвый»
        "alive": vein.bends >= max(3, int(vein.length_km / 10)),
        "peak_elev_m": vein.peak_elev,
    }
    return result
