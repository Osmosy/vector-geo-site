"""Схемы пакета: роза 24 гор, профиль горизонта, отмывка рельефа.

Без внешних зависимостей: PNG пишется напрямую (zlib + struct), схемы — SVG.
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

import numpy as np

from . import geo


def _png_gray(path: Path, array: np.ndarray) -> None:
    """Записать 8-битный grayscale PNG без PIL."""
    h, w = array.shape
    raw = b"".join(b"\x00" + array[y].tobytes() for y in range(h))
    comp = zlib.compress(raw, 6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", comp) + chunk(b"IEND", b""))


def hillshade_png(grid, path: Path, *, azimuth: float = 315.0, altitude: float = 45.0,
                  site_px: tuple[int, int] | None = None) -> Path:
    """Отмывка рельефа: тени по уклону и экспозиции, точка участка — белым крестом."""
    z = np.array(grid.data, dtype="float64")
    z = np.where(np.isnan(z), np.nanmin(z), z)
    dy, dx = np.gradient(z, grid.resolution_m)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = math.radians(360.0 - azimuth + 90.0)
    alt = math.radians(altitude)
    shade = (np.sin(alt) * np.cos(slope)
             + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    shade = np.clip(shade, 0, 1)
    # нормализация контраста по процентилям
    lo, hi = np.percentile(shade, 2), np.percentile(shade, 98)
    shade = np.clip((shade - lo) / max(hi - lo, 1e-6), 0, 1)
    img = (shade * 255).astype("uint8")
    if site_px:
        cx, cy = site_px
        for d in range(-6, 7):
            for (yy, xx) in ((cy + d, cx), (cy, cx + d)):
                if 0 <= yy < img.shape[0] and 0 <= xx < img.shape[1]:
                    img[yy, xx] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    _png_gray(path, img)
    return path


def rose24_svg(sectors: list[dict], site: dict, path: Path, *, size: int = 640) -> Path:
    """Роза 24 гор: длина луча — максимум превышения в кольце 2–8 км, подпись — гора."""
    cx = cy = size / 2
    radius = size * 0.36
    max_elev = max([s["rings"][2]["max_rel_elev_m"] or 0.0 for s in sectors] + [1.0])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
        f'<rect width="{size}" height="{size}" fill="#0f1720"/>',
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="#2b3a4a"/>',
        f'<circle cx="{cx}" cy="{cy}" r="{radius/2:.1f}" fill="none" stroke="#22303e"/>',
        f'<text x="{cx}" y="20" fill="#8fa9c0" font-size="14" text-anchor="middle" font-family="sans-serif">'
        f'24 горы · {site.get("place", "")}</text>',
    ]
    for s in sectors:
        idx = geo.MOUNTAINS.index(s["mountain"])
        center = idx * 15.0
        ring = s["rings"][2]
        length = max(4.0, (ring["max_rel_elev_m"] or 0.0) / max_elev * radius)
        a = math.radians(90.0 - center)
        x1, y1 = cx + length * math.cos(a), cy - length * math.sin(a)
        color = "#4f8ff7" if (ring["max_rel_elev_m"] or 0.0) > 5 else "#33465c"
        if (ring["horizon_deg"] or 0.0) > 1.0:
            color = "#f7c948"
        parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{color}" stroke-width="6" stroke-linecap="round"/>')
        lx, ly = cx + (radius + 22) * math.cos(a), cy - (radius + 22) * math.sin(a)
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" fill="#9fb6cc" font-size="13" text-anchor="middle" dominant-baseline="middle" font-family="sans-serif">{s["mountain"]}</text>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#ffffff"/>')
    facing = site.get("facing_true_deg")
    if facing is not None:
        a = math.radians(90.0 - facing)
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{cx + radius * math.cos(a):.1f}" y2="{cy - radius * math.sin(a):.1f}" stroke="#ff6b6b" stroke-width="2" stroke-dasharray="6 4"/>')
        parts.append(f'<text x="{cx}" y="{size - 12}" fill="#ff6b6b" font-size="13" text-anchor="middle" font-family="sans-serif">фасад {facing:.0f}° · {geo.mountain_name(facing)}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def horizon_svg(sectors: list[dict], path: Path, *, width: int = 900, height: int = 260) -> Path:
    """Профиль горизонта 360°: угол по кольцу 2–8 км."""
    pad = 40
    # E-029: пустое кольцо — `null`; профиль строится по измеренным, а пустые не рисуются
    max_h = max([s["rings"][2]["horizon_deg"] or 0.0 for s in sectors] + [0.5]) or 0.5
    scale = (height - 2 * pad) / max_h
    pts = []
    for s in sectors:
        idx = geo.MOUNTAINS.index(s["mountain"])
        x = pad + (idx + 0.5) * (width - 2 * pad) / 24
        horizon = s["rings"][2]["horizon_deg"]
        if horizon is None:            # не измерено — не рисуем: линия через ноль соврала бы
            continue
        y = height - pad - max(0.0, horizon) * scale
        pts.append(f"{x:.1f},{y:.1f}")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#0f1720"/>',
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4f8ff7" stroke-width="2"/>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#2b3a4a"/>',
    ]
    for s in sectors:
        idx = geo.MOUNTAINS.index(s["mountain"])
        x = pad + (idx + 0.5) * (width - 2 * pad) / 24
        parts.append(f'<text x="{x:.1f}" y="{height - pad + 16}" fill="#7f9bb5" font-size="10" text-anchor="middle" font-family="sans-serif">{s["mountain"]}</text>')
    parts.append(f'<text x="{pad}" y="24" fill="#8fa9c0" font-size="13" font-family="sans-serif">Угол горизонта по 24 горам, кольцо 2–8 км (макс. {max_h:.1f}°)</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
