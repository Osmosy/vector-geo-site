"""Сводка живых прогонов по миру: что работает, а что ломается вне Европы."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def row(path: Path) -> dict:
    p = json.loads(path.read_text())
    s = p.get("site") or {}
    sun = p.get("sun") or {}
    q = p.get("quality") or {}
    lc = p.get("land_cover") or {}
    water = p.get("water") or []
    named = [w for w in water if w.get("name")]
    tidal = [w for w in water if w.get("tidal") is True]
    qs = q.get("warnings") or []
    return {
        "point": path.parent.name,
        "lat": s.get("lat"), "lon": s.get("lon"),
        "crs": s.get("grid_crs"), "crs_kind": s.get("grid_crs_kind"),
        "conv": s.get("convergence_deg"),
        "test": sun.get("used_test_site"),
        "noon": (sun.get("noon_side") or {}).get("side8"),
        "dn": sun.get("days_noon_north"), "ds": sun.get("days_noon_south"),
        "eq": s.get("equatorial"),
        "dem": (p.get("files") or {}).get("dem", {}).get("resolution_m"),
        "dtm": bool((p.get("files") or {}).get("dtm")),
        "dem_conf": q.get("dem_confidence"),
        "water_osm": lc.get("water"), "water_wc": lc.get("water_wc"),
        "objects": len(water), "named": len(named), "tidal": len(tidal),
        "tidal_src": sorted({w.get("tidal_source") for w in tidal if w.get("tidal_source")}),
        "seas": sum(1 for w in water if w.get("seasonal") is True),
        "seas_no": sum(1 for w in water if w.get("seasonal") is False),
        "partial_osm": q.get("partial_osm"), "partial_ovp": q.get("partial_overpass"),
        "dem_shift": None,
        "warn": len(qs),
        "codes": sorted({w.split(":")[0] for w in qs}),
    }


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]]
    rows = []
    for d in dirs:
        f = d / "site_pack.json" if d.is_dir() else d
        try:
            rows.append(row(f))
        except Exception as exc:
            rows.append({"point": d.name, "error": str(exc)})
    keys = ["point", "lat", "lon", "crs", "crs_kind", "conv", "noon", "dn", "ds", "eq",
            "dem", "dtm", "dem_conf", "water_osm", "water_wc", "objects", "named", "tidal",
            "tidal_src", "seas", "seas_no", "partial_osm", "partial_ovp", "warn", "codes",
            "error"]
    for r in rows:
        line = []
        for k in keys:
            if k in r and r[k] is not None:
                v = r[k]
                if isinstance(v, float):
                    v = f"{v:.3f}"
                line.append(f"{k}={v}")
        print("  ".join(line))
        print()


if __name__ == "__main__":
    main()
