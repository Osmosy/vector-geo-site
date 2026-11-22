"""Схема пакета участка `site_pack/1`.

Правило модуля: здесь только измерения. Никаких «благо / неблаго», никаких рекомендаций —
оценка остаётся навыку vector-fengshui-bazi.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_ID = "site_pack/1"

Precision = Literal["point", "building", "district", "settlement"]
Confidence = Literal["high", "medium", "low"]
Kind = Literal["point", "address", "settlement", "district"]


class Query(BaseModel):
    input: str
    kind: Kind
    resolved_by: str
    alternatives: list[dict] = Field(default_factory=list)


class Site(BaseModel):
    lat: float
    lon: float
    elev_m: float
    precision: Precision
    utm_epsg: int
    boundary: str | None = None


class Orientation(BaseModel):
    facing_true_deg: float
    declination_deg: float | None = None
    facing_mag_deg: float | None = None
    facing_mountain: str
    sitting_mountain: str
    source: Literal["user_compass", "osm_building", "model"]
    confidence: Confidence
    near_boundary: bool = False


class ChartYear(BaseModel):
    value: int
    source: str


class Ring(BaseModel):
    r_km: tuple[float, float]
    max_rel_elev_m: float
    mean_rel_elev_m: float
    horizon_deg: float
    water_share: float | None = None
    built_share: float | None = None
    peaks: list[str] = Field(default_factory=list)


class Sector(BaseModel):
    mountain: str
    from_deg: float
    to_deg: float
    rings: list[Ring]
    water: list[str] = Field(default_factory=list)
    wind_share: float | None = None


class Spirit(BaseModel):
    rel_elev_m: float
    dist_km: float | None = None
    continuity: float
    note: str | None = None


class FourSpirits(BaseModel):
    tortoise: Spirit
    bird: Spirit
    dragon: Spirit
    tiger: Spirit
    dragon_vs_tiger: float


class PointMetrics(BaseModel):
    slope_deg: float | None = None
    aspect_deg: float | None = None
    shape: str | None = None
    elev_m: float | None = None
    lift_over_200m_m: float | None = None


class DragonVein(BaseModel):
    to_peak: str | None = None
    length_km: float
    drop_m: float
    bends: int
    saddles: int
    alive: bool | None = None


class Confluence(BaseModel):
    with_: str = Field(alias="with")
    dist_m: float
    mountain: str

    model_config = {"populate_by_name": True}


class WaterMouth(BaseModel):
    mountain: str
    closure_index: float
    width_m: float | None = None
    left_hill_m: float | None = None
    right_hill_m: float | None = None


class WaterObject(BaseModel):
    id: str
    name: str | None = None
    kind: Literal["river", "stream", "canal", "lake", "sea", "dock", "culvert", "road"]
    order: int | None = None
    in_mountain: str | None = None
    out_mountain: str | None = None
    nearest_m: float | None = None
    bend: Literal["outer", "inner", "straight"] | None = None
    straight_at_site: bool | None = None
    tidal: bool | None = None
    confluences: list[Confluence] = Field(default_factory=list)
    water_mouth: WaterMouth | None = None
    outflow_visible: bool | None = None
    curvature_radius_m: float | None = None
    loop: dict | None = None
    between_branches: bool | None = None
    flow_ways_share: float | None = None
    flow_dem_dir: str | None = None
    flow_osm_dir: str | None = None
    flow_agree: bool | None = None
    source: str | None = None


class Exposure(BaseModel):
    enclosure_index: float | None = None
    windy_mountains: list[str] = Field(default_factory=list)
    sky_view_factor: float | None = None
    winter_sun_hours: float | None = None


class LandCover(BaseModel):
    built: float | None = None
    tree: float | None = None
    water: float | None = None
    open: float | None = None


class Quality(BaseModel):
    dem: str
    dem_resolution_m: float | None = None
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SitePack(BaseModel):
    schema_id: str = Field(default=SCHEMA_ID, alias="schema")
    query: Query
    site: Site
    orientation: Orientation
    chart_year: ChartYear
    sectors: list[Sector] = Field(default_factory=list)
    point: PointMetrics | None = None
    four_spirits: FourSpirits | None = None
    dragon_vein: DragonVein | None = None
    water: list[WaterObject] = Field(default_factory=list)
    roads: list[WaterObject] = Field(default_factory=list)
    exposure: Exposure | None = None
    land_cover: LandCover | None = None
    quality: Quality
    attribution: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


def json_schema() -> dict:
    return SitePack.model_json_schema(by_alias=True)
