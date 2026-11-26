"""Схема пакета участка `site_pack/2`.

Правило модуля: здесь только измерения. Никаких «благо / неблаго», никаких рекомендаций —
оценка остаётся навыку vector-fengshui-bazi.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_ID = "site_pack/2"

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
    hydro: dict | None = None
    # H1: полушарие и «около экватора» — факты, а не толкование
    hemisphere: Literal["N", "S"] | None = None
    equatorial: bool | None = None


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
    # D3: у ближнего кольца оба вида рельефа — поверхность (крыши как городские 砂) и земля
    dsm: dict | None = None
    dtm: dict | None = None


class Sector(BaseModel):
    mountain: str
    from_deg: float
    to_deg: float
    rings: list[Ring]
    water: list[str] = Field(default_factory=list)
    water_km: float | None = None
    wind_share: float | None = None
    # W4: доля воды в клине кольца 0.5–2 км по двум источникам, выбранное значение и его цена
    water_share: float | None = None
    water_share_osm: float | None = None
    water_share_wc: float | None = None
    water_share_confidence: Literal["high", "medium", "low"] | None = None
    water_share_reason: Literal["axis_line", "bridge", "dock_lock", "tidal_flat",
                                "stale_osm", "shadow"] | None = None


class Spirit(BaseModel):
    rel_elev_m: float
    dist_km: float | None = None
    continuity: float
    note: str | None = None


class FourSpirits(BaseModel):
    dem_layer: str | None = None      # 砂: относительные высоты хранителей — по этому слою
    """H4: стороны относительно оси участка, без имён школ.

    Кто из них 玄武, 朱雀, 青龍 и 白虎 — решает навык по школе: в пакете только `back`,
    `front`, `left`, `right`, чтобы зеркальная интерпретация не требовала пересчёта данных.
    """
    back: Spirit
    front: Spirit
    left: Spirit
    right: Spirit
    left_vs_right: float


class DemFile(BaseModel):
    """D2: паспорт слоя рельефа. `kind` говорит, что это по природе, а не откуда скачано."""
    source: str
    kind: Literal["dsm", "dsm_smoothed", "dtm"]
    resolution_m: float | None = None
    license: str | None = None
    vertical_datum: str | None = None
    datum_shift_m: float | None = None


class PointMetrics(BaseModel):
    dem_layer: str | None = None      # 穴: уклон, аспект и подъём считаны по этому слою
    slope_deg: float | None = None
    aspect_deg: float | None = None
    shape: str | None = None
    elev_m: float | None = None
    lift_over_200m_m: float | None = None


class DragonVein(BaseModel):
    dem_layer: str | None = None      #  жила: подъём и перевалы считаны по этому слою
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
    dem_layer: str | None = None      # врата: высоты склонов и замкнутость — по этому слою
    mountain: str
    closure_index: float | None = None
    closure_reason: str | None = None
    closure_confidence: Confidence | None = None
    enclosure: float | None = None
    width_m: float | None = None
    left_hill_m: float | None = None
    right_hill_m: float | None = None
    width_capped: bool = False
    capped_sides: int = 0


class WaterObject(BaseModel):
    dem_layer: str | None = None      # течение по рельефу: уклон русла снят с этого слоя
    id: str
    name: str | None = None
    pieces: int = 1
    hidden_share: float | None = None
    visible: bool | None = None
    kind: Literal["river", "stream", "canal", "lake", "sea", "dock", "culvert", "road"]
    order: int | None = None
    discharge_m3s: float | None = None
    hydro_distance_m: float | None = None
    in_mountain: str | None = None
    out_mountain: str | None = None
    nearest_m: float | None = None
    bend: Literal["outer", "inner", "straight"] | None = None
    straight_at_site: bool | None = None
    tidal: bool | None = None
    tidal_source: str | None = None   # osm_tag | known_limit: откуда взят ответ
    tidal_reason: str | None = None   # почему tidal null: нет выгрузки 30 км, границы нет и т. п.
    confluences: list[Confluence] = Field(default_factory=list)
    # вход русла в водоём: не слияние двух русел, поэтому отдельный список (E-017)
    waterbody_links: list[dict] = Field(default_factory=list)
    # линия русла или контур водоёма: в слияниях участвуют только линии (E-017)
    geom: Literal["line", "area"] = "line"
    # поля водоёма: у контура нет входа и выхода воды, изгиба и петли (E-018)
    nearest_azimuth_deg: float | None = None
    nearest_mountain: str | None = None
    area_m2: float | None = None
    site_inside: bool | None = None
    riverbank_of: str | None = None
    dock_channel_of: str | None = None
    water_mouth: WaterMouth | None = None
    outflow_visible: bool | None = None
    curvature_radius_m: float | None = None
    bend_kind: Literal["bend", "loop", "straight"] | None = None
    loop: dict | None = None
    site_position: Literal["inside", "outside_convex", "outside_concave"] | None = None
    between_branches: bool | None = None
    flow_ways_share: float | None = None
    crossings_2km: int | None = None
    crossing_angle_deg: float | None = None
    # длина русла внутри круга 2 км и причина совпадения гор входа и выхода (E-020)
    length_2km_m: float | None = None
    same_mountain_reason: str | None = None
    flow_osm_dir: str | None = None
    flow_dem_dir: str | None = None
    flow_chosen: str | None = None
    flow_rule: int | None = None
    flow_confidence: Confidence | None = None
    flow_agree: bool | None = None
    dem_slope_m_per_km: float | None = None
    dem_slope_snr: float | None = None
    dem_reason: str | None = None
    tidal_source: str | None = None
    source: str | None = None


class SunNoon(BaseModel):
    event: Literal["june_solstice", "december_solstice", "equinox"]
    date: str
    altitude_deg: float
    azimuth_deg: float
    mountain: str
    side8: Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    utc_hour: float | None = None


class SunIllumination(BaseModel):
    dem_layer: str | None = None
    hours_by_side: dict
    total_hours: float | None = None
    blocked_hours: float | None = None
    step_min: int | None = None
    source: str | None = None


class SunBlock(BaseModel):
    """H2: солнце как факт — полдни, сторона полудня, счёт дней, освещённость."""
    source: str
    latitude_abs: float
    noons: list[SunNoon] = Field(default_factory=list)
    noon_side: dict
    days_noon_north: int
    days_noon_south: int
    by_side8: dict = Field(default_factory=dict)
    illumination: SunIllumination | None = None


class Exposure(BaseModel):
    dem_layer: str | None = None      # видимость и замкнутость посчитаны по этому слою
    enclosure_index: float | None = None
    windy_mountains: list[str] = Field(default_factory=list)
    sky_view_factor: float | None = None
    winter_sun_hours: float | None = None
    wind_prevailing_deg: float | None = None
    wind_prevailing_mountain: str | None = None
    wind_speed_ms: float | None = None
    wind_calm_share: float | None = None
    wind_days: int | None = None
    wind_seasonal: dict | None = None
    wind_source: str | None = None
    # H3: откуда дует холодный ветер в холодный сезон полушария
    cold_wind: dict | None = None


class LandCover(BaseModel):
    """Покров: вода — по OSM, остальное — по ESA WorldCover 10 м (W2).

    `water` и `water_wc` стоят рядом намеренно: расхождение двух источников — это то, что
    калибровка обязана объяснить, а не спрятать.
    """
    built: float | None = None
    tree: float | None = None
    water: float | None = None
    open: float | None = None
    water_wc: float | None = None
    source: str | None = None
    ring_km: list[float] | None = None


class Quality(BaseModel):
    dem: str
    dem_source: str | None = None      # copernicus | srtm: по числу источника видно, чем считали
    dem_kind: Literal["dsm", "dsm_smoothed", "dtm"] | None = None
    dem_resolution_m: float | None = None
    # П7: уверенность полей по рельефу — по природе слоя и застройке, не по имени источника
    dem_confidence: Confidence | None = None
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    valid: bool | None = None
    violations: list[dict] = Field(default_factory=list)
    partial: bool = False


class Algo(BaseModel):
    """Версии алгоритмов, которые писали в пакет (П5)."""
    water: str | None = None
    terrain: str | None = None
    geocode: str | None = None
    climate: str | None = None
    sun: str | None = None
    pack: str | None = None


class Files(BaseModel):
    water_geojson: str | None = None
    sha256: dict[str, str] = Field(default_factory=dict)
    # откуда взяты слои: недоступный источник должен быть виден, а не подразумеваться (G5)
    land_cover: dict | None = None
    dem: DemFile | None = None
    dtm: DemFile | None = None


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
    sun: SunBlock | None = None
    exposure: Exposure | None = None
    land_cover: LandCover | None = None
    water_share_rule: str | None = None
    quality: Quality
    algo: Algo = Field(default_factory=Algo)
    files: Files = Field(default_factory=Files)
    summary: dict | None = None
    attribution: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


def json_schema() -> dict:
    return SitePack.model_json_schema(by_alias=True)
