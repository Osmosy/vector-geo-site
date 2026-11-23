# Эталонные места

Факты берутся из карты, лоции или тега, а не из вывода кода: подгонять `expect.yaml` под
расчёт запрещено (ТЗ Б4). Каждый факт снабжён источником в самом `expect.yaml`.

| Эталон | Что проверяется | Состояние |
|---|---|---|
| `london-city` | Темза на восток, приливная, участок на внешнем берегу, петля есть и участок снаружи её (`bend_kind: loop`, `site_position: outside_concave`), слияние с Флитом, врата `width_capped` | проходит (DEM + OSM-фикстуры) |
| `london-isle-of-dogs` | петля, `site_position: inside`, приливная | проходит |
| `kingston-upon-thames` | выше границы прилива: `tidal: false` | проходит (тег OSM ближайшего куска) |
| `sankt-goar-rhine` | река `Rhein` у самого берега, течение на север, ущелье: `closure_index ≥ 0.5` | проходит (DEM с Copernicus, тайл N50/E007) |
| `nizhny-novgorod-strelka` | слияние Волги и Оки у Стрелки | проходит: слияние 1280.7 м (E-008 закрыта, `water 6.1.0+`) |
| `moscow-kremlin` | Москва-река (петля Балчуга), скрытая Неглинная (`visible: false`) | проходит (выгрузка с зеркала maps.mail.ru) |
| плоская равнина | все рельефные поля `null`/`low` | покрыто синтетикой `tests/test_water.py::test_mouth_flat_plane_is_null` |

## Как собрать фикстуры

```bash
python scripts/make_golden.py london-city            # OSM (8 и 30 км) + DEM 90 м
python scripts/make_golden.py --osm-only sankt-goar-rhine   # только OSM, без DEM
```

DEM вырезается из кеша тайлов Copernicus (`~/.cache/vector-geo-site/dem30`). Если тайлов нет,
скрипт их скачает; при недоступности S3 эталон собирается в режиме `--osm-only`, и проверки,
которым нужен рельеф (врата, видимость), пропускаются с явной причиной.

Вода в фикстуре — это уже собранный ответ: `way(around:8 км)` **плюс** члены отношений рек
(`water 6.1.0+`, см. `docs/calc/rivers.md`). Отдельная выгрузка отношений не нужна: она внутри
`osm-water.json` (у таких элементов в тегах есть `_rel`).

## Почему 30-километровая выгрузка

Граница прилива (Теддингтон) лежит в 19 км от Сити — в 8-километровую выборку не попадает.
Широкая выгрузка хранится в `fixtures/osm-water-wide.json.gz` (сжата, иначе десятки мегабайт),
и нужна только для определения приливности по положению вдоль русла.

## Происхождение фикстур

| Эталон | OSM-выгрузка | DEM |
|---|---|---|
| london-city, london-isle-of-dogs, kingston-upon-thames | 22.11.2026, Overpass (osm-water.json) | 22.11.2026, Copernicus GLO-30 |
| nizhny-novgorod-strelka | 23.11.2026, Overpass + отношения OSM | — (без DEM) |
| moscow-kremlin | 23.11.2026, maps.mail.ru:Overpass | 23.11.2026, Copernicus GLO-30 |
| sankt-goar-rhine | 23.11.2026, Overpass | 23.11.2026, Copernicus GLO-30 N50/E007 |

## Зеркала Overpass

Порядок: `maps.mail.ru` (полная планета, отвечает, когда немецкое зеркало в очереди),
`overpass-api.de`, затем `overpass.osm.ch`. Последнее держит только Швейцарию и на запросы о
Москве возвращает пустой список — пустой ответ принимается только после того, как его дали
все зеркала, иначе участок остаётся без воды навсегда (так и был «blocked» у Москвы).

## Проверка пакета целиком, офлайн

`expect.yaml` сверяет метрики через `measure_water`, но пакет можно собрать целиком без сети —
из той же выгрузки:

```bash
geosite pack "51.5138, -0.0984" --year 2018 --offline \
  --osm-file tests/golden/london-city/fixtures/osm-water.json --out /tmp/pack
geosite validate /tmp/pack/site_pack.json
```

Водный слой, склонение и климат считаются без сети, DEM — из кеша тайлов, ветер — из кеша
архива.
