# NOTICE — данные, лицензии и атрибуция

Код и документация этого репозитория — MIT (`LICENSE`).

Модуль скачивает чужие данные; у них свои лицензии и свои требования к атрибуции:

| Слой | Источник | Лицензия | Атрибуция |
|---|---|---|---|
| Рельеф (DEM 30 м) | Copernicus DEM GLO-30, зеркало AWS Open Data | свободная лицензия Copernicus | «produced using Copernicus WorldDEM-30 © DLR e.V.» |
| Геокодер | Nominatim (OpenStreetMap) | ODbL, политика использования Nominatim | «© OpenStreetMap contributors» |
| Вода, застройка (этап 3) | OpenStreetMap / Overpass | ODbL | «© OpenStreetMap contributors» |
| Гидрография (этап 3) | HydroRIVERS (HydroSHEDS) | свободно, в том числе коммерчески | по требованию HydroSHEDS |
| Рельеф — земля (DTM) | AHN4 (Rijkswaterstaat / PDOK), Нидерланды | CC BY 4.0 | «AHN4, Rijkswaterstaat» |
| Рельеф — земля (DTM) | GEDTM30, OpenGeoHub (не подключён: домен недоступен с машины) | CC BY 4.0 | «GEDTM30, OpenGeoHub» |
| Рельеф — земля (DTM) | Environment Agency LIDAR, Англия (не подключён: 429) | Open Government Licence v3 | «© Environment Agency» |
| Покров (этап 3) | ESA WorldCover 2021 v200, 10 м, тайлы 3° | CC BY 4.0 | «ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium» |

Требования, которые модуль соблюдает сам:

- Nominatim — не больше одного запроса в секунду, собственный User-Agent, кеш ответов;
- строки атрибуции пишутся в каждый `site_pack.json` в поле `attribution`;
- вырезки WorldCover (16 км вокруг эталона) лежат в `tests/golden/*/fixtures/worldcover.tif`
  как CC BY 4.0 — с этой пометкой в `NOTICE` и в `tests/golden/README.md`;
- тайлы и ответы геокодера кешируются, чтобы не дёргать сервисы повторно.

Почему не FABDEM и не FathomDEM. Оба — готовые DTM, и оба распространяются под
**CC BY-NC** (некоммерческая лицензия). Приёмка раздела 4 требует, чтобы источников с запретом
коммерческого использования в цепочке не было, поэтому они не используются: лицензия — часть
контракта, а не мелкий шрифт. GEDTM30 выбран именно потому, что он CC BY 4.0.

Скачанные данные в репозиторий не попадают: кеш живёт в `~/.cache/vector-geo-site`.
