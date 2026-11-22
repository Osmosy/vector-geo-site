# NOTICE — данные, лицензии и атрибуция

Код и документация этого репозитория — MIT (`LICENSE`).

Модуль скачивает чужие данные; у них свои лицензии и свои требования к атрибуции:

| Слой | Источник | Лицензия | Атрибуция |
|---|---|---|---|
| Рельеф (DEM 30 м) | Copernicus DEM GLO-30, зеркало AWS Open Data | свободная лицензия Copernicus | «produced using Copernicus WorldDEM-30 © DLR e.V.» |
| Геокодер | Nominatim (OpenStreetMap) | ODbL, политика использования Nominatim | «© OpenStreetMap contributors» |
| Вода, застройка (этап 3) | OpenStreetMap / Overpass | ODbL | «© OpenStreetMap contributors» |
| Гидрография (этап 3) | HydroRIVERS (HydroSHEDS) | свободно, в том числе коммерчески | по требованию HydroSHEDS |

Требования, которые модуль соблюдает сам:

- Nominatim — не больше одного запроса в секунду, собственный User-Agent, кеш ответов;
- строки атрибуции пишутся в каждый `site_pack.json` в поле `attribution`;
- тайлы и ответы геокодера кешируются, чтобы не дёргать сервисы повторно.

Скачанные данные в репозиторий не попадают: кеш живёт в `~/.cache/vector-geo-site`.
