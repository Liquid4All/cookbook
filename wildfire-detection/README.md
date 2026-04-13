# Wildfire Detection with Small Vision Language Models

Example on how to build a wildfire detection system using Small Vision Language Models.

## Data

Historical Sentinel-2 imagery is fetched via [SimSat](https://github.com/DPhi-Space/SimSat) for the following well-documented severe wildfire events. Coordinates are approximate centroids; use a bounding box of at least 0.2 degrees around each. Query within 5 days of the start date for active fire, and 2-4 weeks after for burned scar.

| # | Name | Country | Start date | Lat | Lon | Burned area (ha) | Notes |
|---|------|---------|------------|-----|-----|-----------------|-------|
| 1 | Camp Fire | USA (CA) | 2018-11-08 | 39.81 | -121.53 | 62,000 | Destroyed Paradise; dense smoke, urban interface |
| 2 | Dixie Fire | USA (CA) | 2021-07-13 | 40.00 | -121.20 | 390,000 | Largest single CA fire; long duration |
| 3 | Creek Fire | USA (CA) | 2020-09-04 | 37.20 | -119.20 | 153,000 | Pyrocumulonimbus events; intense smoke |
| 4 | Bootleg Fire | USA (OR) | 2021-07-06 | 42.40 | -121.10 | 163,000 | Generated its own weather |
| 5 | Fort McMurray | Canada | 2016-05-01 | 56.73 | -111.38 | 590,000 | Massive boreal fire; early Sentinel-2 coverage |
| 6 | Nova Scotia fires | Canada | 2023-05-28 | 44.70 | -63.80 | 25,000 | Unusual Eastern Canada fire |
| 7 | Pedrógão Grande | Portugal | 2017-06-17 | 39.90 | -8.15 | 45,000 | Deadliest European fire in decades |
| 8 | Alexandroupolis | Greece | 2023-08-19 | 41.20 | 26.30 | 81,000 | Largest EU fire on record at time |
| 9 | Mugla/Antalya | Turkey | 2021-07-28 | 36.80 | 28.50 | 140,000 | Intense Mediterranean fire |
| 10 | Gospers Mountain | Australia | 2019-10-26 | -33.00 | 150.50 | 500,000 | Part of Black Summer; months-long |
| 11 | Amazon (Para state) | Brazil | 2019-08-01 | -3.00 | -52.00 | N/A | Deforestation fires; very high smoke density |
| 12 | Lahaina, Maui | USA (HI) | 2023-08-08 | 20.87 | -156.68 | 880 | Small area but extreme intensity, wind-driven |
