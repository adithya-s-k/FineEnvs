# ECMWF HRES 10 m wind at the nearest sea cell to Barcelona's Bocana Sud, 2023-2026

`points.jsonl.gz`: the 118,472 10u and 10v messages of 2,599 ECMWF HRES runs (2023-02-05 12 UTC to 2026-01-01 12 UTC)
that the wind env's weather weeks and their calibration read, one line per message: `[grid, init YYYYMMDDHH, step,
param, value in m/s, Last-Modified]`. 1,182,834 bytes, gzip with mtime 0; `../meteocat/pins.json` pins its sha256 and
that of its body. `envs/port_week/tools/build_wind.py` builds the forecasts from it; nothing is fetched from ECMWF.

- **Source:** ECMWF open data, IFS HRES (oper and scda streams), the GRIB2 messages in s3://ecmwf-forecasts
  (https://registry.opendata.aws/ecmwf-forecasts/), via agentenv-portsim v0.4.5's fetch cache, whose
  `data/wind/weather-sources.json` (sha256 `c948aa38…6dc3`) pins the 33,550 messages the weeks use by key, byte range,
  sha256 and Last-Modified.
- **Licence:** CC BY 4.0. Attribution, as ECMWF asks:
  ```
  Contains modified ECMWF open data.
    Copyright: © 2023-2026 European Centre for Medium-Range Weather Forecasts (ECMWF).
    Source: www.ecmwf.int
    Licence Statement: This data is published under a Creative Commons Attribution 4.0 International (CC BY 4.0). https://creativecommons.org/licenses/by/4.0/
    Disclaimer: ECMWF does not accept any liability whatsoever for any error or omission in the data, their availability, or for any loss or damage arising from their use.
    This data is based on data and products of the European Centre for Medium-Range Weather Forecasts (ECMWF).
    Modified: the IFS HRES 10 m wind components (10u, 10v), oper and scda streams, steps 0-72 h, read at one grid point (0.4°: 41.2° N, 2.0° E for runs before 2024-02-28 06 UTC; 0.25°: 41.25° N, 2.25° E from then) from the GRIB2 messages in s3://ecmwf-forecasts. No GRIB data is included.
    Terms of use: https://apps.ecmwf.int/datasets/licences/general/
  ```
- **Extraction:** `python tools/wind_weather.py ecmwf-points` from the per-run cache files, or with `--fetch` from the
  bucket by HTTP range (about 85 GB and 50 minutes; needs eccodes). The runs: every 6-hourly run serving one of the 16
  weather weeks (686 runs, steps 0-72 h every 3 h, with Last-Modified, which decides whether a run was out 9 hours after
  its start), and the 00/12 UTC runs at steps 9-72 h over the calibration periods (0.4°: Feb 2024 to Jan 2025; 0.25°:
  Feb 2024 to Dec 2025). The value is the nearest sea cell's, not an interpolation, which mixes in land cells.
- **What it is and is not:** raw model wind at one point in m/s. Four messages are absent from the archive (steps 54 and
  57 of the 2025-01-21 12 UTC runs) and are null. The calibrated whole knots the agent sees are built from these values
  and the Y7 readings on your machine, and are not committed (`../meteocat/SOURCE.md`).
