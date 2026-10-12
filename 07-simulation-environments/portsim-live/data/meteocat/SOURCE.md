# Meteocat XEMA station Y7, Port de Barcelona – Bocana Sud, 2023-01 to 2026-01

`pins.json`: pins, not data. It holds each of the 37 months of Y7 readings by row count and sha256 as fetched on
2026-10-08, the sha256 of the committed ECMWF points, the sha256 (whole file and per line) of the wind tasks,
references and weather agentenv-portsim v0.4.5 built, and those of the lines tasks, manifest and references it built
at 83eca5c. Those files are generated on your machine by `envs/port_week/tools/build_wind.py` and
`envs/port_week/tools/build_lines.py` and git-ignored: `envs/port_week/tasks/dock-v1-wind/tasks.jsonl`,
`envs/port_week/tasks/wind/references.jsonl` and `envs/port_week/tasks/wind/weather.jsonl`, with the wind pack's
`manifest.json` and a `NOTICE` in each directory, which record the dates and aren't pinned; and
`envs/port_week/tasks/dock-v1-lines/{tasks.jsonl,manifest.json}` and `envs/port_week/tasks/lines/references.jsonl`,
with a `NOTICE` in each directory. The lines references are pinned without each week's solve time, which the build
doesn't write, and, for the check, without the summaries of the one solver configuration CP-SAT doesn't reproduce
from run to run (8 workers, no tie-break).

- **Source:** Meteocat XEMA station Y7, Port de Barcelona – Bocana Sud, Generalitat de Catalunya open data portal,
  dataset `nzvn-apee` ("Dades meteorològiques de la XEMA", https://analisi.transparenciacatalunya.cat/d/nzvn-apee),
  through its anonymous Socrata API.
- **Licence:** not committed; fetched on your machine under the portal's terms:
  ```
  The wind env's observed windows and forecast calibration are built on your machine by build_wind.py from the
  readings of the Servei Meteorològic de Catalunya's (Meteocat) XEMA station Y7, Port de Barcelona – Bocana Sud
  ("Dades meteorològiques de la XEMA", dataset nzvn-apee, Generalitat de Catalunya open data portal:
  https://analisi.transparenciacatalunya.cat/d/nzvn-apee). This repository holds no reading and no data derived from
  them, only the identifiers the build fixes (the 16 ISO weeks with their kind, the 15 pairs) and digests (each
  month's row count and sha256). The files the build writes keep the terms published there
  (https://www.meteo.cat/wpweb/avis-legal/;
  https://web.gencat.cat/ca/generalitat/dades-indicadors/dades-obertes/llicencies), which this repository doesn't
  relicense.
    Font: Servei Meteorològic de Catalunya (Meteocat), estació XEMA Y7; dades extretes el <fetch date>; darrera actualització <portal Last-Modified>; finestres derivades
  ```
  The build writes the Font line with your fetch date into each `NOTICE`, and the dates into the pack's manifest.
  Check those terms before you share the built files, the results and transcripts of wind runs (they repeat the
  observed windows and the calibrated forecasts), or an image or dataset that contains them.
- **Extraction:** variables 30 (VV10, the 30-minute mean wind at 10 m) and 50 (VVx10, the 3-second gust at 10 m) with
  their validation state, one query per month from 2023-01 to 2026-01. An hour is above T kn when either half-hour has
  1.045 x the mean or 0.664 x the gust above T (T = 25 for ships of 300 m or more, 30 for all). The 16 weather weeks and
  the 15 pairs dealt from them are fixed in the build, so the task ids don't move when Meteocat validates a reading.
- **What it is and is not:** the anemometer readings as published, many not yet validated. Validating a month changes
  its sha256 but not its readings; `values_sha256` pins the readings alone, and `build_wind.py fetch` reports each month
  as exact, values-match, values-changed or missing. The built pack and references must still match their pins;
  `build_wind.py verify` lists any task that doesn't.
