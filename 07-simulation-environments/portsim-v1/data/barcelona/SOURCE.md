# Port of Barcelona container calls, 2024

`container_calls_2024.csv`: the 1,784 completed container-ship calls of 2024 at quays 36A (Terminal Catalunya,
BEST) and 24B (APM Terminals Barcelona), one row per call, with the quay sections the port assigned (`sections`, the
port's own numbering), ETA/ETD in UTC, ship dimensions and previous/next port.

- **Source:** Port of Barcelona open data portal, https://opendata.portdebarcelona.cat/ (port calls dataset), via the
  2024 snapshot in https://github.com/alberto-santini/berth-allocation-problems (`generator/raw/Barcelona_2024.csv`).
- **Licence:** CC BY-SA 4.0 (Port de Barcelona). This subset and the task pack derived from it are shared under the
  same licence. Attribution: "Contains data from the Port de Barcelona open data portal".
- **Extraction:** rows with `ANYESCALA == 2024`, `ESCALAESTAT == "Finalitzada"`, `VAIXELLTIPUS == "Portacontenidors"`,
  `MOLLCODI in {36A, 24B}`, a terminal and berth sections; columns renamed to English; decimal commas converted.
- **What it is and is not:** ETA/ETD are effectively berthing and unberthing times (the 2024 record has only 21
  overlapping pairs, all 1-hour or 1-section noise), so the record is the plan the port executed. It has no crane
  counts, moves or waiting times at anchorage.
