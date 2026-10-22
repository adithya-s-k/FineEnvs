# BTS on-time, Form 41 B-43, T-100 and DB1B records

Not committed: the monthly files the packs are built from, under `envs/airops/cache/inputs/bts/` once
`tools/build_pack.py` (v1) or `tools/world_check.py` (v2) has fetched them. Each is pinned by sha256: b6-v1's manifest
lists every file it read with its bytes, sha256, retrieval date and URL (`tasks/b6-v1/manifest.json`, `inputs`), and
each b6-v2 date names the sha256 of the month and year files it was built from (`task.sources`).

- **Source:** U.S. DOT Bureau of Transportation Statistics, TranStats:
  - Reporting Carrier On-Time Performance (1987-present), https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ,
    a zip per month;
  - Air Carrier Financial Reports (Form 41), Schedule B-43 Inventory, https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=GEH,
    a file per year;
  - T-100 Domestic Segment (U.S. Carriers), https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIM, a file
    per month;
  - the Airline Origin and Destination Survey (DB1B), https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFI, for one
    figure in v1 and v2: the median fare, $235 in Q1 2025, in the 523 passenger-minutes a cancelled passenger costs
    (`core/cost.py`). v3 reads the survey at length, DB1C too, for its passengers:
    [../dot-origin-destination/SOURCE.md](../dot-origin-destination/SOURCE.md).
- **Licence:** public domain, works of the U.S. Government (17 U.S.C. § 105); catalog.data.gov lists the datasets under
  the U.S. Public Domain label.
- **Extraction:** from the on-time data, every JetBlue flight of the date with its tail, scheduled and actual times,
  cancellation code and delay causes; a flight's own delay is its departure delay less the late-aircraft part (v2: less
  the program part too), and a cancelled flight's is the 75th percentile of flown flights near it. v1's hourly slot caps
  are JetBlue's ration-by-schedule share of what all carriers moved at the airport; v2 takes the other carriers'
  arrivals at JFK and Boston as the traffic in the FAA's programs. B-43 gives each tail's model and seats (the nearest
  year first); T-100 gives JetBlue's load factors by route and month, so passengers = seats × load factor. v3 books
  each flight's passengers the same way, and for the five days after each date takes every flight the on-time data has
  JetBlue scheduled, with its seats left after its own T-100 load, as the seats a passenger with none that day can
  take (`build/nextday.py`; a day whose month the published files don't reach repeats the latest day built).
- **What it is and is not:** the record of what flew. It has no passengers' itineraries, no crews and no gates; the
  crews are generated (`../crew-rules-and-costs/SOURCE.md`), and so are v3's itineraries
  (`../dot-origin-destination/SOURCE.md`).
