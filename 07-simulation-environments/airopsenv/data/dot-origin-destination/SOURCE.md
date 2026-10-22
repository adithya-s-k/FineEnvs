# DOT Airline Origin and Destination Survey: DB1B Market and DB1C Market (v3)

Committed: the three extracts v3's passengers are generated from, in `envs/airops/tasks/pax/`, each pinned by sha256 in
b6-v3's manifest (`tasks/b6-v3/manifest.json`, `pax_sha256`):

- `b6-transfers.csv`: per period and connecting station (JFK, BOS, FLL, MCO, SJU), JetBlue's departing passengers to
  each destination and those among them who arrived on an earlier JetBlue flight, by origin;
- `b6-dwell.csv`: DB1C's JetBlue-to-JetBlue connections at each station by scheduled connection time, in 15-minute bins
  up to 6 hours, per month and pooled;
- `b6-fares.csv`: per period, the passenger-weighted median one-way market fare on JetBlue ($163.48 to $203.40, each
  period's own dollars) and the sample passengers it is taken over.

The survey files themselves are not committed; `envs/airops/tools/survey_shares.py` reads them into these extracts
(as the plugin's `scripts/pax_shares.py`, which wrote the committed ones).

- **Source:** U.S. DOT, Bureau of Transportation Statistics, Airline Origin and Destination Survey,
  https://www.bts.gov/topics/airlines-and-airports/origin-and-destination-survey-data:
  - DB1B Market, a 10% sample of airline tickets, quarterly, up to 2025 Q2, when its collection ended:
    https://transtats.bts.gov/PREZIP/Origin_and_Destination_Survey_DB1BMarket_{YYYY}_{Q}.zip; 10 quarters, 2022 Q4 to
    2025 Q2 without 2024 Q4;
  - DB1C Market, a 40% sample, monthly from July 2025, with each connection's scheduled time:
    https://ostrapeispubdownloadprod.blob.core.windows.net/ostrapeis-pub-download-prod/ond40/products/db1c_public/DB1C.MARKET.{YYYYMM}.{DDMONYYYY}.zip
    (the release suffix from https://www.bts.gov/topics/airlines-and-airports/origin-and-destination-survey-data-market);
    7 months, 2025-07, 2025-08, 2025-12, 2026-01, 2026-02, 2026-06 and 2026-07.
- **Licence:** public domain, works of the U.S. Government (17 U.S.C. § 105).
- **Extraction:** each market's airports and operating carriers split into segments; a JetBlue-operated segment leaving
  a connecting station counts as a connection from `orig` when the market's segment before it is JetBlue-operated too
  and flies from `orig` to that station (a surface leg is no connection). Passengers weight every row. DB1C's
  `DwellTimeGroup` gives the connection times. Fares: DB1B's `MktFare` (bulk fares left out) and DB1C's `MktAmount`,
  both including taxes and fees, fares under $10 (award and zero-fare tickets) left out. Each date takes its own
  period, the DB1B quarter up to 2025 Q2 and the DB1C month after (`programs/pax.py`, `period`); a refunded passenger's
  fare is taken to 2025 dollars by CPI-U.
- **What it is and is not:** a sample of tickets, domestic itineraries only (U.S. points, Puerto Rico and the U.S.
  Virgin Islands included), not JetBlue's bookings. v3's passengers are generated from it: who connects where, in what
  share and how long they connect. International connections, interline connections and itineraries of three or more
  legs are not generated.
