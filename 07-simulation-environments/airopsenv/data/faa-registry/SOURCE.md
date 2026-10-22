# FAA Aircraft Registry, releasable database

Not committed: `ReleasableAircraft.zip`, under `envs/airops/cache/inputs/faa/` once fetched, pinned by sha256 in
b6-v1's manifest (`inputs`) and in each b6-v2 date (`task.sources.faa_registry`).

- **Source:** FAA Civil Aviation Registry, https://registry.faa.gov/database/ReleasableAircraft.zip.
- **Licence:** public domain, a work of the U.S. Government (17 U.S.C. § 105).
- **Extraction:** the type of a tail that Form 41 B-43 misses (MASTER, then DEREG, through ACFTREF); its seats are then
  the family's median in B-43.
- **What it is and is not:** registrations, not fleet assignments: which aircraft flew which leg comes from the on-time
  records.
