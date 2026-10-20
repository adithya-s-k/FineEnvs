# FAA airport data and airport traffic counts (v2)

Not committed: the NASR airport file and the two capacity profiles, under `envs/airops/cache/inputs/` once
`tools/world_check.py` has fetched them, each pinned by sha256 in every b6-v2 date (`task.sources.nasr`,
`task.sources.capacity`).

- **Source:**
  - FAA NASR 28-day subscription, `APT_BASE.csv`, cycle 2026-10-01,
    https://nfdc.faa.gov/webContent/28DaySub/extra/01_Oct_2026_APT_CSV.zip;
  - FAA Airport Capacity Profiles, JFK (2014),
    https://www.faa.gov/sites/faa.gov/files/airports/planning_capacity/profiles/JFK-Airport-Capacity-Profile-2014.pdf,
    and BOS (2019),
    https://www.faa.gov/sites/faa.gov/files/airports/planning_capacity/profiles/BOS-Airport-Capacity-Profile-2019.pdf;
  - as a check only, typed into `programs/unseen.py` with their sources: the Port Authority of New York and New
    Jersey's 2024 Airport Traffic Report (JFK's movements) and Massport's monthly traffic summaries (Boston's arrivals).
- **Licence:** the FAA files are public domain, works of the U.S. Government (17 U.S.C. § 105). The Port Authority's
  and Massport's counts are facts cited from their published reports; no part of those reports is redistributed.
- **Extraction:** NASR gives each airport's ARTCC, which decides whether a program's scope covers a flight's origin;
  the capacity profiles give the hubs' arrival rates; the monthly counts check the size of the unseen traffic stream
  that BTS doesn't record (foreign, cargo, charter and non-reporting carriers).
- **What it is and is not:** planning figures and published totals, not the FAA's own slot lists.
