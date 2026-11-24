# FAA Air Traffic Control System Command Center advisories

Not committed: the archive's day lists and advisory pages, under `envs/airops/cache/inputs/atcscc/` once fetched. Each
UTC day and hub is pinned by the sha256 of its parsed records (advisory number, header, message, signature and
effective time, as sorted-key compact JSON): b6-v1's manifest lists them (`inputs`), and each b6-v2 date names the
days it read (`task.sources.advisories`).

- **Source:** the ATCSCC Advisories Database, https://www.fly.faa.gov/adv/adv_list (a UTC day's list) and
  /adv/adv_otherdis (each advisory).
- **Licence:** public domain, works of the U.S. Government (17 U.S.C. § 105).
- **Extraction:** every advisory whose control element is JFK or Boston: ground delay programs, ground stops, their
  revisions, extensions and cancellations, and arrival delay notices, each at its signature time. v1 delivers them as
  raw text, news of kind `advisory` from the FAA Command Center. v2 reads them twice: as issued, for what the desks are
  told and the FAA's slots as known at a watch, and as they ran, for the truth (`programs/advisories.py`), with the
  program's scope, rates and exemptions parsed from the text.
- **What it is and is not:** what the Command Center published. The slots it implies are modelled: shared out by
  schedule over every carrier's arrivals plus an unseen stream (`programs/rbs.py`, `programs/unseen.py`); EDCTs and
  compression are the model's, not the FAA's records.
