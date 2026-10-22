# Crew rules, crew bases and costs (v2), and the passenger costs v3 adds

Committed: `envs/airops/core/airops_core/programs/crew_params.json`, every crew and cost parameter of the crew model
with its value, its sources (URL, retrieval date and the quote or table it comes from), a confidence, and the stand-ins
used where no public figure exists (`unsourced_stand_ins`). b6-v2's manifest pins it by sha256
(`crew_params_sha256`). The cost rates are also in `programs/cost.py` with their derivation.

- **Sources:**
  - the rules: 14 CFR part 117 (Tables A and B, the extension of 117.19, rest), 14 CFR 121.467 (flight attendants'
    duty and rest) and 121.391 (cabin minimums), as published by the eCFR; the Federal Register (78 FR 14166; 87 FR
    61452); the FAA Office of Chief Counsel's interpretation of 121.467(b)(14); FAA Order 8900.1;
  - the costs: the FAA's *Economic Values for FAA Investment and Regulatory Decisions*, Section 4, Table 4-7; BLS OEWS,
    ECEC and CPI-U; GSA and DoD per diem rates;
  - the value of a passenger's time, which turns every dollar cost into passenger-minutes in all three versions: the
    same guide's Section 1, Table 1-1, $47.10 an hour for intercity air travel, all purposes, in 2015 dollars,
    https://www.faa.gov/regulations_policies/policy_guidance/benefit_cost/econ-value-section-1-tx-time.pdf, taken to
    2025 dollars by CPI-U: $1.07 a passenger-minute (`core/cost.py`);
  - the crews: JetBlue's Form 10-K filings and press releases, ALPA's and Airline Pilot Central's JetBlue pages, union
    and press reports of crew bases and cabin complements, the EASA type certificate data sheets for each type's
    minimum flight crew, and, for reserve bands and report times, other carriers' contracts and a published study.
- **Licence:** the eCFR, the Federal Register, the FAA's and the BLS's publications and the GSA and DoD rates are works
  of the U.S. Government, public domain (17 U.S.C. § 105). Crew bases, fleet configurations, the deadhead fare and
  reserve practice are facts cited from the sources above, each with its URL; no text of theirs is redistributed beyond
  the short quotes that identify a value.
- **Extraction:** pilots under FAR 117 and flight attendants under 121.467, crew bases by date (pilots BOS, JFK, EWR,
  FLL, MCO, LAX, plus SJU from 2025-03-01; flight attendants the same six, plus SJU from 2024-12-12), reserves at the
  middle of a sourced industry band (pilots 16%, band 13-20%; flight attendants 19%, band 16-22%), and the day's costs
  in passenger-minutes at $1.07 each (2025 dollars). v3 prices each night a passenger waits away from the same lodging
  tables, lodging alone (`programs/crews.py`, `lodging`): GSA's rate for the city and season (the FY master rates files,
  https://www.gsa.gov/travel/plan-a-trip/per-diem-rates/per-diem-files), DoD's for Puerto Rico and the U.S. Virgin
  Islands, GSA's standard rate elsewhere, each taken to 2025 dollars by CPI-U.
- **What it is and is not:** the rules as published, not JetBlue's contracts: union contract rules and multi-day
  limits are not modelled, and JetBlue's pairings, rosters and reserve lists are not public, so the trips are generated
  from the published schedule.
