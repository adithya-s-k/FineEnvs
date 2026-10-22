# DOT's refund rule, 14 CFR 260.2 (v3)

Nothing committed but the rule: v3's passenger system takes a flight 3 hours or more earlier than the one booked only
when no other seat is left (`programs/pax.py`, `fly`).

- **Source:** U.S. DOT, 14 CFR part 260, "Refunds for Airline Fare and Ancillary Service Fees", §260.2, the definition
  of a significantly delayed or changed flight, as published by the eCFR,
  https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260/section-260.2; issued by the final rule
  "Refunds and Other Consumer Protections", 89 FR 32760 (2024-04-26, effective 2024-06-25, compliance for ticket
  refunds from 2024-10-28),
  https://www.federalregister.gov/documents/2024/04/26/2024-07177/refunds-and-other-consumer-protections.
- **Licence:** public domain, works of the U.S. Government (17 U.S.C. § 105).
- **Extraction:** a departure "three hours or more" earlier than the original, on a domestic itinerary, is a
  significant change the passenger may refuse for a refund. So a disrupted passenger is moved onto a JetBlue flight
  leaving 3 hours or more before the one booked only when no other seat is left that day or in the next five days;
  otherwise such a move would be a refund. The line is applied on every date, including those before 2024-10-28.
- **What it is and is not:** the federal floor for refunds, not JetBlue's practice: airlines do rebook passengers onto
  earlier flights, and v3 takes the line as the point past which a passenger would decline.
