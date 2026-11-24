# Iowa Environmental Mesonet: METARs, TAFs and forecast discussions (v1)

Not committed: the monthly ASOS and AFOS files, under `envs/airops/cache/inputs/iem/` once `tools/build_pack.py` has
fetched them, each pinned by bytes, sha256, retrieval date and URL in b6-v1's manifest (`inputs`).

- **Source:** Iowa Environmental Mesonet (IEM), Iowa State University, https://mesonet.agron.iastate.edu/: the ASOS
  archive (routine METARs) and the AFOS text archive (TAFs and Area Forecast Discussions).
- **Licence:** public domain. The IEM disclaimer (https://mesonet.agron.iastate.edu/disclaimer.php) says its data "are
  in the public domain and may be used freely by anyone for any lawful purpose"; attributing the Iowa Environmental
  Mesonet of Iowa State University is appreciated, and given here. The METARs, TAFs and forecast discussions themselves
  are U.S. National Weather Service products (17 U.S.C. § 105).
- **Extraction:** the hub's routine METARs (news of kind `metar`), its TAFs as issued, with amendments (`taf`), and the
  `.AVIATION` section of the local NWS office's forecast discussion (`outlook`), each at its issue time.
- **What it is and is not:** v1's weather is this text; the radar and 3D weather in the Space's viewer are for people.
  v2's desks get no weather text: the weather reaches them through the FAA's programs and each flight's recorded delay.
