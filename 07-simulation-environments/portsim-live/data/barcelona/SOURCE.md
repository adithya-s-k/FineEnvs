# Port of Barcelona calls of every ship type, 2024

`Barcelona_2024.csv` (not committed): the 10,928 stops of the port's 2024 calls of every ship type, the source of the
marine pack's other traffic (`envs/port_week/tasks/dock-v1-marine/`, rules.marine.movements). It is 4,712,806 bytes,
sha256 `123012c078e66a5978d55bcedd4203ef85c4282ceefc0e3d6b01ebdc94207d94`. `tools/marine_references.py` downloads it
once into `~/.cache/portsim-live/raw/8e726a4/` and refuses any other file.

- **Source:** Port of Barcelona open data portal, https://opendata.portdebarcelona.cat/ (port calls dataset), via the
  2024 snapshot `generator/raw/Barcelona_2024.csv` in https://github.com/alberto-santini/berth-allocation-problems at
  commit 8e726a47625da99eec72f911b16a0b76a78a8d88.
- **Licence:** CC BY-SA 4.0 (Port de Barcelona). The marine pack and the marine references derived from it are shared
  under the same licence. Attribution: "Contains data from the Port de Barcelona open data portal".
- **Extraction:** every call that doesn't stop at the task's quay, from hour 0 to hour 264 of the week, ships of 45 m
  or more: a movement each time a ship berths, and one when it leaves a berth for the anchorage (90A) or the sea, as
  `[hour, type, length_m]`; plus the departures of the ships alongside at hour 0. A stop served by two terminal
  operators is listed once per operator and counted once. The pack's manifest sets out each rule and its basis.
- **What it is and is not:** the berthing and leaving times the port recorded, so the other traffic keeps its 2024
  hours whatever the week's own ships do. It has no pilot or tug assignments; the pools (7 pilots, 8 tugs) are
  calibrated on it: the whole port's 2024 traffic exceeds them in 0.25% and 0.01% of hours.
