# airportsdata

A Python package, pinned in `envs/airops/core/pyproject.toml` (`airportsdata>=20260905`); the packs were built with
version 20260905, whose `airports.csv` has sha256
`516c57d9d999f7a3be28ca649d2badbe3b972f07e57dc6173ab973b72d51cf52` (b6-v1's manifest, `airportsdata`; each b6-v2 date,
`task.sources.airportsdata`).

- **Source:** Mike Borsetti and contributors, https://github.com/mborsetti/airportsdata.
- **Licence:** MIT, as below. The packs and the test fixtures copy airportsdata's rows into each task's `airports`, so
  they carry this notice.
- **Extraction:** each IATA code's IANA time zone (the UTC conversion of every BTS time), the airports' names and
  coordinates, which the packs copy into each task's `airports`.
- **What it is and is not:** reference data; nothing in it describes traffic.

```
The MIT License (MIT)

Copyright (c) 2020- Mike Borsetti <mike@borsetti.com>

This project includes data from https://github.com/mwgg/Airports Copyright
(c) 2014 mwgg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
