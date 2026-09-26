# SPDX-License-Identifier: BSD-3-Clause

"""Extract coordinates from a model's free-text reply.

Models emit reasoning and coordinates together, in many shapes. The parser
accepts what they actually produce rather than demanding a schema, and returns
`None` when nothing usable is present so the failure lands in the reward
instead of raising.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# 48.8584, 2.2945   |   -16.49 / -68.12   |   lat: 12.9 lon: 77.5
_DECIMAL_PAIR = re.compile(
    r"(-?\d{1,3}(?:\.\d+)?)\s*(?:,|/|;|\s+and\s+|\s+)\s*(-?\d{1,3}(?:\.\d+)?)"
)

# 48°51'29"N 2°17'40"E
_DMS = re.compile(
    r"(\d{1,3})\s*[°d]\s*(\d{1,2})?\s*['′m]?\s*(\d{1,2}(?:\.\d+)?)?"
    r"\s*[\"″s]?\s*([NSEW])",
    re.IGNORECASE,
)

_LABELLED = re.compile(
    r"lat(?:itude)?\s*[:=]\s*(-?\d{1,3}(?:\.\d+)?)(?!\s*[°d])"
    r".{0,40}?"
    r"lon(?:g|gitude)?\s*[:=]\s*(-?\d{1,3}(?:\.\d+)?)(?!\s*[°d])",
    re.IGNORECASE | re.DOTALL,
)

_TAG = re.compile(r"<guess>(.*?)</guess>", re.IGNORECASE | re.DOTALL)

_JSON_ISH = re.compile(
    r"\"lat(?:itude)?\"\s*:\s*(-?\d{1,3}(?:\.\d+)?)"
    r".{0,60}?"
    r"\"lon(?:g|gitude)?\"\s*:\s*(-?\d{1,3}(?:\.\d+)?)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class ParsedGuess:
    """Outcome of parsing a reply.

    Attributes:
        lat (`float` or `None`):
            Latitude, or `None` when nothing could be extracted.
        lon (`float` or `None`):
            Longitude, or `None` when nothing could be extracted.
        source (`str`):
            Which pattern matched: `"tag"`, `"json"`, `"labelled"`, `"dms"`,
            `"decimal"` or `"none"`.
        note (`str`):
            Short explanation, safe to show the model as feedback.
    """

    lat: float | None
    lon: float | None
    source: str
    note: str = ""

    @property
    def ok(self) -> bool:
        """Whether a usable coordinate pair was extracted."""
        return self.lat is not None and self.lon is not None


def _valid(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def _dms_to_decimal(deg: str, minute: str | None, sec: str | None, hemi: str) -> float:
    value = float(deg) + float(minute or 0) / 60 + float(sec or 0) / 3600
    return -value if hemi.upper() in ("S", "W") else value


def _valid_dms(match: tuple[str, str | None, str | None, str]) -> bool:
    _, minute, sec, _ = match
    return float(minute or 0) < 60 and float(sec or 0) < 60


def parse_guess(response: str) -> ParsedGuess:
    """
    Pull a coordinate pair out of a model reply.

    Patterns are tried most explicit first, so a `<guess>` tag or a labelled
    `lat:`/`lon:` pair wins over a bare number pair that might be a date or a
    step count.

    Args:
        response (`str`):
            The model's unedited reply.

    Returns:
        [`ParsedGuess`]: The extracted coordinates, or a result whose `ok` is
        `False` with a `note` explaining what was wrong.

    Examples:

    ```python
    parse_guess("I think coastal Portugal. <guess>38.72, -9.14</guess>")
    ```
    """
    if not response or not response.strip():
        return ParsedGuess(None, None, "none", "Empty response.")

    tagged = _TAG.search(response)
    haystacks = [(tagged.group(1), "tag")] if tagged else []
    haystacks.append((response, "body"))

    for text, origin in haystacks:
        for pattern, name in ((_JSON_ISH, "json"), (_LABELLED, "labelled")):
            m = pattern.search(text)
            if m:
                lat, lon = float(m.group(1)), float(m.group(2))
                if _valid(lat, lon):
                    src = name if origin == "body" else "tag"
                    return ParsedGuess(lat, lon, src)
                return ParsedGuess(
                    None, None, "none", f"Coordinates out of range: {lat}, {lon}."
                )

        dms = _DMS.findall(text)
        if len(dms) >= 2:
            lat_m = next((d for d in dms if d[3].upper() in ("N", "S")), None)
            lon_m = next((d for d in dms if d[3].upper() in ("E", "W")), None)
            if lat_m and lon_m:
                if not _valid_dms(lat_m) or not _valid_dms(lon_m):
                    return ParsedGuess(
                        None,
                        None,
                        "none",
                        "Invalid DMS coordinates: minutes and seconds must be < 60.",
                    )
                lat = _dms_to_decimal(*lat_m)
                lon = _dms_to_decimal(*lon_m)
                if _valid(lat, lon):
                    return ParsedGuess(lat, lon, "dms")

        m = _DECIMAL_PAIR.search(text)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if _valid(lat, lon):
                src = "decimal" if origin == "body" else "tag"
                return ParsedGuess(lat, lon, src)
            return ParsedGuess(
                None, None, "none", f"Coordinates out of range: {lat}, {lon}."
            )

    return ParsedGuess(
        None,
        None,
        "none",
        "No coordinates found. Reply with a latitude and longitude, for "
        "example <guess>48.8584, 2.2945</guess>.",
    )
