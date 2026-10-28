"""Colour parsing shared by the renderer, the scorer and the HTML profile.

Every colour in a document is stored as `#rrggbb` or `#rrggbbaa`. Inputs are more
forgiving, because they come from three places that each have their own habit:
the source designs store `rgba(255, 102, 51, 1)`, models write CSS names and short hex, and
the browser editor's colour picker emits `#rrggbb`.
"""

from __future__ import annotations

import math
import re

NAMED = {  # the full CSS named-colour table: models use these in HTML
    "aliceblue": "#f0f8ff", "antiquewhite": "#faebd7", "aqua": "#00ffff", "aquamarine": "#7fffd4",
    "azure": "#f0ffff", "beige": "#f5f5dc", "bisque": "#ffe4c4", "black": "#000000",
    "blanchedalmond": "#ffebcd", "blue": "#0000ff", "blueviolet": "#8a2be2", "brown": "#a52a2a",
    "burlywood": "#deb887", "cadetblue": "#5f9ea0", "chartreuse": "#7fff00", "chocolate": "#d2691e",
    "coral": "#ff7f50", "cornflowerblue": "#6495ed", "cornsilk": "#fff8dc", "crimson": "#dc143c",
    "cyan": "#00ffff", "darkblue": "#00008b", "darkcyan": "#008b8b", "darkgoldenrod": "#b8860b",
    "darkgray": "#a9a9a9", "darkgreen": "#006400", "darkgrey": "#a9a9a9", "darkkhaki": "#bdb76b",
    "darkmagenta": "#8b008b", "darkolivegreen": "#556b2f", "darkorange": "#ff8c00",
    "darkorchid": "#9932cc", "darkred": "#8b0000", "darksalmon": "#e9967a",
    "darkseagreen": "#8fbc8f", "darkslateblue": "#483d8b", "darkslategray": "#2f4f4f",
    "darkslategrey": "#2f4f4f", "darkturquoise": "#00ced1", "darkviolet": "#9400d3",
    "deeppink": "#ff1493", "deepskyblue": "#00bfff", "dimgray": "#696969", "dimgrey": "#696969",
    "dodgerblue": "#1e90ff", "firebrick": "#b22222", "floralwhite": "#fffaf0",
    "forestgreen": "#228b22", "fuchsia": "#ff00ff", "gainsboro": "#dcdcdc", "ghostwhite": "#f8f8ff",
    "gold": "#ffd700", "goldenrod": "#daa520", "gray": "#808080", "green": "#008000",
    "greenyellow": "#adff2f", "grey": "#808080", "honeydew": "#f0fff0", "hotpink": "#ff69b4",
    "indianred": "#cd5c5c", "indigo": "#4b0082", "ivory": "#fffff0", "khaki": "#f0e68c",
    "lavender": "#e6e6fa", "lavenderblush": "#fff0f5", "lawngreen": "#7cfc00",
    "lemonchiffon": "#fffacd", "lightblue": "#add8e6", "lightcoral": "#f08080",
    "lightcyan": "#e0ffff", "lightgoldenrodyellow": "#fafad2", "lightgray": "#d3d3d3",
    "lightgreen": "#90ee90", "lightgrey": "#d3d3d3", "lightpink": "#ffb6c1",
    "lightsalmon": "#ffa07a", "lightseagreen": "#20b2aa", "lightskyblue": "#87cefa",
    "lightslategray": "#778899", "lightslategrey": "#778899", "lightsteelblue": "#b0c4de",
    "lightyellow": "#ffffe0", "lime": "#00ff00", "limegreen": "#32cd32", "linen": "#faf0e6",
    "magenta": "#ff00ff", "maroon": "#800000", "mediumaquamarine": "#66cdaa",
    "mediumblue": "#0000cd", "mediumorchid": "#ba55d3", "mediumpurple": "#9370db",
    "mediumseagreen": "#3cb371", "mediumslateblue": "#7b68ee", "mediumspringgreen": "#00fa9a",
    "mediumturquoise": "#48d1cc", "mediumvioletred": "#c71585", "midnightblue": "#191970",
    "mintcream": "#f5fffa", "mistyrose": "#ffe4e1", "moccasin": "#ffe4b5", "navajowhite": "#ffdead",
    "navy": "#000080", "oldlace": "#fdf5e6", "olive": "#808000", "olivedrab": "#6b8e23",
    "orange": "#ffa500", "orangered": "#ff4500", "orchid": "#da70d6", "palegoldenrod": "#eee8aa",
    "palegreen": "#98fb98", "paleturquoise": "#afeeee", "palevioletred": "#db7093",
    "papayawhip": "#ffefd5", "peachpuff": "#ffdab9", "peru": "#cd853f", "pink": "#ffc0cb",
    "plum": "#dda0dd", "powderblue": "#b0e0e6", "purple": "#800080", "rebeccapurple": "#663399",
    "red": "#ff0000", "rosybrown": "#bc8f8f", "royalblue": "#4169e1", "saddlebrown": "#8b4513",
    "salmon": "#fa8072", "sandybrown": "#f4a460", "seagreen": "#2e8b57", "seashell": "#fff5ee",
    "sienna": "#a0522d", "silver": "#c0c0c0", "skyblue": "#87ceeb", "slateblue": "#6a5acd",
    "slategray": "#708090", "slategrey": "#708090", "snow": "#fffafa", "springgreen": "#00ff7f",
    "steelblue": "#4682b4", "tan": "#d2b48c", "teal": "#008080", "thistle": "#d8bfd8",
    "tomato": "#ff6347", "turquoise": "#40e0d0", "violet": "#ee82ee", "wheat": "#f5deb3",
    "white": "#ffffff", "whitesmoke": "#f5f5f5", "yellow": "#ffff00", "yellowgreen": "#9acd32",
    "transparent": "#00000000",
}

_RGB = re.compile(r"rgba?\(\s*([^)]+)\)", re.IGNORECASE)
_HSL = re.compile(r"hsla?\(\s*([^)]+)\)", re.IGNORECASE)


def parse_color(value: str | None, default: str = "#000000") -> str:
    """Normalise any supported colour spelling to `#rrggbb` or `#rrggbbaa`.

    Raises `ValueError` for something that is clearly not a colour, so that a
    tool call with a typo fails loudly instead of silently painting black.
    """
    if value is None or value == "":
        return default
    text = str(value).strip().lower()
    if text in NAMED:
        return NAMED[text]
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) in (3, 4):
            digits = "".join(c * 2 for c in digits)
        if len(digits) in (6, 8) and all(c in "0123456789abcdef" for c in digits):
            if len(digits) == 8 and digits.endswith("ff"):
                digits = digits[:6]
            return "#" + digits
        raise ValueError(f"not a hex colour: {value!r}")
    hsl = _HSL.fullmatch(text)
    if hsl:
        parts = [q.strip() for q in hsl.group(1).replace("/", ",").split(",")]
        if len(parts) not in (3, 4):
            raise ValueError(f"not an hsl() colour: {value!r}")
        h = float(parts[0].replace("deg", "")) % 360
        sat = float(parts[1].rstrip("%")) / 100
        light = float(parts[2].rstrip("%")) / 100
        c = (1 - abs(2 * light - 1)) * sat
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = light - c / 2
        r, g, b = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)][int(h // 60) % 6]
        rgb = ", ".join(str(round((v + m) * 255)) for v in (r, g, b))
        alpha = f", {parts[3]}" if len(parts) == 4 else ""
        return parse_color(f"rgba({rgb}{alpha})" if alpha else f"rgb({rgb})")
    match = _RGB.fullmatch(text)
    if match:
        parts = [p.strip() for p in match.group(1).replace("/", ",").split(",")]
        if len(parts) not in (3, 4):
            raise ValueError(f"not an rgb() colour: {value!r}")
        channels = []
        for part in parts[:3]:
            if part.endswith("%"):
                channels.append(round(float(part[:-1]) * 2.55))
            else:
                channels.append(round(float(part)))
        alpha = 1.0
        if len(parts) == 4:
            a = parts[3]
            alpha = float(a[:-1]) / 100 if a.endswith("%") else float(a)
        channels = [min(255, max(0, c)) for c in channels]
        alpha = min(1.0, max(0.0, alpha))
        out = "#" + "".join(f"{c:02x}" for c in channels)
        if alpha < 1.0:
            out += f"{round(alpha * 255):02x}"
        return out
    raise ValueError(f"unsupported colour: {value!r}")


def to_rgba(color: str) -> tuple[int, int, int, float]:
    """`#rrggbb[aa]` -> (r, g, b, alpha in 0..1)."""
    c = parse_color(color)
    r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    a = int(c[7:9], 16) / 255 if len(c) == 9 else 1.0
    return r, g, b, a


def to_css(color: str) -> str:
    r, g, b, a = to_rgba(color)
    if a >= 1.0:
        return f"#{r:02x}{g:02x}{b:02x}"
    return f"rgba({r}, {g}, {b}, {a:.3f})"


def to_skia(color: str, opacity: float = 1.0) -> int:
    """Pack into the ARGB int skia wants, folding element opacity into alpha."""
    r, g, b, a = to_rgba(color)
    alpha = round(max(0.0, min(1.0, a * opacity)) * 255)
    return (alpha << 24) | (r << 16) | (g << 8) | b


def _srgb_to_linear(c: float) -> float:
    c = c / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def to_lab(color: str) -> tuple[float, float, float]:
    r, g, b, _ = to_rgba(color)
    rl, gl, bl = (_srgb_to_linear(v) for v in (r, g, b))
    x = (0.4124 * rl + 0.3576 * gl + 0.1805 * bl) / 0.95047
    y = 0.2126 * rl + 0.7152 * gl + 0.0722 * bl
    z = (0.0193 * rl + 0.1192 * gl + 0.9505 * bl) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e(a: str, b: str) -> float:
    """CIE76 distance. Coarse, but the scorer only needs "same / close / different"."""
    la, lb = to_lab(a), to_lab(b)
    return math.dist(la, lb)


# A small named palette for writing descriptions a person would recognise.
_PALETTE = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "light gray": (211, 211, 211),
    "gray": (128, 128, 128),
    "dark gray": (64, 64, 64),
    "red": (220, 30, 40),
    "dark red": (128, 16, 24),
    "orange": (255, 140, 0),
    "yellow": (255, 221, 0),
    "gold": (212, 175, 55),
    "beige": (238, 222, 196),
    "brown": (120, 72, 36),
    "light green": (150, 220, 140),
    "green": (40, 160, 70),
    "dark green": (16, 80, 40),
    "teal": (0, 128, 128),
    "light blue": (150, 200, 240),
    "blue": (30, 90, 220),
    "navy": (16, 32, 96),
    "purple": (120, 50, 170),
    "lavender": (200, 180, 230),
    "pink": (245, 160, 190),
    "magenta": (200, 30, 140),
    "cream": (255, 248, 225),
    "burgundy": (110, 20, 50),
}


def color_name(color: str) -> str:
    r, g, b, _ = to_rgba(color)
    hexed = f"#{r:02x}{g:02x}{b:02x}"
    best = min(
        _PALETTE.items(),
        key=lambda kv: delta_e(hexed, "#" + "".join(f"{v:02x}" for v in kv[1])),
    )
    return best[0]
