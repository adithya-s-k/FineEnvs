"""The HTML surface: a restricted HTML/CSS profile that round-trips with the document.

A model on this surface writes a page instead of calling tools. The page is
parsed back into a `DesignDoc` and rendered by the same renderer as every other
surface, so its reward is directly comparable, structure term included. Nothing
here runs in a browser, so model-written markup is never executed.

The profile is what a person would write to position things on a fixed canvas:

    <div id="canvas" style="width:1080px;height:1080px;background:#f4efe6">
      <img src="asset://a3" style="left:0px;top:0px;width:1080px;height:720px">
      <div style="left:80px;top:760px;width:920px;font-family:'Playfair Display';
                  font-size:96px;color:#1d1d1f;text-align:center">Summer Sale</div>
      <div style="left:390px;top:900px;width:300px;height:80px;background:#e63946;
                  border-radius:40px"></div>
    </div>

Every child of the canvas is one element, back to front (or by `z-index`).
`img` is an asset, anything containing text is a text box, anything else is a
shape. `position:absolute` is implied. Nothing wraps; lines break on `<br>`, or
on newlines under `white-space:pre`, exactly as a browser would treat them.

Beyond the profile, the parser reads HTML the way models tend to write it and a
browser would render it: a `<style>` block with tag, class and id rules (by
specificity, inline styles winning), wrapper `<div>`s (flattened, their offsets
and text styles inherited), `right`/`bottom` anchoring, `margin` offsets,
`translate()` centering, `z-index`, `%`/`em`/`rem`/`pt`/`vw`/`vh` lengths, all
CSS colour names and `rgb()`/`hsl()`. Anything it cannot honour is reported
back as a warning, never silently dropped.
"""

from __future__ import annotations

import contextvars
import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from .colors import parse_color, to_css
from .doc import DesignDoc, Element

_VOID = {"img", "br", "hr", "meta", "link", "input", "source"}
_IGNORED_TAGS = {"script", "style", "head", "title", "meta", "link"}


class HTMLProfileError(ValueError):
    pass


# ------------------------------------------------------------ doc -> html


def _num(v: float) -> str:
    r = round(v, 2)
    return str(int(r)) if r == int(r) else str(r)


def _transform(el: Element) -> str:
    parts = []
    if el.angle:
        parts.append(f"rotate({_num(el.angle)}deg)")
    if el.flip_x or el.flip_y:
        parts.append(f"scale({-1 if el.flip_x else 1}, {-1 if el.flip_y else 1})")
    return f"transform:{' '.join(parts)};" if parts else ""


def element_style(el: Element) -> str:
    s = f"left:{_num(el.x)}px;top:{_num(el.y)}px;width:{_num(el.width)}px;"
    if el.kind != "text":
        s += f"height:{_num(el.height)}px;"
    s += _transform(el)
    if el.opacity < 1:
        s += f"opacity:{_num(el.opacity)};"
    if not el.visible:
        s += "display:none;"
    if el.kind == "shape":
        if el.fill:
            s += f"background:{to_css(el.fill)};"
        if el.stroke and el.stroke_width:
            s += f"border:{_num(el.stroke_width)}px solid {to_css(el.stroke)};"
        if el.shape == "ellipse":
            s += "border-radius:50%;"
        elif el.radius:
            s += f"border-radius:{_num(el.radius)}px;"
    if el.kind == "text":
        s += (
            "white-space:pre;"
            f"font-family:'{el.font_family}';font-size:{_num(el.font_size)}px;"
            f"line-height:{_num(el.line_height)};color:{to_css(el.color or '#000000')};"
            f"text-align:{el.align};"
        )
        if el.letter_spacing:
            s += f"letter-spacing:{_num(el.letter_spacing)}px;"
        if el.bold:
            s += "font-weight:700;"
        if el.italic:
            s += "font-style:italic;"
        if el.underline:
            s += "text-decoration:underline;"
        if el.uppercase:
            s += "text-transform:uppercase;"
    return s


def _text_body(el: Element) -> str:
    if not el.runs:
        return html.escape(el.text, quote=False)
    out = []
    for run in el.runs:
        style = ""
        if run.color:
            style += f"color:{to_css(run.color)};"
        if run.bold is not None:
            style += f"font-weight:{700 if run.bold else 400};"
        if run.italic is not None:
            style += f"font-style:{'italic' if run.italic else 'normal'};"
        body = html.escape(run.text, quote=False)
        out.append(f'<span style="{style}">{body}</span>' if style else body)
    return "".join(out)


def doc_to_html(doc: DesignDoc, asset_src=None) -> str:
    """Serialise a document in the profile. `asset_src(id)` picks the img src."""
    src = asset_src or (lambda aid: f"asset://{aid}")
    rows = [
        f'<div id="canvas" style="width:{doc.width}px;height:{doc.height}px;'
        f'background:{to_css(doc.background)}">'
    ]
    for el in doc.elements:
        style = element_style(el)
        if el.kind == "image":
            rows.append(f'  <img data-id="{el.id}" src="{src(el.asset_id)}" style="{style}">')
        elif el.kind == "shape":
            rows.append(f'  <div data-id="{el.id}" style="{style}"></div>')
        else:
            rows.append(f'  <div data-id="{el.id}" style="{style}">{_text_body(el)}</div>')
    rows.append("</div>")
    return "\n".join(rows)


# ------------------------------------------------------------ html -> doc


@dataclass
class _Node:
    tag: str
    attrs: dict
    children: list = field(default_factory=list)  # _Node | str


class _TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("#root", {})
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = _Node(tag.lower(), {k.lower(): (v or "") for k, v in attrs})
        self.stack[-1].children.append(node)
        if node.tag not in _VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(_Node(tag.lower(), {k.lower(): (v or "") for k, v in attrs}))

    def handle_endtag(self, tag):
        tag = tag.lower()
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def parse_style(text: str) -> dict[str, str]:
    out = {}
    for decl in (text or "").split(";"):
        if ":" not in decl:
            continue
        key, value = decl.split(":", 1)
        value = value.strip()
        if value.lower().endswith("!important"):
            value = value[: -len("!important")].strip()
        out[key.strip().lower()] = value
    return out


_LEN = re.compile(r"^(-?[\d.]+)\s*(px|em|rem|pt|%|vw|vh)?$")
# The canvas size for vw/vh, per parse. A ContextVar, since sessions parse concurrently.
_CANVAS: contextvars.ContextVar[tuple[float, float]] = contextvars.ContextVar(
    "canvas", default=(1000.0, 1000.0))


def _length(value: str, em: float = 16.0, pct_of: float | None = None) -> float:
    m = _LEN.match(str(value).strip().lower())
    if not m:
        raise HTMLProfileError(f"unsupported length {value!r} (use px)")
    num, unit = float(m.group(1)), m.group(2)
    if unit == "em":
        return num * em
    if unit == "rem":
        return num * 16.0
    if unit == "pt":
        return num * 4 / 3
    if unit == "vw":
        return num / 100 * _CANVAS.get()[0]
    if unit == "vh":
        return num / 100 * _CANVAS.get()[1]
    if unit == "%":
        if pct_of is None:
            raise HTMLProfileError(f"percent not allowed here: {value!r}")
        return num / 100 * pct_of
    return num


def _find_canvas(node: _Node) -> _Node | None:
    for child in node.children:
        if isinstance(child, _Node):
            if child.attrs.get("id") == "canvas":
                return child
            found = _find_canvas(child)
            if found:
                return found
    return None


def _first_element(node: _Node) -> _Node | None:
    for child in node.children:
        if isinstance(child, _Node) and child.tag not in _IGNORED_TAGS:
            if child.tag in ("html", "body"):
                return _first_element(child)
            return child
    return None


# ------------------------------------------------------------ stylesheets


@dataclass
class _Rule:
    spec: tuple
    order: int
    tag: str | None
    classes: list
    ids: list
    decls: dict


_UNSUPPORTED_SELECTOR = re.compile(r"[:\[\*~+]")
_BOX_SIDES = ("top", "right", "bottom", "left")


def _style_text(node: _Node) -> str:
    out = []
    for child in node.children:
        if isinstance(child, _Node):
            if child.tag == "style":
                out.extend(c for c in child.children if isinstance(c, str))
            else:
                out.append(_style_text(child))
    return "\n".join(out)


def _stylesheet(root: _Node, warnings: list) -> list[_Rule]:
    css = re.sub(r"/\*.*?\*/", "", _style_text(root), flags=re.S)
    rules, order = [], 0
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        decls = parse_style(m.group(2))
        for sel in m.group(1).split(","):
            sel = sel.strip()
            if not sel or sel.startswith("@"):
                continue
            if _UNSUPPORTED_SELECTOR.search(sel):
                warnings.append(f"CSS selector {sel!r} is not supported; that rule was ignored")
                continue
            last = re.split(r"[\s>]+", sel)[-1]
            tag = re.match(r"^([a-zA-Z][\w-]*)", last)
            classes = re.findall(r"\.([\w-]+)", last)
            ids = re.findall(r"#([\w-]+)", last)
            if not (tag or classes or ids):
                continue
            rules.append(_Rule((len(ids), len(classes), 1 if tag else 0), order,
                               tag.group(1).lower() if tag else None, classes, ids, decls))
            order += 1
    rules.sort(key=lambda r: (r.spec, r.order))
    return rules


def _matches(node: _Node, rule: _Rule) -> bool:
    if rule.tag and rule.tag != node.tag:
        return False
    have = set((node.attrs.get("class") or "").split())
    if any(c not in have for c in rule.classes):
        return False
    return all(node.attrs.get("id") == i for i in rule.ids)


def _expand(style: dict) -> dict:
    """margin/padding shorthands -> the four sides."""
    for prop in ("margin", "padding"):
        if prop in style:
            parts = style.pop(prop).split()
            if parts:
                t, r, b, left = {1: parts * 4, 2: parts * 2, 3: parts + [parts[1]],
                                 4: parts}.get(len(parts), parts[:4])
                for side, value in zip(_BOX_SIDES, (t, r, b, left), strict=True):
                    style.setdefault(f"{prop}-{side}", value)
    return style


def _own_style(node: _Node, rules: list[_Rule]) -> dict:
    merged: dict = {}
    for rule in rules:
        if _matches(node, rule):
            merged.update(_expand(dict(rule.decls)))
    merged.update(_expand(parse_style(node.attrs.get("style", ""))))
    return merged


_INHERITED = {"font-family", "font-size", "color", "text-align", "line-height", "letter-spacing",
              "font-weight", "font-style", "text-transform", "white-space", "text-decoration"}
_INLINE_TAGS = {"span", "b", "strong", "i", "em", "u", "br", "small", "sup", "sub", "a", "mark", "font"}
_TAG_FONT = {"h1": 2.0, "h2": 1.5, "h3": 1.17, "h4": 1.0, "h5": 0.83, "h6": 0.67}
_PRE = ("pre", "pre-wrap", "pre-line", "break-spaces")


def _is_wrapper(node: _Node, own: dict) -> bool:
    """A container of other elements (no text of its own, nothing painted)."""
    if node.tag == "img":
        return False
    if any(isinstance(c, str) and c.strip() for c in node.children):
        return False
    blocks = [c for c in node.children if isinstance(c, _Node)
              and c.tag not in _INLINE_TAGS and c.tag not in _IGNORED_TAGS]
    painted = any(k in own for k in ("background", "background-color", "border"))
    return bool(blocks) and not painted


# ------------------------------------------------------------ text


def _bold(value: str) -> bool:
    value = str(value).strip().lower()
    return value in ("bold", "bolder") or (value.isdigit() and int(value) >= 600)


def _text_runs(node: _Node, inherited: dict, warnings: list, rules=(), pre=False) -> list[dict]:
    runs: list[dict] = []
    for child in node.children:
        if isinstance(child, str):
            text = child if pre else re.sub(r"\s+", " ", child)
            if text:
                runs.append({"text": text, **inherited})
        elif child.tag == "br":
            runs.append({"text": "\n", **inherited})
        elif child.tag in _IGNORED_TAGS:
            continue
        else:
            style = _own_style(child, list(rules))
            inner = dict(inherited)
            if "color" in style:
                inner["color"] = parse_color(style["color"])
            if "font-weight" in style:
                inner["bold"] = _bold(style["font-weight"])
            if child.tag in ("b", "strong"):
                inner["bold"] = True
            if "font-style" in style:
                inner["italic"] = style["font-style"] == "italic"
            if child.tag in ("i", "em"):
                inner["italic"] = True
            extra = set(style) - {"color", "font-weight", "font-style"}
            if extra:
                warnings.append(f"<{child.tag}> inside text: {', '.join(sorted(extra))} ignored "
                                f"(only color, font-weight and font-style can vary within a text box)")
            block = child.tag in ("p", "div", "li") or child.tag in _TAG_FONT
            if block and runs and not runs[-1]["text"].endswith("\n"):
                runs.append({"text": "\n", **inherited})
            runs.extend(_text_runs(child, inner, warnings, rules, pre))
            if block and runs and not runs[-1]["text"].endswith("\n"):
                runs.append({"text": "\n", **inherited})
    return runs


def _clean_text(runs: list[dict], pre: bool = True) -> list[dict]:
    """Trim whitespace the way CSS does.

    Under `white-space:pre` only whitespace that contains a line break is
    indentation, so a bare leading or trailing space survives (design copy has
    plenty) while a pretty-printed body drops its indent. Under the default
    `white-space:normal`, whitespace is already collapsed and the ends of each
    line are trimmed.
    """
    if not runs:
        return runs
    text = "".join(r["text"] for r in runs)
    if pre:
        m_lead = re.match(r"^[ \t]*\n[ \t\n]*", text)
        m_trail = re.search(r"[ \t\n]*\n[ \t]*$", text)
        lead = m_lead.end() if m_lead else 0
        trail = len(text) - m_trail.start() if m_trail and m_trail.start() >= lead else 0
    else:
        lead = len(text) - len(text.lstrip(" \n"))
        trail = len(text) - len(text.rstrip(" \n"))
        m_lead = None
    out, pos = [], 0
    for r in runs:
        start, end = pos, pos + len(r["text"])
        pos = end
        s = max(start, lead) - start
        e = min(end, len(text) - trail) - start
        if e > s:
            out.append({**r, "text": r["text"][s:e]})
    if pre and m_lead is not None:
        for r in out:
            r["text"] = re.sub(r"\n[ \t]+", "\n", r["text"])
    if not pre:
        for r in out:
            r["text"] = re.sub(r" *\n *", "\n", r["text"])
    return [r for r in out if r["text"]]


# ------------------------------------------------------------ geometry


_HANDLED = {
    "position", "left", "top", "right", "bottom", "width", "height", "transform", "opacity",
    "display", "visibility", "background", "background-color", "border", "border-radius",
    "font-family", "font-size", "line-height", "color", "text-align", "letter-spacing",
    "font-weight", "font-style", "text-decoration", "text-transform", "white-space",
    "margin-top", "margin-left", "margin-right", "margin-bottom", "padding-top", "padding-left",
    "padding-right", "padding-bottom", "z-index", "box-sizing", "overflow", "object-fit",
    "font-variant", "text-rendering", "-webkit-font-smoothing",
}


def _transform_props(value: str, warnings: list) -> dict:
    """rotate/flip into element fields; translate into a pixel shift and an anchor."""
    out: dict = {"_dx": 0.0, "_dy": 0.0, "_ax": 0.0, "_ay": 0.0}
    for fn, arg in re.findall(r"([a-zA-Z]+)\(([^)]*)\)", value or ""):
        fn = fn.lower()
        args = [a.strip() for a in arg.split(",")]
        if fn == "rotate":
            a = args[0].lower()
            out["angle"] = float(a.replace("turn", "")) * 360 if a.endswith("turn") else float(
                a.replace("deg", "") or 0)
        elif fn in ("translate", "translatex", "translatey"):
            axes = {"translate": ("x", "y"), "translatex": ("x",), "translatey": ("y",)}[fn]
            for axis, a in zip(axes, args, strict=False):
                if a.endswith("%"):
                    out["_ax" if axis == "x" else "_ay"] -= float(a[:-1]) / 100
                else:
                    out["_dx" if axis == "x" else "_dy"] += _length(a)
        elif fn == "scalex" and args[0].startswith("-"):
            out["flip_x"] = True
        elif fn == "scaley" and args[0].startswith("-"):
            out["flip_y"] = True
        elif fn == "scale":
            if args and args[0].startswith("-"):
                out["flip_x"] = True
            if len(args) > 1 and args[1].startswith("-"):
                out["flip_y"] = True
        else:
            warnings.append(f"transform {fn}() is not supported; only rotate, translate and "
                            f"scale(-1) flips")
    return out


def _asset_from_src(src: str, attrs: dict) -> str | None:
    if attrs.get("data-asset"):
        return attrs["data-asset"]
    m = re.search(r"(?:asset://|/)(a\d+)(?:\.[a-z]+)?(?:[?#].*)?$", src or "")
    if m:
        return m.group(1)
    m = re.fullmatch(r"(a\d+)(?:\.[a-z]+)?", (src or "").strip())
    return m.group(1) if m else None


def _first_color(value: str, warnings: list, what: str) -> str | None:
    value = value.strip()
    if value in ("none", "transparent", ""):
        return None
    if "gradient(" in value:
        found = re.search(r"(#[0-9a-fA-F]{3,8}|rgba?\([^)]*\)|hsla?\([^)]*\)|\b[a-zA-Z]+\b(?=\s*[\d,)]))",
                          value.split("(", 1)[1])
        warnings.append(f"{what}: gradients are not supported; used its first colour")
        return parse_color(found.group(1)) if found else None
    token = re.match(r"(#[0-9a-fA-F]{3,8}|rgba?\([^)]*\)|hsla?\([^)]*\)|[a-zA-Z]+)", value)
    if not token:
        return None
    try:
        return parse_color(token.group(1))
    except ValueError:
        warnings.append(f"{what}: {value!r} not understood")
        return None


def html_to_doc(markup: str, width: int, height: int) -> tuple[dict, list[str]]:
    """Parse model-written HTML into a document dict (validated later by the op)."""
    if not markup or not markup.strip():
        raise HTMLProfileError("empty HTML")
    if len(markup) > 400_000:
        raise HTMLProfileError("HTML is too large (400 KB limit)")
    _CANVAS.set((float(width), float(height)))
    builder = _TreeBuilder()
    builder.feed(markup)
    builder.close()
    warnings: list[str] = []
    rules = _stylesheet(builder.root, warnings)

    canvas = _find_canvas(builder.root)
    if canvas is None:
        # Accept an un-id'd wrapper only if it is plainly a canvas: a div with a
        # size, holding elements. Anything else (a lone <p>, a fragment) would
        # silently replace the whole design with nothing.
        first = _first_element(builder.root)
        st = _own_style(first, rules) if first is not None else {}
        if first is not None and first.tag == "div" and "width" in st and "height" in st:
            canvas = first
    if canvas is None:
        raise HTMLProfileError('no canvas element; wrap the design in <div id="canvas" '
                               'style="width:...px;height:...px">')
    root_style = _own_style(canvas, rules)
    _layout_warning(canvas, root_style, warnings)
    background = "#ffffff"
    bg = root_style.get("background") or root_style.get("background-color")
    if bg:
        background = _first_color(bg, warnings, "canvas background") or "#ffffff"
    for dim, size in (("width", width), ("height", height)):
        if dim in root_style:
            try:
                if round(_length(root_style[dim])) != size:
                    warnings.append(f"canvas {dim} is fixed at {size}px")
            except HTMLProfileError:
                pass

    items: list[tuple] = []  # (node, style, own, ox, oy, box_w, box_h, order)

    def flatten(node: _Node, ox: float, oy: float, inherited: dict, box_w: float, box_h: float):
        for child in node.children:
            if isinstance(child, str):
                if child.strip() and not child.strip().startswith("```"):
                    warnings.append(f"loose text outside an element ignored: {child.strip()[:30]!r}")
                continue
            if child.tag in _IGNORED_TAGS or child.tag == "br":
                continue
            own = _own_style(child, rules)
            style = {**inherited, **own}
            if _is_wrapper(child, own):
                _layout_warning(child, own, warnings)
                fs = _font_px(style, child)
                wx = _px(own, "left", fs, box_w) + _px(own, "margin-left", fs, box_w)
                wy = _px(own, "top", fs, box_h) + _px(own, "margin-top", fs, box_h)
                w2 = _length(own["width"], fs, box_w) if "width" in own else box_w
                h2 = _length(own["height"], fs, box_h) if "height" in own else box_h
                keep = {k: v for k, v in style.items() if k in _INHERITED}
                flatten(child, ox + wx, oy + wy, keep, w2, h2)
                continue
            items.append((child, style, own, ox, oy, box_w, box_h, len(items)))

    flatten(canvas, 0.0, 0.0, {k: v for k, v in root_style.items() if k in _INHERITED},
            float(width), float(height))

    def z_of(item):
        z = item[2].get("z-index", "0")
        try:
            return int(float(z))
        except ValueError:
            return 0

    items.sort(key=lambda it: (z_of(it), it[7]))

    elements = []
    for n, (child, style, own, ox, oy, box_w, box_h, _) in enumerate(items, 1):
        unknown = sorted(set(own) - _HANDLED)
        if unknown:
            warnings.append(f"<{child.tag}> properties ignored: {', '.join(unknown)}")
        if own.get("object-fit", "fill") not in ("fill", ""):
            warnings.append(f"object-fit:{own['object-fit']} is not supported; the image is "
                            f"stretched to its box")
        if not any(own.get(k, "auto") != "auto" for k in ("left", "top", "right", "bottom",
                                                          "margin-left", "margin-top")):
            warnings.append(f"<{child.tag}> has no left/top; it is placed at the top-left of its "
                            f"container (every element is absolutely positioned)")
        eid = child.attrs.get("data-id") or f"h{n}"
        try:
            el = _element(child, style, own, ox, oy, box_w, box_h, warnings, rules)
        except (HTMLProfileError, ValueError) as exc:
            raise HTMLProfileError(f"element {n} (<{child.tag}>): {exc}") from None
        if el is None:
            continue
        el["id"] = eid
        elements.append(el)

    # Keep editor ids stable when they are well-formed, renumber the rest.
    seen = set()
    for i, el in enumerate(elements, 1):
        if not re.fullmatch(r"e\d+", el["id"]) or el["id"] in seen:
            el["id"] = f"h{i}"
        seen.add(el["id"])
    taken = {el["id"] for el in elements if el["id"].startswith("e")}
    nxt = max([int(i[1:]) for i in taken] + [0]) + 1
    for el in elements:
        if el["id"].startswith("h"):
            el["id"] = f"e{nxt}"
            nxt += 1
    return {"background": background, "elements": elements, "next_id": nxt}, sorted(set(warnings))


_FLOW_LAYOUT = ("flex", "inline-flex", "grid", "inline-grid", "table")


def _layout_warning(node: _Node, style: dict, warnings: list) -> None:
    """Containers are not laid out: say so when the page relied on it."""
    display = style.get("display", "")
    used = [k for k in ("justify-content", "align-items", "gap", "flex-direction", "grid-template-columns",
                        "place-items", "padding-left", "padding-top") if k in style]
    if display in _FLOW_LAYOUT or used:
        what = f"display:{display}" if display in _FLOW_LAYOUT else ", ".join(used)
        warnings.append(f"<{node.tag}> container layout ({what}) is not applied; position each "
                        f"element with left/top instead")


def _font_px(style: dict, node: _Node) -> float:
    base = 16.0 * _TAG_FONT.get(node.tag, 1.0)
    return _length(style["font-size"], 16.0, 16.0) if "font-size" in style else base


def _px(style: dict, key: str, em: float, pct_of: float) -> float:
    value = style.get(key)
    if value in (None, "", "auto"):
        return 0.0
    return _length(value, em, pct_of)


def _element(child, style, own, ox, oy, box_w, box_h, warnings, rules) -> dict | None:
    font_size = _font_px(style, child)
    visible = own.get("display") != "none" and own.get("visibility") != "hidden"
    geom: dict = {"visible": visible}
    if "width" in own and own["width"] != "auto":
        geom["width"] = _length(own["width"], font_size, box_w)
    if "height" in own and own["height"] != "auto":
        geom["height"] = _length(own["height"], font_size, box_h)
    if "opacity" in own:
        geom["opacity"] = max(0.0, min(1.0, float(own["opacity"])))
    tf = _transform_props(own.get("transform", ""), warnings)
    ax, ay = tf.pop("_ax"), tf.pop("_ay")
    dx, dy = tf.pop("_dx"), tf.pop("_dy")
    geom.update(tf)

    # Horizontal: left wins; otherwise right anchors the element's right edge.
    ml, mt = _px(own, "margin-left", font_size, box_w), _px(own, "margin-top", font_size, box_h)
    if own.get("left", "auto") != "auto":
        x = ox + _length(own["left"], font_size, box_w) + ml
    elif own.get("right", "auto") != "auto":
        x = ox + box_w - _length(own["right"], font_size, box_w) - _px(own, "margin-right", font_size, box_w)
        ax += 1.0
    else:
        x = ox + ml
    if own.get("top", "auto") != "auto":
        y = oy + _length(own["top"], font_size, box_h) + mt
    elif own.get("bottom", "auto") != "auto":
        y = oy + box_h - _length(own["bottom"], font_size, box_h) - _px(own, "margin-bottom", font_size, box_h)
        ay += 1.0
    else:
        y = oy + mt
    geom["x"], geom["y"] = x + dx, y + dy
    anchor = [round(ax, 6), round(ay, 6)] if (ax or ay) else None

    if child.tag == "img":
        asset = _asset_from_src(child.attrs.get("src", ""), child.attrs)
        if not asset:
            raise HTMLProfileError(f"img src must be asset://<id>, got {child.attrs.get('src')!r}")
        if "width" not in geom or "height" not in geom:
            raise HTMLProfileError("img needs width and height")
        out = {"kind": "image", "asset_id": asset, **geom}
        if anchor:
            out["_anchor"] = anchor
        return out

    pre = style.get("white-space", "normal") in _PRE
    base = {}
    if "color" in style:
        base["color"] = parse_color(style["color"])
    base["bold"] = _bold(style["font-weight"]) if "font-weight" in style else child.tag in _TAG_FONT
    base["italic"] = style.get("font-style", "") == "italic"
    runs = _clean_text(_text_runs(child, {}, warnings, rules, pre), pre)
    text = "".join(r["text"] for r in runs)
    if text.strip():
        family = style.get("font-family", "").split(",")[0].strip().strip("'\"") or None
        lh_raw = str(style.get("line-height", "normal")).strip()
        if lh_raw == "normal":
            line_height = 1.2
        elif lh_raw.endswith("%"):
            line_height = float(lh_raw[:-1]) / 100
        elif re.search(r"[a-z]", lh_raw):
            line_height = _length(lh_raw, font_size) / font_size
        else:
            line_height = float(lh_raw)
        ls_raw = str(style.get("letter-spacing", "0")).strip()
        letter_spacing = 0.0 if ls_raw == "normal" else _length(ls_raw, font_size)
        el_runs = None
        if len({(r.get("color"), r.get("bold"), r.get("italic")) for r in runs}) > 1:
            el_runs = [{k: v for k, v in r.items() if v is not None} for r in runs]
        align = style.get("text-align", "left")
        if align in ("start", "justify"):
            align = "left"
        elif align == "end":
            align = "right"
        if align not in ("left", "center", "right"):
            warnings.append(f"text-align {align!r} not supported; using left")
            align = "left"
        pad_l, pad_t = _px(own, "padding-left", font_size, box_w), _px(own, "padding-top", font_size, box_h)
        if "width" not in geom and align != "left" and child.tag in ("div", "p", *_TAG_FONT) \
                and own.get("left", "auto") != "auto" and box_w < _CANVAS.get()[0] - 0.5:
            # A block inside a sized wrapper spans it, which is what gives
            # centred or right-aligned copy something to align within.
            geom["width"] = box_w - (geom["x"] - ox)
        out = {
            "kind": "text", "text": text, "font_family": family, "font_size": font_size,
            "color": base.get("color", "#000000"), "bold": base["bold"], "italic": base["italic"],
            "underline": "underline" in style.get("text-decoration", ""), "align": align,
            "line_height": line_height, "letter_spacing": letter_spacing,
            "uppercase": style.get("text-transform", "") == "uppercase", "runs": el_runs,
            "auto_width": "width" not in geom,
            **{k: v for k, v in geom.items() if k != "height"},
            "width": geom.get("width", 10.0),
        }
        out["x"] += pad_l
        out["y"] += pad_t
        if anchor:
            out["_anchor"] = anchor
        return out

    fill = own.get("background") or own.get("background-color")
    stroke, stroke_width = None, 0.0
    if "border" in own:
        for part in own["border"].split():
            if _LEN.match(part):
                stroke_width = _length(part)
            elif part not in ("solid", "dashed", "dotted", "none"):
                try:
                    stroke = parse_color(part)
                except ValueError:
                    pass
    if "width" not in geom or "height" not in geom:
        if not fill and not stroke:
            return None  # an empty, unsized div paints nothing
        raise HTMLProfileError("a shape needs width and height")
    radius_raw = own.get("border-radius", "0").split()[0]
    if radius_raw in ("50%", "100%"):
        shape, radius = "ellipse", 0.0
    else:
        shape = "rect"
        radius = _length(radius_raw, font_size, min(geom["width"], geom["height"]))
    out = {"kind": "shape", "shape": shape,
           "fill": _first_color(fill, warnings, "background") if fill else None,
           "stroke": stroke, "stroke_width": stroke_width, "radius": radius, **geom}
    if anchor:
        out["_anchor"] = anchor
    return out
