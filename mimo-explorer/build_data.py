# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow", "huggingface_hub>=0.30", "langdetect"]
# ///
"""Turn XiaomiMiMo/MiMo-V2.6-RL-oss into the static data the explorer loads.

    uv run build_data.py                  # downloads what it needs from the Hub
    RAW_DIR=raw uv run build_data.py      # or reuse a local copy of the same files

Writes site/data/, gzipped (Spaces serve static files uncompressed, so the page inflates them itself):
    index.json.gz      one small record per environment: what search, filters and the map need
    <domain>.json.gz   full briefs and domain-specific detail, fetched when an environment is opened

Rubric *questions* are kept; `pass_anchor` and `gold_answer` are not. They are the answers,
and a browsing tool is not the place to hand them out.
"""

from __future__ import annotations

import collections
import gzip
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

REPO = "XiaomiMiMo/MiMo-V2.6-RL-oss"
OUT = Path(__file__).parent / "web" / "data"
PARQUETS = {
    "code": "code.parquet",
    "webdev": "webdev.parquet",
    "cyber": "cyber.parquet",
    "music": "music.parquet",
    "general": "general/train.parquet",
}


# ── fetch ────────────────────────────────────────────────────────────────────
def fetch() -> tuple[Path, list[str]]:
    from huggingface_hub import HfApi, snapshot_download

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    raw = os.environ.get("RAW_DIR")
    if raw:
        return Path(raw), files
    path = snapshot_download(
        REPO,
        repo_type="dataset",
        allow_patterns=list(PARQUETS.values()) + ["general/envs/*/verifier_meta.json"],
    )
    return Path(path), files


# ── small helpers ────────────────────────────────────────────────────────────
def clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= n:
        return text
    cut = text[: n - 1]
    if " " in cut[n // 2:]:            # end on a whole word when there is one to end on
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:-") + "…"


def plain(text: str) -> str:
    """Markdown down to prose, for one-line snippets: no headings, emphasis, fences or links."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(\*\*|__|\*|`)", "", text)
    return re.sub(r"^\s*[-*•]\s+", "", text, flags=re.M)


def snippet(text: str, title: str, n: int = 280) -> str:
    """The brief after its headline, so a card does not say the same sentence twice."""
    body = re.sub(r"\s+", " ", plain(text)).strip()
    stem = title.rstrip("…").strip()
    at = body.find(stem) if stem else -1
    if 0 <= at < 60:                       # the headline, possibly after a greeting
        end = at + len(stem)
        if title.endswith("…"):            # it was cut: carry on from the end of that sentence
            m = re.search(r"[.!?。！？](\s|$)", body[end:])
            end = end + m.end() if m else len(body)
        rest = body[end:].lstrip(" .,;:!?。，；：！？")
        if len(rest) > 30:
            body = rest
    return clip(body, n)


def first_sentence(text: str, n: int = 110) -> str:
    text = re.sub(r"^\s*(prompt|task)\s*:\s*", "", text.strip(), flags=re.I)
    lines = [l.strip().lstrip("#").strip().strip("*").strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    # skip greetings and stubs ("Hi,", "Hello team") in favour of the first line that says something
    line = next((l for l in lines if len(l) >= 18 and not re.match(r"(?i)^(hi|hello|hey|dear|greetings)\b", l)),
                lines[0] if lines else text)
    # end of the first sentence, not the dot in "U.S." / "Inc." / "e.g."
    end = None
    for m in re.finditer(r"[.!?。！？](?=\s|$)", line):
        before = line[:m.start()].rsplit(None, 1)[-1] if line[:m.start()].strip() else ""
        if m.group() == "." and (re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]?", before) or before.lower() in ABBREV or len(before) == 1):
            continue
        if m.end() > 25:
            end = m.end()
            break
    return clip(line[:end] if end else line, n)


ABBREV = {"inc", "corp", "co", "ltd", "llc", "plc", "no", "mr", "ms", "mrs", "dr", "st", "vs", "etc", "jr", "sr", "fig", "approx"}


def lang_of(text: str) -> str:
    """A coarse language label. CJK is decided by script, the rest by langdetect."""
    if re.search(r"[一-鿿]", text):
        return "zh"
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    try:
        return detect(text[:600])
    except Exception:
        return "unknown"


LANG_NAMES = {
    "en": "English", "zh": "Chinese", "es": "Spanish", "tr": "Turkish", "ar": "Arabic",
    "fr": "French", "de": "German", "pt": "Portuguese", "ru": "Russian", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "id": "Indonesian", "vi": "Vietnamese", "nl": "Dutch",
    "pl": "Polish", "hi": "Hindi", "fa": "Persian", "uk": "Ukrainian", "th": "Thai",
}


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code, "Other")


def instance(row) -> dict:
    return json.loads(row["extra_info"]["instance_json"])


def brief(row) -> str:
    return row["prompt"][-1]["content"]


# ── code ─────────────────────────────────────────────────────────────────────
EXT_LANG = {
    "py": "Python", "go": "Go", "js": "JavaScript", "jsx": "JavaScript", "mjs": "JavaScript",
    "cjs": "JavaScript", "ts": "TypeScript", "tsx": "TypeScript", "rb": "Ruby", "php": "PHP",
    "java": "Java", "rs": "Rust", "c": "C", "h": "C", "cpp": "C++", "cc": "C++", "hpp": "C++",
    "cs": "C#", "kt": "Kotlin", "scala": "Scala", "swift": "Swift", "dart": "Dart", "lua": "Lua",
    "ex": "Elixir", "exs": "Elixir", "hs": "Haskell", "vue": "Vue", "svelte": "Svelte",
}


def build_code(df):
    index, detail = [], {}
    for _, row in df.iterrows():
        j = instance(row)
        text = brief(row)
        # patches mix `+++ b/path` and bare `+++ path` headers; /dev/null is a deletion, not a file
        touched = [f for f in re.findall(r"^\+\+\+ (?:b/)?(\S+)", j.get("test_patch") or "", re.M) if f != "/dev/null"]
        langs = collections.Counter(
            EXT_LANG[f.rsplit(".", 1)[-1].lower()] for f in touched if f.rsplit(".", 1)[-1].lower() in EXT_LANG
        )
        language = langs.most_common(1)[0][0] if langs else "Unknown"
        eid = j["instance_id"]
        title = first_sentence(re.sub(r"^\[[A-Z]+\]\s*", "", text))
        index.append({
            "id": eid, "d": "code", "t": title,
            "s": snippet(re.sub(r"^\[[A-Z]+\]\s*", "", text), title), "f": {"prog": language},
        })
        detail[eid] = {
            "brief": text,
            "meta": [["Programming language", language], ["Test command", j.get("test_command", "")],
                     ["Working directory", j.get("cwd", "")], ["Verifier timeout", f"{j.get('verifier_timeout_sec', '?')}s"]],
            "files": {"Files the tests touch": touched[:40]},
        }
    return index, detail


# ── cyber ────────────────────────────────────────────────────────────────────
CRASH = re.compile(r"^(?P<san>[\w ]*?Sanitizer|[\w-]+):\s*(?P<kind>[\w-]+)(?: on [\w-]+)?\s+in\s+(?:function\s+)?`?(?P<fn>[^`\s]+)`?(?:.*?in file `(?P<file>[^`]+)`)?")


def build_cyber(df):
    index, detail = [], {}
    for _, row in df.iterrows():
        j = instance(row)
        text = brief(row)
        m = CRASH.match(text.strip())
        san = m.group("san") if m else "Unknown"
        kind = m.group("kind") if m else "unknown"
        fn = m.group("fn") if m else ""
        file = (m.group("file") if m else "") or ""
        project = file.split("/")[0] if "/" in file else "unknown"
        eid = j["instance_id"]
        index.append({
            "id": eid, "d": "cyber", "t": f"{kind} in {fn}" if fn else clip(text, 110),
            "s": clip(plain(text), 280),
            "f": {"crash": kind, "project": project, "sanitizer": san},
        })
        detail[eid] = {
            "brief": text,
            "meta": [["Crash", kind], ["Sanitizer", san], ["Project", project], ["Function", fn],
                     ["File", file], ["Source", "ARVO (reproduced OSS-Fuzz crashes)"]],
            "extra": clip(j.get("description", ""), 2000) if j.get("description") != text else "",
        }
    return index, detail


# ── webdev ───────────────────────────────────────────────────────────────────
# Specific kinds of site. "Landing page" is a format more than a kind, so it only wins when
# nothing more specific is named; otherwise the earliest specific mention is the primary type.
SITE_TYPES = [
    ("Restaurant & food", r"restaurant|\bcaf[eé]\b|bakery|bistro|coffee shop|food truck|\bdiner\b|pizzeria"),
    ("Portfolio", r"portfolio"), ("Dashboard", r"dashboard"),
    ("SaaS / product", r"\bsaas\b|product (?:page|showcase|site|launch)"),
    ("Blog & editorial", r"\bblog\b|magazine|newsletter|news site"),
    ("E-commerce", r"e-?commerce|online store|storefront|\bshop\b"),
    ("Agency", r"\bagency\b"), ("Game", r"\bgame\b"),
    ("Event", r"event (?:landing|site|page|website|microsite)|conference|festival|wedding"),
    ("Tool / app", r"\bweb app\b|calculator|\btracker\b|\bquiz\b"),
]
LANDING = r"landing page"
FRAMEWORKS = [
    ("React", r"\breact\b"), ("Vue", r"\bvue\b"), ("Next.js", r"next\.?js"), ("Astro", r"\bastro\b"),
    ("Svelte", r"svelte"), ("Three.js", r"three\.?js|webgl"), ("Tailwind", r"tailwind"),
    ("Plain HTML/CSS", r"plain html|vanilla (?:js|javascript)|html,? css"),
]
STYLES = [
    ("Minimalist", r"minimalis|\bminimal\b"), ("Dark mode", r"dark[- ]mode|dark theme"), ("Brutalist", r"brutalis"),
    ("Glassmorphism", r"glassmorph|glass cards|frosted glass"), ("Neumorphism", r"neumorph"), ("Swiss", r"\bswiss\b"),
    ("Scandinavian", r"scandinav|nordic"), ("Art deco", r"art[- ]deco"), ("Retro", r"\bretro\b|vintage|\by2k\b"),
    ("Cyberpunk", r"cyberpunk|\bneon\b"), ("Material", r"material design"), ("Flat", r"flat design"),
    ("Luxury", r"luxur"), ("Playful", r"playful|whimsical|cartoon"), ("Corporate", r"corporate"),
]


def primary_site(text: str) -> str:
    low = text.lower()
    hits = [(m.start(), name) for name, pat in SITE_TYPES if (m := re.search(pat, low))]
    if hits:
        return min(hits)[1]
    return "Landing page" if re.search(LANDING, low) else "Other"


def tags(text: str, vocab) -> list[str]:
    low = text.lower()
    return [name for name, pat in vocab if re.search(pat, low)]


def build_webdev(df):
    index, detail = [], {}
    for _, row in df.iterrows():
        j = instance(row)
        text = brief(row)
        site = primary_site(text)
        fws = tags(text, FRAMEWORKS) or ["Unspecified"]
        styles = tags(text, STYLES)
        language = lang_name(lang_of(text))
        eid = j["instance_id"]
        index.append({
            "id": eid, "d": "webdev", "t": first_sentence(text), "s": snippet(text, first_sentence(text)),
            "f": {"site": site, "framework": fws, "style": styles or ["None named"], "language": language},
        })
        detail[eid] = {
            "brief": text,
            "meta": [["Site type", site], ["Framework (mentioned)", ", ".join(fws)],
                     ["Style (mentioned)", ", ".join(styles) or "none named"], ["Brief language", language]],
        }
    return index, detail


# ── music ────────────────────────────────────────────────────────────────────
# The dataset's style tags are Chinese on every row, including English briefs.
MUSIC_STYLE = {
    "马林巴曲": ("Marimba piece", "Solo instrument"), "爱尔兰 reel": ("Irish reel", "Folk & world"),
    "谐谑曲": ("Scherzo", "Classical forms"), "八音盒音乐": ("Music box", "Solo instrument"),
    "小夜曲": ("Serenade", "Classical forms"), "马祖卡": ("Mazurka", "Dance"), "钢琴独奏": ("Piano solo", "Piano"),
    "弦乐合奏": ("String ensemble", "Ensemble"), "波罗乃兹": ("Polonaise", "Dance"),
    "管风琴前奏曲": ("Organ prelude", "Classical forms"), "木管五重奏": ("Woodwind quintet", "Ensemble"),
    "极简主义钢琴循环（Glass 风）": ("Minimalist piano loop (Glass)", "Piano"),
    "阿尔贝蒂低音钢琴小品": ("Alberti-bass piano piece", "Piano"), "简单钢琴小品": ("Simple piano piece", "Piano"),
    "长笛独奏": ("Flute solo", "Solo instrument"), "Stride 钢琴": ("Stride piano", "Jazz & blues"),
    "古典小步舞曲": ("Classical minuet", "Dance"), "巴赫风格四声部众赞歌": ("Bach-style chorale", "Classical forms"),
    "拉格泰姆钢琴": ("Ragtime piano", "Jazz & blues"), "莫扎特风格钢琴小品": ("Mozart-style piano piece", "Piano"),
    "手风琴曲": ("Accordion piece", "Solo instrument"), "肖邦风格圆舞曲": ("Chopin-style waltz", "Dance"),
    "回旋曲": ("Rondo", "Classical forms"), "铜管五重奏": ("Brass quintet", "Ensemble"),
    "弦乐四重奏（海顿风）": ("String quartet (Haydn)", "Ensemble"), "萨蒂风格裸体歌舞": ("Satie-style gymnopédie", "Piano"),
    "波尔卡": ("Polka", "Dance"), "英式乡村舞曲": ("English country dance", "Dance"),
    "木管三重奏": ("Woodwind trio", "Ensemble"), "新古典钢琴（Einaudi 风）": ("Neoclassical piano (Einaudi)", "Piano"),
    "卡农": ("Canon", "Classical forms"), "俄罗斯民歌": ("Russian folk song", "Folk & world"),
    "巴洛克二声部创意曲": ("Baroque two-part invention", "Classical forms"), "爱尔兰 jig": ("Irish jig", "Folk & world"),
    "竖琴曲": ("Harp piece", "Solo instrument"), "古典主题与变奏": ("Theme and variations", "Classical forms"),
    "古典小奏鸣曲": ("Classical sonatina", "Classical forms"), "桑巴": ("Samba", "Folk & world"),
    "摇摆爵士大乐队": ("Swing big band", "Jazz & blues"), "Afrobeats 现代": ("Afrobeats", "Pop & modern"),
    "巴尔干 7/8 舞曲": ("Balkan 7/8 dance", "Folk & world"), "灵歌": ("Spiritual", "Folk & world"),
    "苏格兰 hornpipe": ("Scottish hornpipe", "Folk & world"), "古筝独奏": ("Guzheng solo", "Chinese traditional"),
    "12 小节布鲁斯钢琴": ("12-bar blues piano", "Jazz & blues"), "管弦乐团全奏": ("Full orchestra", "Ensemble"),
    "Reggaeton dembow": ("Reggaeton dembow", "Pop & modern"), "口琴曲": ("Harmonica piece", "Solo instrument"),
    "比波普": ("Bebop", "Jazz & blues"), "浪漫派大提琴旋律": ("Romantic cello melody", "Solo instrument"),
    "J-Pop": ("J-Pop", "Pop & modern"), "浪漫派小提琴小品": ("Romantic violin piece", "Solo instrument"),
    "拉丁爵士": ("Latin jazz", "Jazz & blues"), "钢琴四手联弹": ("Piano four hands", "Piano"),
    "氛围钢琴": ("Ambient piano", "Piano"), "西部乡村": ("Country & western", "Folk & world"),
    "琵琶曲": ("Pipa piece", "Chinese traditional"), "萨克斯独奏": ("Saxophone solo", "Solo instrument"),
    "中国风流行": ("Chinese-style pop", "Pop & modern"), "江南丝竹": ("Jiangnan sizhu", "Chinese traditional"),
    "尤克里里曲": ("Ukulele piece", "Solo instrument"), "小号独奏": ("Trumpet solo", "Solo instrument"),
    "木吉他弹唱伴奏": ("Acoustic guitar accompaniment", "Pop & modern"), "摇滚乐队编制": ("Rock band", "Pop & modern"),
    "新灵魂乐": ("Neo-soul", "Pop & modern"), "维也纳圆舞曲": ("Viennese waltz", "Dance"),
    "肖邦风格夜曲": ("Chopin-style nocturne", "Piano"), "蓝草乐": ("Bluegrass", "Folk & world"),
    "爵士钢琴三重奏": ("Jazz piano trio", "Jazz & blues"), "克莱兹默旋律": ("Klezmer melody", "Folk & world"),
    "R&B 慢歌": ("R&B ballad", "Pop & modern"), "小提琴独奏": ("Violin solo", "Solo instrument"),
    "合成器浪潮（Synthwave）": ("Synthwave", "Pop & modern"), "德彪西风格印象派钢琴": ("Debussy-style impressionist piano", "Piano"),
    "吉他指弹": ("Fingerstyle guitar", "Solo instrument"), "爵士华尔兹": ("Jazz waltz", "Jazz & blues"),
    "大提琴独奏": ("Cello solo", "Solo instrument"), "City Pop（日式）": ("City pop", "Pop & modern"),
    "竹笛曲": ("Dizi (bamboo flute) piece", "Chinese traditional"),
}


def music_key(text: str) -> str:
    m = re.search(r"([升降]?)([A-G])([#♯b♭]?)(大调|小调)", text)
    if m:
        acc = {"升": "♯", "#": "♯", "♯": "♯", "降": "♭", "b": "♭", "♭": "♭"}.get(m.group(1) or m.group(3), "")
        return f"{m.group(2)}{acc} {'major' if m.group(4) == '大调' else 'minor'}"
    m = re.search(r"\b([A-G])(?:([#♯b♭])| (sharp|flat))? (major|minor)\b", text)
    if m:
        acc = "♯" if (m.group(2) in ("#", "♯") or m.group(3) == "sharp") else "♭" if (m.group(2) in ("b", "♭") or m.group(3) == "flat") else ""
        return f"{m.group(1)}{acc} {m.group(4)}"
    return ""


def tempo(bpm) -> str:
    try:
        b = float(bpm)
    except (TypeError, ValueError):
        return "Unknown"
    return "Slow (<80)" if b < 80 else "Moderate (80–120)" if b <= 120 else "Fast (>120)"


def build_music(df):
    index, detail = [], {}
    for n, (_, row) in enumerate(df.iterrows()):
        e = row["extra_info"]
        text = brief(row)
        eid = f"music-{e.get('src_id', n)}"
        voices = int(e.get("nvoice_want") or 0)
        tag = str(e.get("tag") or "")
        style, family = MUSIC_STYLE.get(tag, (tag or "Unknown", "Other"))
        key = music_key(text)
        index.append({
            "id": eid, "d": "music", "t": f"{style} in {key}" if key else style, "s": snippet(text, f"{style} in {key}" if key else style),
            "f": {"family": family, "style": style, "meter": str(e.get("meter") or "?"),
                  "voices": f"{voices} voice{'s' if voices != 1 else ''}", "tempo": tempo(e.get("bpm")),
                  "language": lang_name(e.get("lang") or lang_of(text))},
        })
        detail[eid] = {
            "brief": text,
            "meta": [["Style", f"{style} ({tag})" if tag and tag != style else style], ["Family", family],
                     ["Key", key or "not stated"], ["Tempo", f"{e.get('bpm')} BPM"], ["Meter", str(e.get("meter"))],
                     ["Voices", str(voices)], ["Length", f"{e.get('length')} bars"], ["Output", "ABC notation"]],
        }
    return index, detail


# ── general ──────────────────────────────────────────────────────────────────
INDUSTRY = {
    "accounting_audit_tax": "Accounting, audit & tax", "finance_insurance": "Finance & insurance",
    "healthcare_ops": "Healthcare operations", "consulting_bizops_analytics": "Consulting & analytics",
    "hr_people": "HR & people", "government_public": "Government & public sector",
    "it_information_systems": "IT & information systems", "education_research_admin": "Education & research admin",
    "energy_utilities_esg": "Energy, utilities & ESG", "legal_compliance": "Legal & compliance",
    "construction_pm": "Construction & project mgmt", "ecommerce_ops": "E-commerce operations",
    "manufacturing_quality": "Manufacturing & quality", "hospitality_fnb": "Hospitality & F&B",
    "last_mile_logistics": "Last-mile logistics", "agriculture_coop": "Agriculture co-ops",
}
ENV_ID = re.compile(r"^s3k_\d+_(?P<slug>.+?)_(?P<lang>en|zh)_(?P<tier>t\d)_rl_\d+$")
FILE_KIND = {
    "xlsx": "Spreadsheet", "xls": "Spreadsheet", "csv": "Spreadsheet", "docx": "Document", "doc": "Document",
    "pdf": "PDF", "pptx": "Slides", "png": "Image", "jpg": "Image", "jpeg": "Image", "svg": "Image",
    "html": "Web page", "xml": "Data", "json": "Data", "zip": "Archive", "md": "Document", "txt": "Document",
}


# display names live in app/names.py so the app and this build agree
sys.path.insert(0, str(Path(__file__).parent))
from app.names import pretty_system  # noqa: E402


def build_general(df, raw: Path, files: list[str]):
    on_disk = collections.defaultdict(list)
    for f in files:
        m = re.match(r"general/envs/([^/]+)/(.+)", f)
        if m:
            on_disk[m.group(1)].append(m.group(2))

    index, detail = [], {}
    for _, row in df.iterrows():
        j = instance(row)
        text = brief(row)
        eid = j["instance_id"]
        if j.get("dataset_type") == "terminal_bench":
            cat = j.get("category") or "Other"
            sub = j.get("subcategory") or ""
            index.append({
                "id": eid, "d": "general",
                "t": clip(j.get("display_description") or first_sentence(text), 120), "s": snippet(text, clip(j.get("display_description") or first_sentence(text), 120)),
                "f": {"industry": "Terminal tasks", "source": "Terminal-bench style",
                      "tier": "Unrated", "language": lang_name(lang_of(text))},
            })
            detail[eid] = {
                "brief": text,
                "meta": [["Category", f"{cat} / {sub}" if sub else cat], ["Tags", ", ".join(j.get("tags") or [])],
                         ["Resources", f"{j.get('cpus', '?')} CPU · {j.get('memory_mb', '?')} MB · internet {'on' if j.get('allow_internet') else 'off'}"],
                         ["Agent timeout", f"{j.get('agent_timeout_sec', '?')}s"]],
                "files": {"Test files": list(j.get("tests_files") or [])[:30]} if j.get("tests_files") else {},
            }
            continue

        m = ENV_ID.match(eid)
        slug, lang, tier = (m.group("slug"), m.group("lang"), m.group("tier")) if m else ("other", "en", "t?")
        fs = on_disk.get(eid, [])
        systems = sorted({pretty_system(Path(f).stem) for f in fs if f.startswith("tools/") and f.endswith(".py")})
        workspace = sorted(f[len("workspace/"):] for f in fs if f.startswith("workspace/"))
        by_kind = collections.defaultdict(list)
        for w in workspace:
            by_kind[FILE_KIND.get(w.rsplit(".", 1)[-1].lower(), "Other")].append(w)

        rubric = []
        meta_path = raw / "general" / "envs" / eid / "verifier_meta.json"
        if meta_path.exists():
            for it in json.loads(meta_path.read_text()).get("items", []):
                rubric.append({"tier": it.get("tier"), "method": it.get("method"),
                               "weight": it.get("weight"), "q": it.get("question", "")})

        industry = INDUSTRY.get(slug, slug.replace("_", " ").capitalize())
        index.append({
            "id": eid, "d": "general", "t": first_sentence(text, 120), "s": snippet(text, first_sentence(text, 120)),
            "f": {"industry": industry, "source": "Simulated workplace", "tier": f"Tier {tier[1:]}",
                  "language": lang_name(lang)},
            "n": {"systems": len(systems), "files": len(workspace), "checks": len(rubric)},
        })
        detail[eid] = {
            "brief": text,
            "meta": [["Industry", industry], ["Tier", f"Tier {tier[1:]} of 5"], ["Language", lang_name(lang)],
                     ["Grading", f"{len(rubric)} rubric checks"]],
            "systems": systems,
            "files": {k: v for k, v in sorted(by_kind.items(), key=lambda kv: -len(kv[1]))},
            "rubric": rubric,
            "tree": f"https://huggingface.co/datasets/{REPO}/tree/main/general/envs/{eid}",
        }
    return index, detail


# ── main ─────────────────────────────────────────────────────────────────────
DOMAINS = {
    "code": ("Code", "Fix real issues in real repos", "hidden unit tests", "prog"),
    "webdev": ("Webdev", "Build a website from a brief", "a vision judge", "site"),
    "cyber": ("Cyber", "Reproduce a real crash", "reproducing the crash", "crash"),
    "music": ("Music", "Compose in ABC notation", "a symbolic scorer", "family"),
    "general": ("General", "Knowledge work in a simulated workplace", "a weighted rubric", "industry"),
}
FACET_LABELS = {
    "language": "Brief language", "prog": "Programming language", "crash": "Crash type", "project": "Project", "sanitizer": "Sanitizer",
    "site": "Site type", "framework": "Framework", "style": "Style", "meter": "Meter", "voices": "Voices",
    "tempo": "Tempo", "family": "Family", "industry": "Industry", "source": "Source", "tier": "Tier",
}


# Webdev's tags are read from the brief's wording, not from metadata, so they say so.
LABEL_OVERRIDES = {("webdev", "site"): "Site type (mentioned)", ("webdev", "framework"): "Framework (mentioned)",
                   ("webdev", "style"): "Style (mentioned)"}


def write(path: Path, doc) -> None:
    # mtime=0 so an unchanged build produces byte-identical files (and no spurious commits)
    raw = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode()
    path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    for stale in (path.with_suffix(""),):   # drop the uncompressed copy from earlier builds
        stale.unlink(missing_ok=True)


def main() -> None:
    raw, files = fetch()
    frames = {d: pd.read_parquet(raw / p) for d, p in PARQUETS.items()}
    built = {
        "code": build_code(frames["code"]),
        "webdev": build_webdev(frames["webdev"]),
        "cyber": build_cyber(frames["cyber"]),
        "music": build_music(frames["music"]),
        "general": build_general(frames["general"], raw, files),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    everything, domains = [], []
    for d, (index, detail) in built.items():
        ids = [r["id"] for r in index]
        assert len(ids) == len(set(ids)), f"duplicate ids in {d}"
        everything += index
        write(OUT / f"{d}.json.gz", detail)
        name, task, verifier, main_facet = DOMAINS[d]
        facets = sorted({k for r in index for k in r["f"]}, key=lambda k: (k != main_facet, k))
        domains.append({"id": d, "name": name, "task": task, "verifier": verifier, "count": len(index),
                        "main": main_facet,
                        "facets": [[k, LABEL_OVERRIDES.get((d, k), FACET_LABELS.get(k, k))] for k in facets]})
        print(f"{d:8} {len(index):5} envs  detail {(OUT / f'{d}.json.gz').stat().st_size / 1e6:.2f} MB gz")

    index_doc = {"source": REPO, "total": len(everything), "domains": domains, "envs": everything}
    write(OUT / "index.json.gz", index_doc)
    print(f"index    {len(everything):5} envs  {(OUT / 'index.json.gz').stat().st_size / 1e6:.2f} MB gz")


if __name__ == "__main__":
    main()
