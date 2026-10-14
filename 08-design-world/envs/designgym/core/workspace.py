"""Where a GUI episode's images live on "this computer".

By default (no workspace) the task's images are `a1.png`, `a2.png`, ... in Pictures: what every eval and trace so
far saw. A *workspace* makes the computer look lived-in instead:

- the task's images have everyday names (`IMG_4021.jpg`, `beach-sunset.jpg`, `logo_final (1).png`) and sit in
  everyday folders (Downloads, Desktop, Pictures, Documents, or a project folder inside one of them), not
  necessarily all in the same one;
- other pictures sit next to them: *decoys*, real images from other designs. Opening one inserts it like any image
  (it is an asset of the episode), so picking the wrong file is a real mistake the grade counts;
- documents sit around too (a brief, an invoice, a zip), as on anyone's computer.

The prompt names the task's files (not where they are): the agent has to find them, and look at each to see which
part of the target it is, as a person would.

A workspace is a plain JSON spec, so a caller can pass its own (e.g. names from captions) or ask for a generic one:

    {"folder": "Downloads/Spring sale",
     "files": [{"name": "IMG_4021.jpg", "asset": "a1"}, {"name": "logo.png", "asset": "a2", "folder": "Desktop"},
               {"name": "coffee-cup.png", "asset": "x1"}, ...],
     "decoys": {"x1": {"task_id": "train-01234", "asset": "a3"}},
     "others": [{"path": "Documents/brief.docx"}, {"path": "Downloads/invoice-0921.pdf"}]}

`files` must name every task asset; a file's own `folder` overrides the spec's (where most of them are). Decoy ids
are `x1`, `x2`, ... and never collide with task assets.
"""

from __future__ import annotations

import io
import random
import re
import zipfile
from functools import lru_cache
from pathlib import Path

FOLDERS = ("Downloads", "Desktop", "Pictures", "Documents")
PROJECTS = ("Spring sale", "client work", "social posts", "campaign", "assets", "new folder", "event", "shop",
            "brand kit", "drafts", "final", "Instagram", "flyer", "website")
_PHOTO_NAMES = ("IMG_{n:04d}.jpg", "DSC0{n:04d}.JPG", "photo-{n}.jpg", "pexels-photo-{n:07d}.jpeg",
                "image ({k}).jpg", "PXL_2026{n:04d}_1{n:05d}.jpg", "unsplash-{n:06d}.jpg")
_GRAPHIC_NAMES = ("logo.png", "icon-{k}.png", "graphic ({k}).png", "shape_{n}.png", "Untitled-{k}.png",
                  "element-{n}.png", "sticker_{k}.png", "badge.png", "ornament ({k}).png")
_OTHERS = ("brief.docx", "invoice-{n:04d}.pdf", "notes.txt", "fonts.zip", "copy-final.docx", "budget.xlsx",
           "contract.pdf", "meeting notes.txt", "press-kit.zip", "references.pdf", "todo.txt")


def _photo(kind: str) -> bool:
    return kind in ("photo", "background")


def unique(name: str, taken: set[str]) -> str:
    """`name`, or `name (2)`, `name (3)`, ... before the extension, as a file manager would. Names are told apart by
    their stem too (an app may name an imported picture after it): `logo.png` and `logo.jpg` are not both used."""
    stem, dot, ext = name.rpartition(".")
    stem, ext = (stem, "." + ext) if dot else (name, "")
    key = lambda st: st.lower()  # noqa: E731
    if key(stem) not in taken:
        taken.add(key(stem))
        return name
    k = 2
    while key(f"{stem} ({k})") in taken:
        k += 1
    taken.add(key(f"{stem} ({k})"))
    return f"{stem} ({k}){ext}"


@lru_cache(maxsize=4096)
def opaque(path) -> bool:
    """Whether a picture has no see-through pixels (only those can be JPEGs: a JPEG has no transparency, and a
    cut-out saved as one would sit on a white box)."""
    from PIL import Image

    with Image.open(path) as img:
        if img.mode in ("RGB", "L", "CMYK", "YCbCr") and "transparency" not in img.info:
            return True
        return img.convert("RGBA").getchannel("A").getextrema()[0] >= 250


def generic_name(kind: str, rng: random.Random, taken: set[str], opaque: bool = True) -> str:
    """An everyday file name for an image of this kind (camera, stock site, download, export); a picture with
    see-through parts gets a PNG's kind of name."""
    pat = rng.choice(_PHOTO_NAMES if _photo(kind) and opaque else _GRAPHIC_NAMES)
    return unique(pat.format(n=rng.randrange(1, 9999999) % 10 ** 7, k=rng.randrange(1, 9)), taken)


def spread(files: list[dict], task_ids: set[str], folder: str, rng: random.Random) -> None:
    """Put the files in folders, in place: often all the task's images in `folder`, otherwise over two or three
    places (a second project folder, another of the home folders), as files collect on a real computer. Decoys sit
    beside them or elsewhere. A file in `folder` gets no `folder` key."""
    mine = [f for f in files if f["asset"] in task_ids]
    places = [folder]
    if len(mine) > 1 and rng.random() < 0.55:
        for _ in range(rng.randint(1, min(2, len(mine) - 1))):
            base = rng.choice(FOLDERS)
            cand = f"{base}/{rng.choice(PROJECTS)}" if rng.random() < 0.3 else base
            if cand not in places and not cand.startswith(folder + "/") and not folder.startswith(cand + "/"):
                places.append(cand)
    order = list(range(len(mine)))
    rng.shuffle(order)
    for k, i in enumerate(order):  # every place holds at least one of the task's images
        where = places[k] if k < len(places) else rng.choice(places)
        if where != folder:
            mine[i]["folder"] = where
    for f in files:
        if f["asset"] not in task_ids and rng.random() < 0.45:
            where = rng.choice((*places, *FOLDERS))
            if where != folder:
                f["folder"] = where


def folder_of(spec: dict, f: dict) -> str:
    """The folder (under home) a workspace file is in."""
    return str(f.get("folder") or spec["folder"]).strip("/")


def realistic(task: dict, pack, seed, *, decoys: tuple[int, int] = (3, 7), names: dict[str, str] | None = None,
              decoy_names: dict[tuple[str, str], str] | None = None) -> dict:
    """A generic workspace for one task: a seeded folder, everyday names (or `names` per asset id), a few decoys
    (images from other tasks of the same split, by kind: photos among photos) and some documents."""
    rng = random.Random(f"workspace:{seed}:{task['task_id']}")
    base = rng.choice(FOLDERS)
    folder = f"{base}/{rng.choice(PROJECTS)}" if rng.random() < 0.35 else base
    taken: set[str] = set()
    paths = pack.asset_paths(task)
    files = [{"name": unique(names[a["id"]], taken) if names and a["id"] in names else
              generic_name(a["kind"], rng, taken, opaque(str(paths[a["id"]]))), "asset": a["id"]}
             for a in task["assets"]]
    # decoys: other tasks' images, drawn from the same split so they look like the same kind of work
    split = task.get("split") or "train"
    count = pack.count(split)
    want = rng.randint(*decoys)
    spec_decoys: dict[str, dict] = {}
    tries = 0
    while len(spec_decoys) < want and tries < 50 and count > 1:
        tries += 1
        other = pack.at(split, rng.randrange(count))
        if other["task_id"] == task["task_id"] or not other["assets"]:
            continue
        a = rng.choice(other["assets"])
        if any(d["task_id"] == other["task_id"] and d["asset"] == a["id"] for d in spec_decoys.values()):
            continue
        did = f"x{len(spec_decoys) + 1}"
        spec_decoys[did] = {"task_id": other["task_id"], "asset": a["id"]}
        nm = (decoy_names or {}).get((other["task_id"], a["id"]))
        op = opaque(str(pack.asset_paths(other)[a["id"]]))
        files.append({"name": unique(nm, taken) if nm else generic_name(a["kind"], rng, taken, op), "asset": did})
    spread(files, {a["id"] for a in task["assets"]}, folder, rng)
    rng.shuffle(files)
    others = []
    for _ in range(rng.randint(2, 5)):
        where = rng.choice((folder, *FOLDERS))
        nm = rng.choice(_OTHERS).format(n=rng.randrange(1, 9999))
        others.append({"path": f"{where}/{nm}"})
    return {"folder": folder, "files": files, "decoys": spec_decoys, "others": others}


def validate(spec: dict, task: dict) -> dict:
    """Check a caller's spec: every task asset is a file, names are plain file names, decoy ids are x<n>."""
    if not isinstance(spec, dict) or "folder" not in spec or "files" not in spec:
        raise ValueError("a workspace needs a folder and files")
    folder = str(spec["folder"]).strip("/")
    if not folder or ".." in folder.split("/"):
        raise ValueError(f"bad workspace folder {spec['folder']!r}")
    ids = {a["id"] for a in task["assets"]}
    decoys = spec.get("decoys") or {}
    for d in decoys:
        if not re.fullmatch(r"x\d+", d) or d in ids:
            raise ValueError(f"decoy ids are x1, x2, ... (not {d!r})")
    named = set()
    for f in spec["files"]:
        if "/" in f["name"] or f["name"] in ("", ".", ".."):
            raise ValueError(f"bad file name {f['name']!r}")
        if "folder" in f:
            sub = str(f["folder"]).strip("/")
            if not sub or ".." in sub.split("/"):
                raise ValueError(f"bad folder {f['folder']!r} for {f['name']!r}")
        if f["asset"] not in ids and f["asset"] not in decoys:
            raise ValueError(f"file {f['name']!r} names unknown asset {f['asset']!r}")
        named.add(f["asset"])
    if ids - named:
        raise ValueError(f"the workspace has no file for {sorted(ids - named)}")
    for o in spec.get("others") or []:
        if ".." in str(o["path"]).split("/"):
            raise ValueError(f"bad path {o['path']!r}")
    return {**spec, "folder": folder}


def decoy_assets(spec: dict, pack) -> tuple[dict[str, Path], dict[str, dict]]:
    """The decoys' files and their asset context (aspect, kind), under their decoy ids."""
    paths, ctx = {}, {}
    for did, src in (spec.get("decoys") or {}).items():
        other = pack.get(src["task_id"])
        paths[did] = pack.asset_paths(other)[src["asset"]]
        ctx[did] = pack.asset_ctx(other)[src["asset"]]
    return paths, ctx


def document_bytes(path: str) -> bytes:
    """Small, valid contents for a clutter document of this type (it opens in a viewer, it is not a task file)."""
    name = Path(path).name
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext == "pdf":
        return (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
                b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 595 842]>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n")
    if ext in ("zip", "docx", "xlsx"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("readme.txt", f"{name}\n")
        return buf.getvalue()
    rng = random.Random(name)  # a text file with a few everyday lines in it
    lines = ["call printer about proofs", "send invoice by friday", "fonts: check licence", "logo v2 - darker?",
             "book the venue", "reply to Sam re: dates", "order more paper", "pick up samples", "colours: see brief",
             "photos from the shoot - final selects", "update price list", "meeting moved to 3pm"]
    rng.shuffle(lines)
    return (f"{Path(name).stem}\n\n" + "\n".join(f"- {x}" for x in lines[: rng.randint(3, 9)]) + "\n").encode()


def layout(spec: dict) -> dict:
    """What the editor's own file window shows: each file's name, asset and folder, and the documents."""
    return {"folder": spec["folder"], "files": [{**f, "folder": folder_of(spec, f)} for f in spec["files"]],
            "others": [dict(o) for o in spec.get("others") or []]}


def file_names(spec: dict) -> dict[str, str]:
    """asset id -> file name, for the task's own images (what the prompt lists)."""
    return {f["asset"]: f["name"] for f in spec["files"] if not re.fullmatch(r"x\d+", f["asset"])}


def file_facts(path) -> dict:
    """A picture file's pixel size and bytes on disk, as a file window shows them."""
    return dict(_facts(str(path)))


@lru_cache(maxsize=4096)
def _facts(path: str) -> tuple:
    from PIL import Image

    p = Path(path)
    with Image.open(p) as img:
        w, h = img.size
    return (("w", w), ("h", h), ("bytes", p.stat().st_size))
