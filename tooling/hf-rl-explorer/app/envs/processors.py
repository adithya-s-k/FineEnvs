"""The processors that ship with the explorer, most specific first. Add yours here (see CUSTOM_ENVS.md):

    class MyFormat(Processor):
        id, name, framework = "my-format", "My format", "My format"
        about = "One sentence: what a row is and how it's graded."
        def match(self, ds): return 0.9 if ds.has("my_column") else 0
        def view(self, row, i, ds, roles): ... return {"title", "id", "chips", "sections", "withheld", "glance", "run"?}

and list it in PROCESSORS. `card` and `roles` have sensible defaults (base.Processor), so most only need `view`.
"""

from __future__ import annotations

import base64
import binascii
import gzip
import io
import posixpath
import re
import tarfile
import tomllib
from typing import Any

from .base import Dataset, Processor, first_line, is_messages, parse_json, section, text_of, withhold
from .nemogym import HIDE as NEMO_HIDE
from .nemogym import NemoGym
from .verifiers_rows import SWE_HIDE
from .verifiers_rows import Verifiers
from .verl_rows import HIDE as VERL_HIDE
from .verl_rows import Verl


# ── Harbor tasks packed into rows (a tar.gz per row), e.g. open-thoughts/TaskTrove ──────────────────────────
BLOB_COLS = ("task_binary", "task_tar", "tar", "archive", "task_archive", "files_tar", "bundle")
MAX_TAR = 25 * 2**20


def unpack(blob: Any) -> dict[str, bytes]:
    """A row's packed task as {path: bytes}: base64 text, a list of byte values, or bytes; tar, tar.gz or zip."""
    if isinstance(blob, str):
        try:
            data = base64.b64decode(blob, validate=False)
        except (binascii.Error, ValueError):
            return {}
    elif isinstance(blob, list):
        data = bytes(x & 0xFF for x in blob[:MAX_TAR])
    elif isinstance(blob, (bytes, bytearray)):
        data = bytes(blob)
    else:
        return {}
    if len(data) > MAX_TAR:
        return {}
    files: dict[str, bytes] = {}
    try:
        if data[:2] == b"PK":
            import zipfile

            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for n in z.namelist()[:3000]:
                    if not n.endswith("/") and z.getinfo(n).file_size < 4 * 2**20:
                        files[n] = z.read(n)
        else:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
                total = 0
                for m in t.getmembers()[:3000]:
                    if m.isfile() and m.size < 4 * 2**20 and total < 60 * 2**20:
                        f = t.extractfile(m)
                        if f:
                            files[m.name] = f.read()
                            total += m.size
    except (tarfile.TarError, OSError, EOFError, gzip.BadGzipFile, ValueError):
        return {}
    # a folder around everything (task_0/task.toml): strip it, so paths read like a Harbor task's own
    names = [n.lstrip("./") for n in files]
    files = dict(zip(names, files.values()))
    tomls = [n for n in names if posixpath.basename(n) == "task.toml"]
    root = posixpath.dirname(min(tomls, key=lambda n: n.count("/"))) if tomls else ""
    if root:
        files = {n[len(root) + 1:]: b for n, b in files.items() if n.startswith(root + "/")}
    return {n: b for n, b in files.items() if n and not any(p.startswith("..") for p in n.split("/"))}


class PackedHarbor(Processor):
    id, name, framework = "harbor-packed", "Harbor tasks in rows", "Harbor (packed)"
    about = "Each row packs one whole Harbor task (task.toml, instruction.md, tests/, environment/) into an archive; it's unpacked when opened."

    def blob(self, ds: Dataset) -> str | None:
        return next((c for c in ds.columns if c.lower() in BLOB_COLS), None)

    def match(self, ds: Dataset) -> float:
        col = self.blob(ds)
        if not col or not any(r.get(col) for r in ds.sample[:20]):
            return 0
        if Traces().conv(ds):   # recorded rollouts that also pack the task they ran on: traces, with the task inside
            return 0.5
        return 0.97 if any(t in ds.tags for t in ("library:harbor", "harbor")) else 0.8

    def roles(self, ds: Dataset) -> dict[str, Any]:
        """The preamble the first tasks' instructions share ("Environment Setup … # Terminal Automation Request"),
        so a title can be the task's own first line."""
        from .base import _common_prefix

        col = self.blob(ds)
        texts = []
        for r in ds.sample[:20]:
            b = r.get(col)
            if isinstance(b, (str, list, bytes)) and len(b) < 2 * 2**20:
                texts.append(unpack(b).get("instruction.md", b"").decode("utf-8", "replace"))
        return {"task": None, "id": "path", "title": None, "answer": [], "grading": [], "environment": [],
                "_prefix": _common_prefix(texts) if len([t for t in texts if t]) >= 3 else ""}

    def _title(self, instruction: str, meta: dict[str, Any], roles: dict[str, Any], fallback: str) -> tuple[str, str]:
        p = roles.get("_prefix") or ""
        own = instruction[len(p):].lstrip() if p and instruction.startswith(p) else instruction
        title = str(meta.get("title") or "") or first_line(own) or fallback
        return title, re.sub(r"\s+", " ", own).strip()

    def card(self, row: dict[str, Any], i: int, roles: dict[str, Any]) -> dict[str, Any]:
        ident = next((str(row[c]) for c in ("path", "task_id", "id", "name") if row.get(c) is not None), None)
        blob = next((row[c] for c in row if c.lower() in BLOB_COLS), None)
        files = unpack(blob) if isinstance(blob, (str, list, bytes)) and len(blob) < 2 * 2**20 else {}
        instruction = files.get("instruction.md", b"").decode("utf-8", "replace")
        meta: dict[str, Any] = {}
        try:
            meta = (tomllib.loads(files["task.toml"].decode("utf-8", "replace")).get("metadata") or {}) if "task.toml" in files else {}
        except tomllib.TOMLDecodeError:
            pass
        title, rest = self._title(instruction, meta, roles, ident or f"Task {i}")
        chips = [str(meta[k]) for k in ("difficulty", "category") if isinstance(meta.get(k), str)][:2]
        snippet = rest[len(title.rstrip("…")):].strip() if rest.startswith(title.rstrip("…")[:60]) else rest
        return {"i": i, "id": ident, "title": title[:200], "snippet": snippet[:260], "chips": chips,
                "_text": "" if meta.get("title") or roles.get("_prefix") else instruction}

    def view(self, row: dict[str, Any], i: int, ds: Dataset, roles: dict[str, Any]) -> dict[str, Any]:
        from .. import catalog

        files = unpack(row.get(self.blob(ds)))
        ident = next((str(row[c]) for c in ("path", "task_id", "id", "name") if row.get(c) is not None), None)
        if not files:
            return {"title": ident or f"Task {i}", "id": ident, "chips": [], "withheld": [], "glance": [["Row", f"{i:,}"]],
                    "sections": [section("task", "The task", "This row's archive couldn't be opened (too large, or not a tar/zip).", kind="markdown")]}
        texts = {n: b.decode("utf-8", "replace") for n, b in files.items() if b"\x00" not in b[:4096]}
        toml_text = texts.get("task.toml", "")
        try:
            doc = tomllib.loads(toml_text) if toml_text else {}
        except tomllib.TOMLDecodeError:
            doc = {}
        meta = doc.get("metadata") or {}
        instruction = texts.get("instruction.md", "")
        title, _ = self._title(instruction, meta, roles, ident or f"Task {i}")
        sections = [section("task", "The task", instruction or "This task has no instruction.md.", kind="markdown", note="instruction.md, word for word")]
        test_sh = texts.get("tests/test.sh")
        if test_sh and not catalog._withheld("tests/test.sh", test_sh):
            sections.append(section("grading", "How it's graded", {"path": "tests/test.sh", "text": catalog._mask_answers("tests/test.sh", test_sh)},
                                    kind="code", note="tests/test.sh, run after the agent"))
        docker = texts.get("environment/Dockerfile")
        if docker:
            sections.append(section("environment", "What it runs in", {"path": "environment/Dockerfile", "text": docker}, kind="code"))
        clean_meta, gone = withhold({k: v for k, v in meta.items() if not catalog.CONTACT_KEY.search(k)})
        if clean_meta:
            sections.append(section("metadata", "Metadata", clean_meta, kind="value"))
        tree = [{"path": n, "size": len(b), "withheld": catalog._withheld(n, texts.get(n))} for n, b in sorted(files.items())]
        sections.append(section("files", "Files", {"tree": tree}, kind="files", note=f"{len(tree)} files, unpacked from the row"))
        env = doc.get("environment") or {}
        from .. import catalog as _catalog
        look = {"env": {"image": env.get("docker_image"), "network": _catalog._network(env),
                        "compose": any(re.match(r"environment/(docker-)?compose\.ya?ml$", n) for n in files)}}
        runnable = _catalog.runnable(look, texts)
        return {"packed_run": {"runnable": runnable, "verifier_env": sorted(((doc.get("verifier") or {}).get("env") or {}).keys()),
                               "toml": toml_text or ""},
                "title": title[:200], "id": ident, "chips": [str(meta[k]) for k in ("difficulty", "category") if isinstance(meta.get(k), str)][:3],
                "sections": sections, "withheld": gone + [t["path"] for t in tree if t["withheld"]],
                "glance": [["Row", f"{i:,}"], ["Task", ident or "–"], ["Runs in", env.get("docker_image") or ("a Dockerfile" if docker else "–")],
                           ["Graded by", "tests/test.sh" if test_sh else "–"], ["Files", f"{len(tree):,}"]]}

    def file(self, row: dict[str, Any], ds: Dataset, rel: str) -> dict[str, Any]:
        """One file of the unpacked task, for the file viewer (answers withheld, as for any Harbor task)."""
        from .. import catalog

        files = unpack(row.get(self.blob(ds)))
        if rel not in files:
            return {"path": rel, "error": "not a file in this task"}
        b = files[rel]
        if b"\x00" in b[:4096]:
            return {"path": rel, "size": len(b), "binary": True}
        text = b.decode("utf-8", "replace")
        if catalog._withheld(rel, text):
            return {"path": rel, "size": len(b), "withheld": True}
        if rel == "task.toml":
            text = catalog._masked_toml(text)
        return {"path": rel, "size": len(b), "text": catalog._mask_answers(rel, text)[:512 * 1024], "truncated": len(text) > 512 * 1024}


# ── XiaomiMiMo/MiMo-V2.6-RL-oss, the raw form the MiMo explorer reads ────────────────────────────────────
MIMO_HIDE = re.compile(r"^(patch|gold_patch|fix_patch|solution_patch|poc|poc_\w+|reference_\w+)$", re.I)


def _slug(task_id: str) -> str:   # mimo_harbor.adapter.slug: how the Harbor conversion names its task folders
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in task_id.lower()).strip("-.")


class MiMo(Processor):
    id, name, framework = "mimo", "MiMo-V2.6 RL", "MiMo raw"
    about = ("Xiaomi's MiMo-V2.6 RL environments as released: one verl-style row per environment (a chat prompt, a "
             "reward style, and an instance with its image, tests or rubric). Each has a Harbor twin that runs here.")

    def match(self, ds: Dataset) -> float:
        if ds.spec.lower() == "xiaomimimo/mimo-v2.6-rl-oss":
            return 1.0
        # MiMo's rows are verl-shaped with an agent_name; what sets them apart is the instance packed in extra_info
        packed = any(isinstance(parse_json(r.get("extra_info")), dict) and "instance_json" in parse_json(r.get("extra_info")) for r in ds.sample[:10])
        return 0.92 if ds.has("data_source", "ability", "agent_name", "extra_info") and packed else 0

    def instance(self, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        extra = parse_json(row.get("extra_info")) or {}
        inst = parse_json(extra.get("instance_json")) if isinstance(extra, dict) else {}
        return (extra if isinstance(extra, dict) else {}), (inst if isinstance(inst, dict) else {})

    def task_id(self, row: dict[str, Any], config: str) -> str | None:
        extra, inst = self.instance(row)
        tid = inst.get("instance_id") or extra.get("instance_id")
        if not tid and config == "music" and extra.get("src_id") is not None:
            tid = f"music-{extra['src_id']}"
        return str(tid) if tid else None

    def card(self, row: dict[str, Any], i: int, roles: dict[str, Any]) -> dict[str, Any]:
        extra, inst = self.instance(row)
        prompt = text_of(row.get("prompt"))
        title = inst.get("display_title") or first_line(inst.get("description") or "") or first_line(prompt) or f"Row {i}"
        chips = [x for x in (row.get("ability"), inst.get("category"), inst.get("subcategory"), extra.get("lang")) if isinstance(x, str) and x][:4]
        tid = inst.get("instance_id") or extra.get("instance_id") or (f"music-{extra['src_id']}" if extra.get("src_id") is not None else None)
        return {"i": i, "id": tid, "title": str(title)[:200], "snippet": re.sub(r"\s+", " ", prompt)[:260] if first_line(prompt) != title else "", "chips": chips}

    def view(self, row: dict[str, Any], i: int, ds: Dataset, roles: dict[str, Any]) -> dict[str, Any]:
        config = roles.get("_config", "")
        extra, inst = self.instance(row)
        inst = {k: v for k, v in inst.items() if not MIMO_HIDE.match(k)}
        clean_inst, gone = withhold({k: parse_json(v) for k, v in inst.items()})
        prompt = parse_json(row.get("prompt"))
        rm = parse_json(row.get("reward_model")) or {}
        sections = [section("task", "The task", prompt, note="the prompt, as the model got it")]
        grading: dict[str, Any] = {"Reward style": rm.get("style") if isinstance(rm, dict) else None}
        for k in ("test_command", "verifier_timeout_sec", "description"):
            if clean_inst.get(k) is not None:
                grading[k] = clean_inst.pop(k)
        tests = clean_inst.pop("test_patch", None)
        rubric = clean_inst.pop("tests_files", None)
        sections.append(section("grading", "How it's graded", {k: v for k, v in grading.items() if v is not None}, kind="value",
                                note="from reward_model and the instance"))
        if tests:
            sections.append(section("tests", "Hidden tests", {"path": "test.patch", "text": tests}, kind="code", note="applied after the agent, then run"))
        if rubric:
            sections.append(section("rubric", "Verifier files", rubric, kind="value", note="the rubric's questions; its anchors are withheld"))
        env = {k: clean_inst.pop(k) for k in ("docker_image", "cwd", "cpus", "memory_mb", "storage_mb", "allow_internet", "agent_timeout_sec") if k in clean_inst}
        if env:
            sections.append(section("environment", "What it runs in", env, kind="value"))
        if config == "music":
            sections.append(section("music", "The brief", {k: v for k, v in extra.items() if k not in ("index",)}, kind="value"))
        rest = {k: v for k, v in clean_inst.items() if k not in ("problem_statement", "instance_id", "dataset_type")}
        if rest:
            sections.append(section("data", "The rest of the instance", rest, kind="value"))
        tid = self.task_id(row, config)
        run = twin(config, tid)
        return {"title": self.card(row, i, roles)["title"], "id": tid, "chips": self.card(row, i, roles)["chips"], "sections": sections,
                "withheld": sorted(set(gone + [f"instance.{k}" for k in self.instance(row)[1] if MIMO_HIDE.match(k)])),
                "glance": [["Domain", config or "–"], ["Ability", row.get("ability") or "–"], ["Agent", row.get("agent_name") or "–"],
                           ["Reward", (rm.get("style") if isinstance(rm, dict) else None) or "–"], ["Image", env.get("docker_image") or "–"]],
                "run": run}


def twin(config: str, tid: str | None) -> dict[str, Any] | None:
    """A MiMo row's Harbor task in the FineEnvs conversion, when this explorer has that dataset's index."""
    from .. import catalog

    if not tid:
        return None
    path = f"tasks/{_slug(tid)}"
    for name in ([config, "terminal"] if config == "general" else [config]):
        spec = f"FineEnvs/MiMo-V2.6-RL-harbor-{name}"
        idx = catalog._read_index(spec)
        if idx and any(r["path"] == path for r in idx["tasks"]):
            return {"dataset": spec, "path": path, "label": f"{spec.split('/')[1]} · {path}"}
    return None


# ── verl / SkyRL RL rows (data_source, prompt, ability, reward_model | env_class, reward_spec): verl_rows.py ─────
# ── NeMo Gym (nvidia/Nemotron-RL-*): responses_create_params, agent_ref and the verifier's fields ──────────────
# read in nemogym.py, with NeMo Gym's own configs: which resources server grades a row, and how to run it
# ── agent traces (conversations / messages with the agent's turns), e.g. open-thoughts/AgentTrove ──────────
class Traces(Processor):
    id, name, framework = "traces", "Agent traces", "Traces"
    about = "Each row is a recorded rollout: the conversation between an agent and its environment, with its result."

    def conv(self, ds: Dataset) -> str | None:
        for c in ("conversations", "messages", "trajectory", "conversation"):
            if c in ds.columns and any(is_messages(parse_json(r.get(c))) and any(m.get("role") == "assistant" for m in parse_json(r.get(c))) for r in ds.sample[:5]):
                return c
        return None

    def match(self, ds: Dataset) -> float:
        c = self.conv(ds)
        if not c:
            return 0
        return 0.85 if any(x in ds.columns for x in ("agent", "model", "trial_name", "result", "reward")) else 0.5

    def view(self, row, i, ds, roles):
        c = self.conv(ds)
        clean, gone = withhold({k: parse_json(v) for k, v in row.items() if k != c})
        conv = parse_json(row.get(c)) or []
        turns = sum(1 for m in conv if isinstance(m, dict) and m.get("role") == "assistant")
        result = {k: clean.pop(k) for k in ("result", "reward", "score", "success", "passed", "judgment") if k in clean}
        sections = [section("trace", "The trajectory", conv, kind="messages", note=f"{len(conv)} messages, {turns} from the agent")]
        packed = PackedHarbor()
        blob = packed.blob(ds)
        if blob and row.get(blob):   # the task it ran on, packed alongside
            for k in (blob,):
                clean.pop(k, None)
            task = packed.view(row, i, ds, packed.roles(ds))
            files = next((x for x in task["sections"] if x["kind"] == "files"), None)
            if files:
                sections.append({**files, "title": "The task's files"})
            gone = gone + [w for w in task["withheld"] if w not in gone]
        if result:
            sections.append(section("result", "Result", result, kind="value"))
        sections.append(section("data", "About this run", clean, kind="value"))
        first_user = next((m.get("content") for m in conv if isinstance(m, dict) and m.get("role") == "user"), "")
        title = str(clean.get("trial_name") or clean.get("task") or first_line(first_user if isinstance(first_user, str) else "") or f"Trace {i}")
        return {"title": title[:200], "id": str(clean.get("run_id") or clean.get("id") or i), "chips": [str(clean[k]) for k in ("agent", "model") if clean.get(k)][:3],
                "sections": sections, "withheld": gone, "glance": [["Row", f"{i:,}"], ["Agent turns", f"{turns:,}"], *[[k.title(), str(v)[:60]] for k, v in result.items()],
                                                               *[[k.title(), str(clean[k])[:60]] for k in ("agent", "model") if clean.get(k)]]}

    def file(self, row: dict[str, Any], ds: Dataset, rel: str) -> dict[str, Any]:
        return PackedHarbor().file(row, ds, rel)

    def card(self, row, i, roles):
        title = str(row.get("trial_name") or row.get("task") or f"Trace {i}")
        chips = [str(row[k]) for k in ("agent", "model", "result") if isinstance(row.get(k), str) and row.get(k)][:3]
        return {"i": i, "id": str(row.get("run_id") or i), "title": title[:200], "snippet": "", "chips": chips}


# ── data for Verifiers environments (Prime's Environments Hub): verifiers_rows.py ─────────────────────────────
class Generic(Processor):
    pass


PROCESSORS: list[Processor] = [PackedHarbor(), MiMo(), NemoGym(), Verl(), Traces(), Verifiers(), Generic()]
# fields each format's reader knows hold answers though their names don't say so: search never matches them either
HIDDEN: frozenset[str] = NEMO_HIDE | VERL_HIDE | SWE_HIDE
