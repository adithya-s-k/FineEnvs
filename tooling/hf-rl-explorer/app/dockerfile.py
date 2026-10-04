"""Running a Dockerfile-only task on an HF Sandbox, which starts from a prebuilt image and builds nothing.

Most task Dockerfiles are one stage: FROM an image, then RUN, COPY, WORKDIR, ENV and ARG. Those can be replayed in
the sandbox itself: start it from the FROM image, run each RUN there, upload each COPY from the task's environment
folder, and keep ENV and WORKDIR for every later command (the agent's and the grader's too). What can't be replayed
(several stages, COPY --from, heredocs, ADD from a URL) is refused with the reason, so the run panel can say why.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field

IGNORED = {"CMD", "ENTRYPOINT", "EXPOSE", "LABEL", "HEALTHCHECK", "VOLUME", "STOPSIGNAL", "MAINTAINER", "USER", "SHELL"}


@dataclass
class Plan:
    base: str
    steps: list[tuple[str, object]] = field(default_factory=list)   # ("run", cmd) ("copy", (srcs, dst)) ("workdir", p) ("env", {k: v})


class NotReplayable(ValueError):
    pass


def _lines(text: str) -> list[str]:
    """Instructions, one per entry: comments dropped, continuation lines joined."""
    out, cur = [], ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if not cur and (not line.strip() or line.lstrip().startswith("#")):
            continue
        if cur and line.lstrip().startswith("#"):   # a comment inside a continued instruction
            continue
        if line.endswith("\\"):
            cur += line[:-1] + " "
            continue
        cur += line
        out.append(cur.strip())
        cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


def _subst(s: str, args: dict[str, str]) -> str:
    return re.sub(r"\$\{(\w+)(?::-([^}]*))?\}|\$(\w+)",
                  lambda m: args.get(m.group(1) or m.group(3), m.group(2) or "") if (m.group(1) or m.group(3)) in args or m.group(2) is not None
                  else m.group(0), s)


def _pairs(rest: str) -> dict[str, str]:
    """ENV/ARG bodies: `K=V K2="v 2"` or the old `K V` form."""
    parts = shlex.split(rest)
    if parts and "=" not in parts[0]:
        return {parts[0]: " ".join(parts[1:])}
    return dict(p.split("=", 1) if "=" in p else (p, "") for p in parts)


def plan(text: str) -> Plan:
    """The replay plan for a Dockerfile, or NotReplayable with the reason."""
    if re.search(r"<<-?\s*['\"]?\w+", text):
        raise NotReplayable("its Dockerfile uses a heredoc")
    args: dict[str, str] = {}
    base = None
    steps: list[tuple[str, object]] = []
    for line in _lines(text):
        op, _, rest = line.partition(" ")
        op, rest = op.upper(), rest.strip()
        if op == "FROM":
            if base:
                raise NotReplayable("its Dockerfile has several stages")
            base = _subst(rest.split()[0] if not rest.startswith("--") else rest.split()[1], args)
            continue
        if op == "ARG":
            for k, v in _pairs(rest).items():
                args.setdefault(k, v)
            continue
        if base is None:
            raise NotReplayable("its Dockerfile has no FROM")
        rest = _subst(rest, args)
        if op == "RUN":
            if rest.startswith("--"):
                flags = re.match(r"((?:--\S+\s+)+)(.*)", rest, re.S)
                rest = flags.group(2) if flags else rest
            if rest.startswith("["):
                try:
                    rest = " ".join(shlex.quote(x) for x in json.loads(rest))
                except ValueError:
                    raise NotReplayable("a RUN line it can't read")
            steps.append(("run", rest))
        elif op in ("COPY", "ADD"):
            parts = [p for p in (json.loads(rest) if rest.startswith("[") else shlex.split(rest))]
            flags = [p for p in parts if p.startswith("--")]
            if any(f.startswith("--from") for f in flags):
                raise NotReplayable("its Dockerfile copies from another stage")
            parts = [p for p in parts if not p.startswith("--")]
            if len(parts) < 2:
                raise NotReplayable("a COPY line it can't read")
            if any(re.match(r"https?://", p) for p in parts[:-1]):
                raise NotReplayable("its Dockerfile adds a file from a URL")
            steps.append(("copy", (parts[:-1], parts[-1])))
        elif op == "WORKDIR":
            steps.append(("workdir", rest))
        elif op == "ENV":
            steps.append(("env", _pairs(rest)))
        elif op in IGNORED or op == "ONBUILD":
            continue
        else:
            raise NotReplayable(f"its Dockerfile uses {op}")
    if not base:
        raise NotReplayable("its Dockerfile has no FROM")
    if base.lower() == "scratch":
        raise NotReplayable("its Dockerfile starts FROM scratch")
    return Plan(base=base, steps=steps)


def check(text: str | None) -> tuple[bool, str]:
    """Whether a task with this Dockerfile can run here, and if not, why."""
    if not text:
        return False, "it has neither a prebuilt image nor a Dockerfile"
    try:
        plan(text)
        return True, ""
    except NotReplayable as e:
        return False, str(e)
