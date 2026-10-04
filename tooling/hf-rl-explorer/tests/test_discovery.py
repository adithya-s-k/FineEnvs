"""Discovery evidence is metadata, not a runtime attestation or an editorial decision."""
from types import SimpleNamespace

import pytest
from huggingface_hub import SpaceInfo

from app import catalog


def space(sid="org/env", *, tags=(), files=(), **kw):
    return SpaceInfo(id=sid, sdk="docker", private=False, tags=list(tags), likes=0,
                     siblings=[{"rfilename": f} for f in files], **kw)


@pytest.mark.parametrize("files", [("openenv.yaml",), ("my_env/openenv.yml",)])
def test_manifest_discovery(files):
    d = catalog._space_summary(space(files=files))
    assert d["openenv"] and d["manifest"] == files[0]


@pytest.mark.parametrize("file", ["src/openenv.yaml", "templates/openenv.yaml", "examples/openenv.yaml",
                                 "build/openenv.yaml", "vendor/openenv.yaml", "envs/echo/openenv.yaml",
                                 "src/openenv/cli/templates/openenv.yaml", ".venv/openenv.yaml"])
def test_vendored_manifests_are_not_declarations(file):
    d = catalog._space_summary(space(files=[file]))
    assert not d["openenv"] and d["manifest"] is None


@pytest.mark.parametrize("tag", ["openenv", "library:openenv", "openenv-0.7.0"])
def test_tag_only_remains_a_candidate_without_a_manifest(tag):
    d = catalog._space_summary(space(tags=[tag]))
    assert d["openenv"] and d["manifest"] is None


def test_names_and_generic_rl_mcp_tags_do_not_establish_openenv():
    d = catalog._space_summary(space("openenv/not-an-env", tags=["rl-environment", "mcp-server"]))
    assert not d["openenv"] and d["mcp"]


def test_space_census_deduplicates_and_applies_visibility_to_featured(monkeypatch):
    good = space(tags=["openenv"])
    article = space("org/article", tags=["openenv", "research-article-template"])
    private = space("org/private"); private.private = True
    static = space("org/static"); static.sdk = "static"
    by = {s.id: s for s in [good, article, private, static]}
    monkeypatch.setattr(catalog, "collections", lambda: [{"ids": [f"space:{s}" for s in by]}])
    monkeypatch.setattr(catalog, "_api", lambda: SimpleNamespace(list_spaces=lambda **kw: iter(by.values()),
                                                               space_info=lambda sid, **kw: by[sid]))
    assert [r["id"] for r in catalog.spaces()] == [good.id]


def test_failed_full_census_is_not_silently_published(monkeypatch):
    def listing(**kw):
        yield space(tags=["openenv"])
        raise ConnectionError("page two unavailable")

    monkeypatch.setattr(catalog, "collections", lambda: [])
    monkeypatch.setattr(catalog, "_api", lambda: SimpleNamespace(list_spaces=listing))
    with pytest.raises(ConnectionError):
        catalog.spaces(full=True)
    assert len(catalog.spaces(full=False)) == 1   # a quick bootstrap may serve candidates while refreshing


def test_collection_never_reclassifies_framework(monkeypatch):
    d = catalog._space_summary(space(tags=["ors"]))
    monkeypatch.setattr(catalog, "_listing_rows", lambda: [d])
    monkeypatch.setattr(catalog, "collections", lambda: [{"id": "fineenvs", "ids": [d["key"]]}])
    monkeypatch.setattr(catalog, "hidden", lambda: set())
    assert catalog.environments()[0]["framework"] == "ors"
