"""Exercise the shipped entrypoint without starting services or using real credentials."""
import os
import subprocess
from pathlib import Path

import pytest

ARTICLE = Path(__file__).resolve().parents[2]


def run_entrypoint(tmp_path, **settings):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("chmod", "nginx"):
        executable = bindir / name
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    env = dict(os.environ, PATH=f"{bindir}:/usr/bin:/bin")
    for key in ("NOTION_TOKEN", "NOTION_PAGE_ID", "ENABLE_NOTION_IMPORT"):
        env.pop(key, None)
    env.update(settings)
    return subprocess.run(["/bin/bash", str(ARTICLE / "entrypoint.sh")], env=env,
                          capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize("missing", ["NOTION_PAGE_ID", "NOTION_TOKEN"])
def test_missing_notion_setting_reports_presence_without_values(tmp_path, missing):
    settings = {"ENABLE_NOTION_IMPORT": "true", "NOTION_TOKEN": "fake-secret-never-log",
                "NOTION_PAGE_ID": "fake-private-page-never-log"}
    settings.pop(missing)
    output = run_entrypoint(tmp_path, **settings)
    assert "fake-secret-never-log" not in output
    assert "fake-private-page-never-log" not in output
    for key in ("NOTION_TOKEN", "NOTION_PAGE_ID"):
        assert f"{key}: {'NOT SET' if key == missing else 'SET'}" in output
    assert "Using pre-built content" in output


def test_disabled_import_does_not_emit_setting_values(tmp_path):
    output = run_entrypoint(tmp_path, ENABLE_NOTION_IMPORT="false",
                            NOTION_TOKEN="fake-secret-never-log", NOTION_PAGE_ID="fake-page")
    assert "Notion import disabled" in output
    assert "fake-secret-never-log" not in output
    assert "fake-page" not in output
