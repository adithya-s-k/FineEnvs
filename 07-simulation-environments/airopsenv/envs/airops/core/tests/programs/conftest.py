import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from program_helpers import APT_BASE, DATE, HUB, SLICE, TAXI_OUT, WINDOW

from airops_core.build import bts, pack
from airops_core.programs import advisories, candidates, rbs, repairs, unseen


@pytest.fixture(scope="session")
def slice_inputs(tmp_path_factory) -> Path:
    """v1's fixture slice of the inputs with its rows' TaxiOut, the NASR slice and placeholder capacity profiles
    added."""
    root = tmp_path_factory.mktemp("inputs") / "inputs"
    shutil.copytree(SLICE, root)
    with zipfile.ZipFile(bts.ontime_path(root, "2023-06")) as z:
        name = z.namelist()[0]
        rows = pd.read_csv(z.open(name), dtype=str)
    with zipfile.ZipFile(bts.ontime_path(root, "2023-06"), "w") as z:
        z.writestr(name, rows.assign(TaxiOut=pd.read_csv(TAXI_OUT, dtype=str).TaxiOut).to_csv(index=False))
    path = advisories.nasr_path(root)
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("APT_BASE.csv", APT_BASE.read_text())
    for hub in rbs.PROFILE:
        rbs.profile_path(root, hub).parent.mkdir(parents=True, exist_ok=True)
        rbs.profile_path(root, hub).write_bytes(b"%PDF-1.4 placeholder\n")
    return root


@pytest.fixture(scope="session")
def fits(slice_inputs, tmp_path_factory):
    return candidates.fits_of((DATE, HUB, slice_inputs, tmp_path_factory.mktemp("derived")))


@pytest.fixture(scope="session")
def built(slice_inputs, fits, tmp_path_factory):
    """The slice's JFK evening of 26 June 2023 as a hub-window task (eight tails, so the size floor comes down)."""
    return build(slice_inputs, fits, tmp_path_factory.mktemp("derived"))


@pytest.fixture(scope="session")
def task(built):
    return built.task


@pytest.fixture(scope="session")
def crewed(slice_inputs, fits, tmp_path_factory):
    """The same evening with the crew model at JetBlue's bases."""
    return build(slice_inputs, fits, tmp_path_factory.mktemp("derived"), repairs.BASES).task


def build(slice_inputs, fits, derived, bases=()):
    e_fit, e_median = unseen.table(fits, {"JFK": 1.0, "BOS": 1.0})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pack, "MIN_TAILS", 5)
        return candidates.build_candidate(DATE, HUB, slice_inputs, derived, e_fit, e_median,
                                          {"JFK": 1.0, "BOS": 1.0}, window=WINDOW, split="train", bases=bases)
