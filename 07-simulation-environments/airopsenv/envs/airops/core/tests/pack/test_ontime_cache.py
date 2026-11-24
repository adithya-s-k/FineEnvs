"""bts.read_ontime's parquet cache under parallel workers on a cold cache, as step 0 and build_pack start them."""

import multiprocessing as mp
import zipfile

import pandas as pd
import pytest

from airops_core.build import bts

WORKERS, ROUNDS, ROWS = 8, 3, 120_000


@pytest.fixture(scope="module")
def ontime_zip(tmp_path_factory):
    """A month zip with every column read_ontime keeps, big enough that the parquet write takes a while."""
    n = ROWS
    df = pd.DataFrame({
        "FlightDate": ["2023-06-01"] * n, "Reporting_Airline": ["B6"] * n, "Tail_Number": ["N123JB"] * n,
        "Flight_Number_Reporting_Airline": [str(k % 9000) for k in range(n)], "Origin": ["JFK"] * n,
        "Dest": ["BOS"] * n, "CancellationCode": [""] * n, "Div1Airport": [""] * n, "Div2Airport": [""] * n,
        "CRSDepTime": ["0830"] * n, "CRSArrTime": ["0945"] * n, "DepTime": ["0835"] * n,
        **{c: [0.0] * n for c in bts.NUMBERS}, "CRSElapsedTime": [75.0] * n})
    path = tmp_path_factory.mktemp("inputs") / "ontime_2023_06.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ontime.csv", df.to_csv(index=False))
    return path


def _read(path, derived, barrier, errors):
    barrier.wait()
    try:
        bts.read_ontime(path, derived)
    except Exception as e:  # reported to the parent, which fails the test
        errors.put(f"{type(e).__name__}: {e}")


def test_parallel_workers_on_a_cold_cache_all_read_the_month(ontime_zip, tmp_path):
    ctx = mp.get_context("fork")
    for r in range(ROUNDS):
        derived = tmp_path / f"derived{r}"
        barrier, errors = ctx.Barrier(WORKERS), ctx.Queue()
        procs = [ctx.Process(target=_read, args=(ontime_zip, derived, barrier, errors)) for _ in range(WORKERS)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(120)
        failed = []
        while not errors.empty():
            failed.append(errors.get())
        assert not failed, f"round {r}: {len(failed)} of {WORKERS} workers failed: {failed[0]}"
        assert all(p.exitcode == 0 for p in procs)
        assert [p.name for p in derived.iterdir()] == [f"ontime_{bts.fetch.sha256(ontime_zip)[:16]}.parquet"]
        assert len(bts.read_ontime(ontime_zip, derived)) == ROWS
