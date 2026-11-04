"""The verifier's submission unpacking accepts only regular Submission*.lean files.

    python -m unittest discover -s tests
"""

from __future__ import annotations

import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

GRADE = Path(__file__).parents[1] / "openai_math_harbor" / "templates" / "tests" / "grade.py"
spec = importlib.util.spec_from_file_location("grade", GRADE)
grade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grade)


def make_tar(path: Path, entries: list[tuple[str, str, bytes]]) -> None:
    """entries: (name, kind, data) with kind in reg/dir/sym/lnk/fifo."""
    with tarfile.open(path, "w") as tar:
        for name, kind, data in entries:
            info = tarfile.TarInfo(name)
            if kind == "reg":
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                continue
            info.type = {"dir": tarfile.DIRTYPE, "sym": tarfile.SYMTYPE, "lnk": tarfile.LNKTYPE,
                         "fifo": tarfile.FIFOTYPE}[kind]
            info.linkname = data.decode()
            tar.addfile(info)


class UnpackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.tar = self.tmp / "submission.tar"
        self.dest = self.tmp / "run"
        self.dest.mkdir()
        grade.SUBMISSION_TAR = self.tar

    def unpack(self, entries):
        make_tar(self.tar, entries)
        return grade.unpack_submission(self.dest)

    def test_accepts_root_and_modules(self):
        info = self.unpack([
            ("Submission.lean", "reg", b"import Submission.A.B\n"),
            ("Submission", "dir", b""),
            ("Submission/A/B.lean", "reg", "theorem t : True := trivial -- ℕ\n".encode()),
        ])
        self.assertEqual(info["files"], 2)
        self.assertTrue((self.dest / "Submission" / "A" / "B.lean").is_file())

    def test_accepts_unicode_identifiers(self):
        info = self.unpack([
            ("Submission.lean", "reg", b"import Submission.Arith.Kappa\xe2\x82\x81\n"),
            ("Submission/Arith/Kappa₁.lean", "reg", b"theorem k : True := trivial\n"),
            ("Submission/αβ/Lemma'.lean", "reg", b""),
        ])
        self.assertEqual(info["files"], 3)

    def test_rejections(self):
        cases = {
            "missing root": [("Submission/A.lean", "reg", b"")],
            "path traversal": [("Submission.lean", "reg", b""), ("Submission/../../etc/x.lean", "reg", b"")],
            "absolute path": [("Submission.lean", "reg", b""), ("/tmp/Submission.lean", "reg", b"")],
            "symlink": [("Submission.lean", "sym", b"/tests/Challenge.lean")],
            "hardlink": [("Submission.lean", "reg", b""), ("Submission/X.lean", "lnk", b"Submission.lean")],
            "fifo": [("Submission.lean", "reg", b""), ("Submission/X.lean", "fifo", b"")],
            "other root": [("Submission.lean", "reg", b""), ("Challenge.lean", "reg", b"")],
            "lakefile": [("Submission.lean", "reg", b""), ("lakefile.lean", "reg", b"")],
            "non-lean": [("Submission.lean", "reg", b""), ("Submission/x.olean", "reg", b"")],
            "bad module name": [("Submission.lean", "reg", b""), ("Submission/a-b.lean", "reg", b"")],
            "digit first": [("Submission.lean", "reg", b""), ("Submission/1x.lean", "reg", b"")],
            "dot component": [("Submission.lean", "reg", b""), ("Submission/../x.lean", "reg", b"")],
            "space": [("Submission.lean", "reg", b""), ("Submission/a b.lean", "reg", b"")],
            "non-utf8": [("Submission.lean", "reg", b"\xff\xfe")],
            "duplicate": [("Submission.lean", "reg", b"a"), ("Submission.lean", "reg", b"b")],
        }
        for label, entries in cases.items():
            with self.subTest(label):
                for p in self.dest.iterdir():
                    import shutil
                    shutil.rmtree(p) if p.is_dir() else p.unlink()
                with self.assertRaises(grade.Rejected):
                    self.unpack(entries)
        self.assertFalse((self.tmp / "etc").exists())

    def test_missing_archive(self):
        with self.assertRaises(grade.Rejected):
            grade.unpack_submission(self.dest)

    def test_archive_symlink(self):
        self.tar.symlink_to("/etc/hostname")
        with self.assertRaises(grade.Rejected):
            grade.unpack_submission(self.dest)

    def test_timeout_kills_grandchildren(self):
        import time
        # A grandchild holding stdout open is what made the old `timeout`-based call hang.
        start = time.time()
        code, out, _ = grade.run_with_timeout(
            ["bash", "-c", "echo started; (sleep 120; echo late) & sleep 120"],
            cwd=self.tmp, timeout_sec=2)
        self.assertIsNone(code)
        self.assertIn("started", out)
        self.assertNotIn("late", out)
        self.assertLess(time.time() - start, 15)

    def test_run_with_timeout_normal_exit(self):
        code, out, _ = grade.run_with_timeout(["bash", "-c", "echo ok; exit 3"], cwd=self.tmp, timeout_sec=10)
        self.assertEqual((code, out.strip()), (3, "ok"))

    def test_ensure_readable_touches_only_what_needs_it(self):
        import os
        root = self.tmp / "tree"; (root / "sub").mkdir(parents=True)
        good, bad, writable = root / "good.olean", root / "sub" / "bad.olean", root / "w.olean"
        for f in (good, bad, writable): f.write_text("x")
        good.chmod(0o644); bad.chmod(0o600); writable.chmod(0o666)
        before = os.stat(good).st_ctime_ns
        changed = grade.ensure_readable((str(root),))
        self.assertEqual(changed, 2)
        self.assertEqual(bad.stat().st_mode & 0o777, 0o644)
        self.assertEqual(writable.stat().st_mode & 0o777, 0o644)
        self.assertEqual(os.stat(good).st_ctime_ns, before)   # untouched: no overlay copy-up
        self.assertEqual(grade.ensure_readable((str(root),)), 0)

    def test_out_of_memory_detection(self):
        self.assertTrue(grade.out_of_memory("✖ Building M\nerror: Lean exited with code 137\n"))
        self.assertTrue(grade.out_of_memory("uncaught exception: Child exited with 137"))
        self.assertFalse(grade.out_of_memory("uncaught exception: Child exited with 1"))
        reward, reason = grade.classify(1, "Building Submission\nerror: Lean exited with code 137\n")
        self.assertEqual(reward, 0)
        self.assertIn("out of memory", reason)

    def test_empty_scratch_only_touches_finished_modules(self):
        import os, subprocess, sys
        ir = self.tmp / "ir" / "Submission"; ir.mkdir(parents=True)
        def module(name, hash_age=None):
            for ext in (".setup.json", ".c"):
                (ir / (name + ext)).write_text("x" * 100)
            os.utime(ir / (name + ".setup.json"), (1000, 1000))
            if hash_age is not None:
                (ir / (name + ".c.hash")).write_text("1")
                os.utime(ir / (name + ".c.hash"), (1000 + hash_age, 1000 + hash_age))
        module("Done", hash_age=5)       # Lake finished: hash written after the setup
        module("Building")               # Lean or leanir may still read the setup
        module("Stale", hash_age=-5)     # hash older than the setup: a rebuild is in progress
        subprocess.run([sys.executable, "-c", grade.EMPTY_SCRATCH, str(self.tmp / "ir")], check=True)
        size = lambda n: (ir / n).stat().st_size
        self.assertEqual((size("Done.setup.json"), size("Done.c")), (0, 0))
        self.assertEqual((size("Building.setup.json"), size("Building.c")), (100, 100))
        self.assertEqual((size("Stale.setup.json"), size("Stale.c")), (100, 100))
        self.assertEqual(size("Done.c.hash"), 1)

    def test_out_of_disk_detection(self):
        out = "Building Submission\nerror: resource exhausted (error code: 28, no space left on device)\n"
        reward, reason = grade.classify(1, out)
        self.assertEqual(reward, 0)
        self.assertIn("out of disk", reason)
        self.assertTrue(grade.out_of_disk("OSError: [Errno 28] No space left on device"))

    def test_classify(self):
        ok = "Building Challenge\nBuilding Submission\nYour solution is okay!\n"
        self.assertEqual(grade.classify(0, ok)[0], 1)
        self.assertEqual(grade.classify(1, ok)[0], 0)              # success line but nonzero exit
        self.assertEqual(grade.classify(0, "Building Submission\n")[0], 0)  # exit 0, no success line
        self.assertEqual(grade.classify(None, ok)[0], 0)            # timeout
        self.assertIsNone(grade.classify(1, "Building Challenge\nerror\n")[0])  # trusted side broke
        self.assertIsNone(grade.classify(None, "Building Challenge\n")[0])      # timed out before the submission


if __name__ == "__main__":
    unittest.main()
