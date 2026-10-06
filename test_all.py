#!/usr/bin/env python3
"""Tests for the whole toolbox: common, arr, ani, bpc, bpcmeta, cls, btx, astc, etc2.

One self-contained file, standard library ``unittest`` only (pytest runs it too).
Every input (zip, .bpc, PNG, KTX, COL/CLS, ANI/IFP) is generated on the fly,
nothing has to be checked in next to it.

    python -m unittest test_all -v
    python -m pytest test_all.py                        # if you use pytest
    coverage run --branch -m unittest test_all && coverage report -m

Put this file next to the tool scripts (or in a tests/ folder one level below).
"""

import argparse
import contextlib
import io
import os
import random
import runpy
import struct
import sys
import tempfile
import unittest
import zipfile
import zlib
from concurrent.futures import BrokenExecutor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from unittest import mock


def _find_root():
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here, os.path.dirname(here)):
        if os.path.exists(os.path.join(folder, "common.py")):
            return folder
    raise RuntimeError("common.py not found: put this file next to the tool scripts")


ROOT = _find_root()
sys.path.insert(0, ROOT)
# the tests must not depend on an optional rsnumpy that happens to be installed
os.environ["BTX_ARRAY_BACKEND"] = "numpy"

from PIL import Image, PngImagePlugin  # noqa: E402

import ani  # noqa: E402
import arr  # noqa: E402
import astc  # noqa: E402
import badge  # noqa: E402
import bpc  # noqa: E402
import bpcmeta  # noqa: E402
import btx  # noqa: E402
import cls  # noqa: E402
import common  # noqa: E402
import etc2  # noqa: E402
from arr import np  # noqa: E402
from common import Pool  # noqa: E402

KEY = b"1cK1a5UF2tU8*G2lW#&%"  # spelled out here on purpose: it is part of the format


# ----------------------------------------------------------------- helpers

def run_main(module, *args):
    """Run module.main() with the given argv. Returns (exit code, stdout, stderr)."""
    out, err, code = io.StringIO(), io.StringIO(), 0
    argv = [module.__name__ + ".py"] + [str(a) for a in args]
    with mock.patch.object(sys, "argv", argv), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            module.main()
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return code, out.getvalue(), err.getvalue()


def capture(fn, *args, **kwargs):
    """Call fn, return (result, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        res = fn(*args, **kwargs)
    return res, out.getvalue(), err.getvalue()


def run_script(name, *args):
    """Execute a script as ``python name.py args`` (covers the __main__ guard)."""
    out, err, code = io.StringIO(), io.StringIO(), 0
    argv = [name] + [str(a) for a in args]
    with mock.patch.object(sys, "argv", argv), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            runpy.run_path(os.path.join(ROOT, name), run_name="__main__")
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return code, out.getvalue(), err.getvalue()


def naive_xor(data, offset=0):
    return bytes(b ^ KEY[(offset + i) % len(KEY)] for i, b in enumerate(data))


def make_zip(entries, compress=zipfile.ZIP_DEFLATED):
    """entries: {name: bytes}; a name ending in / is a directory entry."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compress) as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return buf.getvalue()


def make_png(width=16, height=16, alpha=False, seed=0, extra=None):
    """A deterministic gradient + noise RGBA PNG (bytes)."""
    rnd = random.Random(seed)
    img = Image.new("RGBA", (width, height))
    img.putdata([
        (
            (x * 255 // max(1, width - 1) + rnd.randrange(8)) & 255,
            (y * 255 // max(1, height - 1) + rnd.randrange(8)) & 255,
            ((x + y) * 255 // max(1, width + height - 2) + rnd.randrange(8)) & 255,
            (255 if not alpha else (x * 255 // max(1, width - 1)) // 2 + 100),
        )
        for y in range(height) for x in range(width)
    ])
    buf = io.BytesIO()
    img.save(buf, "PNG", pnginfo=extra)
    return buf.getvalue()


def png_pixels(png):
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    return img.size, img.tobytes()


def mean_abs_diff(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / max(1, len(a))


def ani_bytes(name=b"walk", size=None, body=b"\x01\x02\x03\x04" * 4):
    """A valid .ani: magic | name (24) | type | size | body."""
    total = 36 + len(body)
    head = b"ANP3" + name.ljust(24, b"\0") + struct.pack("<I", 0x126)
    head += struct.pack("<I", total - 8 if size is None else size)
    return head + body


def ifp_bytes(name=b"walk", size=None, body=b"\x01\x02\x03\x04" * 4):
    """A valid .ifp: magic | size | name (24) | type | body."""
    total = 36 + len(body)
    return (b"ANP3" + struct.pack("<I", total - 8 if size is None else size)
            + name.ljust(24, b"\0") + struct.pack("<I", 0x126) + body)


def col_model(name=b"car", spheres=0, boxes=0, verts=0, faces=0, bounds=None, trailing=b""):
    """One COL1 model. Geometry arrays are filled with zeros."""
    bounds = bounds or (2.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0, 1.0, 2.0, 3.0)
    body = name.ljust(22, b"\0") + struct.pack("<H", 0) + struct.pack("<10f", *bounds)
    for count, size in ((spheres, 20), (0, 0), (boxes, 28), (verts, 12), (faces, 16)):
        body += struct.pack("<I", count) + b"\0" * (count * size)
    body += trailing
    return b"COLL" + struct.pack("<I", len(body)) + body


def cls_record(name=b"car", kind=b"CED2", tail=None, size=None, tag=b"CLST",
               bounds=(1, 2, 3, 4, -1, -2, -3, 1, 2, 3)):
    """One CLS record. bounds: center xyz, radius, low xyz, high xyz."""
    body = name.ljust(20, b"\0") + kind + struct.pack("<10f", *bounds)
    body += tail if tail is not None else b"\0" * 48
    return tag + struct.pack("<I", len(body) if size is None else size) + body


class TmpCase(unittest.TestCase):
    """A fresh temporary folder per test."""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = td.name

    def path(self, *parts):
        return os.path.join(self.tmp, *parts)

    def put(self, rel, data=b""):
        full = self.path(*rel.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(data)
        return full

    def read(self, rel):
        with open(self.path(*rel.split("/")), "rb") as f:
            return f.read()

    def exists(self, rel):
        return os.path.exists(self.path(*rel.split("/")))


# ------------------------------------------------------------------ common

def upper(data, pool=None):
    if data == b"bad":
        raise ValueError("boom")
    return data.upper()


def upper_note(data, pool=None):
    return data.upper(), "noted"


def silent_fail(data, pool=None):
    raise RuntimeError()


RULE = (".x", ".y", upper)


class TestCpusExpandBands(TmpCase):
    def test_cpus_uses_affinity(self):
        with mock.patch.object(os, "sched_getaffinity", create=True, return_value={0, 1, 2}):
            self.assertEqual(common.cpus(), 3)

    def test_cpus_falls_back_to_cpu_count(self):
        for exc in (AttributeError, OSError):
            with mock.patch.object(os, "sched_getaffinity", create=True, side_effect=exc), \
                    mock.patch.object(os, "cpu_count", return_value=6):
                self.assertEqual(common.cpus(), 6)

    def test_cpus_unknown_is_one(self):
        with mock.patch.object(os, "sched_getaffinity", create=True, side_effect=OSError), \
                mock.patch.object(os, "cpu_count", return_value=None):
            self.assertEqual(common.cpus(), 1)

    def test_expand_glob_sorted(self):
        b, a = self.put("b.txt"), self.put("a.txt")
        self.assertEqual(common.expand([self.path("*.txt")]), [a, b])
        self.assertEqual(common.expand([a]), [a])

    def test_expand_keeps_unmatched_and_existing(self):
        pattern = self.path("nothing*.txt")
        self.assertEqual(common.expand([pattern]), [pattern])
        literal = self.put("odd[1].txt")  # exists, so it is not a pattern
        self.assertEqual(common.expand([literal]), [literal])
        self.assertEqual(common.expand([self.path("missing.txt")]), [self.path("missing.txt")])

    def test_bands(self):
        self.assertEqual(common.bands(10, 3), [(0, 4), (4, 8), (8, 10)])
        self.assertEqual(common.bands(3, 10), [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(common.bands(5, 1), [(0, 5)])
        self.assertEqual(common.bands(4, 0), [(0, 4)])


class TestAtomicWrite(TmpCase):
    def test_atomic_success_and_creates_folder(self):
        target = self.path("new", "sub", "f.bin")
        with common.atomic(target) as tmp:
            self.assertEqual(os.path.dirname(tmp), os.path.dirname(target))
            with open(tmp, "wb") as f:
                f.write(b"hi")
            self.assertFalse(os.path.exists(target))
        self.assertEqual(self.read("new/sub/f.bin"), b"hi")
        self.assertEqual(os.listdir(os.path.dirname(target)), ["f.bin"])

    def test_atomic_failure_cleans_tmp_and_keeps_old(self):
        target = self.put("f.bin", b"old")
        with self.assertRaises(RuntimeError):
            with common.atomic(target) as tmp:
                with open(tmp, "wb") as f:
                    f.write(b"partial")
                raise RuntimeError("x")
        self.assertEqual(self.read("f.bin"), b"old")
        self.assertEqual(os.listdir(self.tmp), ["f.bin"])

    def test_atomic_failure_without_tmp_file(self):
        with self.assertRaises(KeyError):
            with common.atomic(self.path("f.bin")):
                raise KeyError("x")
        self.assertEqual(os.listdir(self.tmp), [])

    def test_write(self):
        common.write(self.path("w", "f.bin"), b"data")
        self.assertEqual(self.read("w/f.bin"), b"data")
        common.write(self.path("w", "f.bin"), b"newer")
        self.assertEqual(self.read("w/f.bin"), b"newer")


class TestPool(unittest.TestCase):
    TASKS = [(2, 3), (3, 2), (5, 0), (7, 2)]
    WANT = [8, 9, 1, 49]

    def test_serial_pool(self):
        with Pool(1) as pool:
            self.assertEqual(list(pool.map(pow, self.TASKS)), self.WANT)
            self.assertIsNone(pool.ex)

    def test_default_jobs(self):
        with mock.patch.object(common, "cpus", return_value=64):
            self.assertEqual(Pool().jobs, 8)
        with mock.patch.object(common, "cpus", return_value=1):
            self.assertEqual(Pool(0).jobs, 1)
        self.assertEqual(Pool(3).jobs, 3)

    def test_threads_keep_order(self):
        with Pool(3) as pool:
            self.assertEqual(list(pool.map(pow, self.TASKS)), self.WANT)
            self.assertIsInstance(pool.ex, ThreadPoolExecutor)

    def test_small_window(self):
        with Pool(2) as pool:
            self.assertEqual(list(pool.map(pow, self.TASKS, window=1)), self.WANT)

    def test_start_twice_and_empty_map(self):
        with Pool(2) as pool:
            pool.start()
            ex = pool.ex
            pool.start()
            self.assertIs(pool.ex, ex)
            self.assertEqual(list(pool.map(pow, [])), [])

    def test_env_is_exported_to_workers(self):
        with mock.patch.dict(os.environ):
            with Pool(2, env={"COMMON_TEST_VAR": "42"}) as pool:
                pool.start()
                self.assertEqual(os.environ["COMMON_TEST_VAR"], "42")

    def test_process_pool(self):
        with Pool(2, procs=True) as pool:
            self.assertEqual(list(pool.map(pow, self.TASKS)), self.WANT)
            self.assertIn(type(pool.ex), (ProcessPoolExecutor, ThreadPoolExecutor))

    def test_process_pool_falls_back_to_threads(self):
        with mock.patch.object(common, "ProcessPoolExecutor", side_effect=OSError("no fork")):
            with Pool(2, procs=True) as pool:
                self.assertEqual(list(pool.map(pow, self.TASKS)), self.WANT)
                self.assertIsInstance(pool.ex, ThreadPoolExecutor)

    def test_submit_failure_runs_inline(self):
        pool = Pool(2)
        pool.start()
        pool.ex.shutdown()
        broken = mock.Mock()
        broken.submit.side_effect = RuntimeError("shut down")
        pool.ex = broken
        self.assertEqual(list(pool.map(pow, self.TASKS)), self.WANT)
        self.assertIsNone(pool.ex)

    def test_broken_executor_runs_inline(self):
        fut = Future()
        fut.set_exception(BrokenExecutor("worker died"))
        ex = mock.Mock()
        ex.submit.return_value = fut
        pool = Pool(2)
        pool.start()
        pool.ex.shutdown()
        pool.ex = ex
        self.assertEqual(list(pool.map(pow, self.TASKS)), self.WANT)
        self.assertIsNone(pool.ex)

    def test_worker_exception_propagates(self):
        with Pool(2) as pool:
            with self.assertRaises(ZeroDivisionError):
                list(pool.map(pow, [(0, -1)]))

    def test_close_is_idempotent(self):
        pool = Pool(2)
        pool.close()
        pool.start()
        pool.close()
        pool.close()
        self.assertIsNone(pool.ex)


class TestInferWalk(TmpCase):
    def test_infer(self):
        exts = (".ani", ".ifp")
        self.assertEqual(common.infer(["a.ani", "b.ANI"], exts), ".ani")
        self.assertEqual(common.infer(["a.ifp"], exts), ".ifp")
        self.assertIsNone(common.infer(["a.ani", "b.ifp"], exts))
        self.assertIsNone(common.infer(["a.txt"], exts))
        self.assertIsNone(common.infer(["a.ani", "pack.zip"], exts))
        self.assertIsNone(common.infer(["a.ani", "pack.BPC"], exts))
        self.assertIsNone(common.infer([self.tmp], exts))

    def test_walk_sorted_any_depth(self):
        top, deep, other = self.put("z.txt"), self.put("a/x.TXT"), self.put("b/c/y.txt")
        self.put("a/skip.bin")
        self.assertEqual(list(common.walk(self.tmp, (".txt",))), [top, deep, other])

    def test_walk_skips_folder(self):
        top, other = self.put("z.txt"), self.put("b/y.txt")
        self.put("a/inner.txt")
        skip = os.path.realpath(self.path("a"))
        self.assertEqual(list(common.walk(self.tmp, (".txt",), skip)), [top, other])


class TestConvertHelpers(TmpCase):
    def test_call(self):
        self.assertEqual(common.call(upper, b"ab"), (True, b"AB", ""))
        self.assertEqual(common.call(upper_note, b"ab"), (True, b"AB", "noted"))
        self.assertEqual(common.call(upper, b"bad"), (False, "boom", ""))
        self.assertEqual(common.call(silent_fail, b"x"), (False, "RuntimeError", ""))

    def test_convert_file(self):
        src = self.put("a.x", b"abc")
        ok, res, note = common.convert_file(upper_note, src, self.path("o", "a.y"))
        self.assertEqual((ok, res, note), (True, "", "noted"))
        self.assertEqual(self.read("o/a.y"), b"ABC")

    def test_convert_file_conversion_error(self):
        src = self.put("a.x", b"bad")
        self.assertEqual(common.convert_file(upper, src, self.path("a.y")), (False, "boom", ""))
        self.assertFalse(self.exists("a.y"))

    def test_convert_file_io_errors(self):
        ok, msg, note = common.convert_file(upper, self.path("missing.x"), self.path("a.y"))
        self.assertFalse(ok)
        self.assertTrue(msg)
        src = self.put("a.x", b"abc")
        blocker = self.put("file", b"")
        ok, msg, note = common.convert_file(upper, src, os.path.join(blocker, "a.y"))
        self.assertFalse(ok)


class TestEditZip(TmpCase):
    def build(self):
        inner = make_zip({"n.x": b"inner", "m.txt": b"keep"})
        plain = make_zip({"only.txt": b"nothing here"})
        return make_zip({
            "dir/": b"",
            "one.x": b"one",
            "two.X": b"two",
            "bad.x": b"bad",
            "keep.txt": b"keep",
            "skip.x": b"old",
            "skip.y": b"already",
            "inner.zip": inner,
            "plain.bpc": plain,
            "broken.zip": b"this is not a zip",
        })

    def check(self, pool, rule=RULE):
        new, log = common.edit_zip(self.build(), rule, pool, "p/")
        with zipfile.ZipFile(io.BytesIO(new)) as z:
            names = z.namelist()
            self.assertEqual(names, ["dir/", "one.y", "two.y", "bad.x", "keep.txt", "skip.x",
                                     "skip.y", "inner.zip", "plain.bpc", "broken.zip"])
            self.assertEqual(z.read("one.y"), b"ONE")
            self.assertEqual(z.read("two.y"), b"TWO")
            self.assertEqual(z.read("bad.x"), b"bad")
            self.assertEqual(z.read("keep.txt"), b"keep")
            self.assertEqual(z.read("skip.x"), b"old")
            self.assertEqual(z.read("broken.zip"), b"this is not a zip")
            with zipfile.ZipFile(io.BytesIO(z.read("inner.zip"))) as inner:
                self.assertEqual(inner.namelist(), ["n.y", "m.txt"])
                self.assertEqual(inner.read("n.y"), b"INNER")
            self.assertEqual(z.read("plain.bpc"), make_zip({"only.txt": b"nothing here"}))
        return log

    def test_serial(self):
        log = self.check(Pool(1))
        self.assertIn(("ok", "p/one.x -> one.y"), log)
        self.assertIn(("ok", "p/two.X -> two.y"), log)
        self.assertIn(("fail", "p/bad.x: boom"), log)
        self.assertIn(("skip", "p/skip.x: skip.y already there"), log)
        self.assertIn(("ok", "p/inner.zip/n.x -> n.y"), log)
        self.assertTrue(any(s == "fail" and "broken.zip" in t for s, t in log))

    def test_threaded(self):
        with Pool(2) as pool:
            log = self.check(pool)
        self.assertEqual(sorted(s for s, _ in log).count("ok"), 3)

    def test_note_in_log(self):
        _, log = common.edit_zip(make_zip({"a.x": b"a"}), (".x", ".y", upper_note), Pool(1))
        self.assertEqual(log, [("ok", "a.x -> a.y (noted)")])

    def test_metadata_is_kept(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            info = zipfile.ZipInfo("a.x", (2020, 5, 6, 7, 8, 10))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            info.comment = b"hello"
            z.writestr(info, b"a")
        new, _ = common.edit_zip(buf.getvalue(), RULE, Pool(1))
        with zipfile.ZipFile(io.BytesIO(new)) as z:
            got = z.getinfo("a.y")
        self.assertEqual(got.date_time, (2020, 5, 6, 7, 8, 10))
        self.assertEqual(got.compress_type, zipfile.ZIP_STORED)
        self.assertEqual(got.external_attr, 0o600 << 16)
        self.assertEqual(got.comment, b"hello")


class TestConvertZip(TmpCase):
    def test_in_place_keeps_backup(self):
        raw = make_zip({"a.x": b"a", "b.txt": b"b"})
        src = self.put("pack.zip", raw)
        log = common.convert_zip(RULE, src, src, Pool(1))
        self.assertEqual([s for s, _ in log], ["ok"])
        self.assertEqual(self.read("pack.zip.bak"), raw)
        with zipfile.ZipFile(self.path("pack.zip")) as z:
            self.assertEqual(z.namelist(), ["a.y", "b.txt"])

    def test_in_place_nothing_to_do(self):
        raw = make_zip({"b.txt": b"b"})
        src = self.put("pack.zip", raw)
        log = common.convert_zip(RULE, src, src, Pool(1))
        self.assertEqual(log, [("skip", f"{src}: nothing to convert")])
        self.assertEqual(self.read("pack.zip"), raw)
        self.assertFalse(self.exists("pack.zip.bak"))

    def test_other_destination(self):
        src = self.put("pack.zip", make_zip({"b.txt": b"b"}))
        dst = self.path("out", "copy.zip")
        self.assertEqual(common.convert_zip(RULE, src, dst, Pool(1)), [])
        with zipfile.ZipFile(dst) as z:
            self.assertEqual(z.namelist(), ["b.txt"])
        src2 = self.put("two.zip", make_zip({"a.x": b"a"}))
        dst2 = self.path("out", "two.zip")
        common.convert_zip(RULE, src2, dst2, Pool(1))
        self.assertFalse(self.exists("two.zip.bak"))
        self.assertFalse(self.exists("out/two.zip.bak"))

    def test_not_a_zip(self):
        zip_path = self.put("bad.zip", b"garbage")
        (status, text), = common.convert_zip(RULE, zip_path, zip_path, Pool(1))
        self.assertEqual(status, "fail")
        self.assertIn("not a zip", text)
        self.assertNotIn("encrypted", text)
        bpc_path = self.put("bad.bpc", b"garbage")
        (status, text), = common.convert_zip(RULE, bpc_path, bpc_path, Pool(1))
        self.assertIn("encrypted? bpc.py decrypts it", text)


class TestPlan(TmpCase):
    def test_single_file(self):
        f = self.put("a.x")
        self.assertEqual(common.plan([f], RULE, None, True), ([(f, self.path("a.y"))], [], []))
        self.assertEqual(common.plan([f], RULE, self.path("o.bin"), True),
                         ([(f, self.path("o.bin"))], [], []))

    def test_single_archive(self):
        z = self.put("a.zip")
        self.assertEqual(common.plan([z], RULE, None, True), ([], [(z, z)], []))
        self.assertEqual(common.plan([z], RULE, self.path("o.zip"), True),
                         ([], [(z, self.path("o.zip"))], []))

    def test_folder_in_place(self):
        a, b, z = self.put("d/a.x"), self.put("d/s/b.X"), self.put("d/p.bpc")
        self.put("d/other.txt")
        files, zips, errors = common.plan([self.path("d")], RULE, None, True)
        self.assertEqual(files, [(a, self.path("d", "a.y")), (b, self.path("d", "s", "b.y"))])
        self.assertEqual(zips, [(z, z)])
        self.assertEqual(errors, [])

    def test_folder_with_out(self):
        a, z = self.put("d/s/a.x"), self.put("d/p.zip")
        files, zips, errors = common.plan([self.path("d")], RULE, self.path("out"), True)
        self.assertEqual(files, [(a, self.path("out", "s", "a.y"))])
        self.assertEqual(zips, [(z, self.path("out", "p.zip"))])

    def test_two_inputs_with_out(self):
        a, f = self.put("d/a.x"), self.put("e.x")
        files, zips, errors = common.plan([self.path("d"), f], RULE, self.path("out"), True)
        self.assertEqual(files, [(a, self.path("out", "d", "a.y")), (f, self.path("out", "e.y"))])

    def test_out_inside_source_is_skipped(self):
        a = self.put("d/a.x")
        self.put("d/out/old.x")
        files, _, _ = common.plan([self.path("d")], RULE, self.path("d", "out"), True)
        self.assertEqual(files, [(a, self.path("d", "out", "a.y"))])

    def test_errors(self):
        self.put("d/a.x")
        wrong = self.put("w.txt")
        missing = self.path("missing")
        files, zips, errors = common.plan([wrong, missing, self.path("d")], RULE, None, True)
        self.assertEqual(len(files), 1)
        self.assertEqual(errors, [f"{wrong}: expected a .x file", f"{missing}: no such file or directory"])

    def test_archives_off(self):
        z = self.put("d/p.zip")
        self.put("d/a.x")
        files, zips, errors = common.plan([self.path("d"), z], RULE, None, False)
        self.assertEqual(len(files), 1)
        self.assertEqual(zips, [])
        self.assertEqual(errors, [f"{z}: expected a .x file"])


class TestRun(TmpCase):
    def test_single_file(self):
        f = self.put("a.x", b"abc")
        code, out, err = capture(common.run, [f], RULE, jobs=1)
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), f"{f} -> {self.path('a.y')}")
        self.assertEqual(err, "")
        self.assertEqual(self.read("a.y"), b"ABC")

    def test_single_file_note(self):
        f = self.put("a.x", b"abc")
        _, out, _ = capture(common.run, [f], (".x", ".y", upper_note), jobs=1)
        self.assertIn("(noted)", out)

    def test_many_files_with_failure_and_summary(self):
        a, b = self.put("d/a.x", b"abc"), self.put("d/b.x", b"bad")
        self.put("d/c.x", b"zzz")
        code, out, err = capture(common.run, [self.path("d")], RULE, jobs=2)
        self.assertEqual(code, 1)
        self.assertIn("2 converted, 1 failed, 0 skipped", out)
        self.assertIn(f"{b}: boom", err)
        self.assertEqual(self.read("d/a.y"), b"ABC")
        self.assertFalse(self.exists("d/b.y"))

    def test_nothing_to_convert(self):
        self.put("d/a.txt")
        code, out, err = capture(common.run, [self.path("d")], RULE)
        self.assertEqual(code, 1)
        self.assertIn("nothing to convert: no .x files found", err)

    def test_only_errors(self):
        code, out, err = capture(common.run, [self.path("missing")], RULE)
        self.assertEqual(code, 1)
        self.assertIn("error: ", err)
        self.assertNotIn("nothing to convert", err)

    def test_wildcards_and_zips(self):
        self.put("d/a.x", b"a")
        self.put("d/p.zip", make_zip({"in.x": b"in", "keep.txt": b"k"}))
        self.put("d/q.zip", make_zip({"keep.txt": b"k"}))
        code, out, err = capture(common.run, [self.path("d", "*")], RULE, jobs=1)
        self.assertEqual(code, 0)
        self.assertIn("2 converted, 0 failed, 1 skipped", out)
        self.assertIn("nothing to convert", out)
        self.assertTrue(self.exists("d/p.zip.bak"))

    def test_zip_only(self):
        self.put("p.zip", make_zip({"in.x": b"in"}))
        code, out, _ = capture(common.run, [self.path("p.zip")], RULE, jobs=1)
        self.assertEqual(code, 0)
        self.assertIn("p.zip/in.x -> in.y", out)

    def test_single_file_plus_zip_uses_pool_map(self):
        self.put("a.x", b"a")
        self.put("p.zip", make_zip({"in.x": b"in"}))
        code, out, _ = capture(common.run, [self.path("a.x"), self.path("p.zip")], RULE, jobs=2)
        self.assertEqual(code, 0)
        self.assertIn("2 converted", out)

    def test_archives_off(self):
        self.put("d/p.zip", make_zip({"in.x": b"in"}))
        code, _, err = capture(common.run, [self.path("d")], RULE, archives=False)
        self.assertEqual(code, 1)
        self.assertIn("nothing to convert", err)


# --------------------------------------------------------------------- arr

class TestArr(unittest.TestCase):
    def load(self, **env):
        with mock.patch.dict(os.environ, env):
            return arr.load()

    def test_module_exposes_numpy(self):
        import numpy
        self.assertIs(arr.np, numpy)

    def test_forced_numpy(self):
        import numpy
        self.assertIs(self.load(BTX_ARRAY_BACKEND="numpy"), numpy)
        self.assertIs(self.load(BTX_ARRAY_BACKEND=" NumPy "), numpy)

    def test_bad_value(self):
        with self.assertRaises(SystemExit) as cm:
            self.load(BTX_ARRAY_BACKEND="cupy")
        self.assertIn("must be auto, rsnumpy or numpy", str(cm.exception))

    def test_rsnumpy_missing(self):
        import numpy
        with mock.patch.dict(sys.modules, {"rsnumpy": None}):
            with self.assertRaises(SystemExit) as cm:
                self.load(BTX_ARRAY_BACKEND="rsnumpy")
            self.assertIn("rsnumpy is not installed", str(cm.exception))
            self.assertIs(self.load(BTX_ARRAY_BACKEND="auto"), numpy)

    def test_rsnumpy_used_when_present(self):
        fake = mock.Mock()
        with mock.patch.dict(sys.modules, {"rsnumpy": fake}):
            self.assertIs(self.load(BTX_ARRAY_BACKEND="auto", BTX_ARRAY_THREADS="3"), fake)
            fake.set_num_threads.assert_called_with(3)
            self.assertIs(self.load(BTX_ARRAY_BACKEND="rsnumpy", BTX_ARRAY_THREADS=""), fake)
            fake.set_num_threads.assert_called_with(os.cpu_count() or 1)

    def test_rsnumpy_thread_setting_failure_is_ignored(self):
        fake = mock.Mock()
        fake.set_num_threads.side_effect = RuntimeError("nope")
        with mock.patch.dict(sys.modules, {"rsnumpy": fake}):
            self.assertIs(self.load(BTX_ARRAY_BACKEND="auto"), fake)

    def test_no_array_library_at_all(self):
        with mock.patch.dict(sys.modules, {"rsnumpy": None, "numpy": None}):
            with self.assertRaises(SystemExit) as cm:
                self.load(BTX_ARRAY_BACKEND="auto")
            self.assertIn("need numpy", str(cm.exception))


# --------------------------------------------------------------------- ani

class TestAniConvert(unittest.TestCase):
    def test_ani_to_ifp_moves_fields(self):
        data = ani_bytes(b"walk", body=b"BODY" * 5)
        out = ani.to_ifp(data)
        self.assertEqual(out, ifp_bytes(b"walk", body=b"BODY" * 5))
        self.assertEqual(out[:4], b"ANP3")
        self.assertEqual(struct.unpack_from("<I", out, 4)[0], len(out) - 8)
        self.assertEqual(out[8:12], b"walk")
        self.assertEqual(struct.unpack_from("<I", out, 32)[0], 0x126)

    def test_ifp_to_ani_moves_fields(self):
        data = ifp_bytes(b"run", body=b"xyz!" * 3)
        self.assertEqual(ani.to_ani(data), ani_bytes(b"run", body=b"xyz!" * 3))

    def test_roundtrip_short_name(self):
        data = ani_bytes(b"idle")
        self.assertEqual(ani.to_ani(ani.to_ifp(data)), data)

    def test_last_four_name_bytes_do_not_survive(self):
        data = ani_bytes(b"A" * 24)
        ifp = ani.to_ifp(data)
        self.assertEqual(ifp[8:28], b"A" * 20)
        self.assertEqual(ifp[28:32], bytes(4))
        self.assertEqual(ani.to_ani(ifp)[4:28], b"A" * 20 + bytes(4))

    def test_header_only_file(self):
        data = ani_bytes(b"x", body=b"")
        self.assertEqual(ani.to_ani(ani.to_ifp(data)), data)

    def test_pool_argument_is_ignored(self):
        data = ani_bytes()
        self.assertEqual(ani.to_ifp(data, object()), ani.to_ifp(data))
        self.assertEqual(ani.to_ani(ifp_bytes(), object()), ani.to_ani(ifp_bytes()))

    def test_errors(self):
        good = ani_bytes()
        cases = [
            (ani.to_ifp, good[:35], "too small"),
            (ani.to_ifp, b"", "too small"),
            (ani.to_ifp, b"XXXX" + good[4:], "not an ANP3 file"),
            (ani.to_ifp, good[:28] + struct.pack("<I", 0x127) + good[32:], "not an ANI file"),
            (ani.to_ifp, ifp_bytes(), "not an ANI file"),
            (ani.to_ifp, ani_bytes(size=5), "header says 5"),
            (ani.to_ifp, good + b"\0", "wrong size field"),
            (ani.to_ani, ifp_bytes()[:20], "too small"),
            (ani.to_ani, b"XXXX" + ifp_bytes()[4:], "not an ANP3 file"),
            (ani.to_ani, ani_bytes(), "not an IFP file"),
            (ani.to_ani, ifp_bytes(size=9999), "header says 9999"),
        ]
        for fn, data, text in cases:
            with self.subTest(fn=fn.__name__, text=text):
                with self.assertRaises(ValueError) as cm:
                    fn(data)
                self.assertIn(text, str(cm.exception))


class TestAniMain(TmpCase):
    def test_file_ani_to_ifp(self):
        src = self.put("walk.ani", ani_bytes(b"walk"))
        code, out, err = run_main(ani, src)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("walk.ifp", out)
        self.assertEqual(self.read("walk.ifp"), ifp_bytes(b"walk"))

    def test_file_ifp_to_ani(self):
        src = self.put("walk.ifp", ifp_bytes(b"walk"))
        code, _, _ = run_main(ani, src)
        self.assertEqual(code, 0)
        self.assertEqual(self.read("walk.ani"), ani_bytes(b"walk"))

    def test_output_path(self):
        src = self.put("walk.ani", ani_bytes())
        self.assertEqual(run_main(ani, src, "-o", self.path("o", "x.bin"))[0], 0)
        self.assertEqual(self.read("o/x.bin"), ifp_bytes())

    def test_several_files_to_folder(self):
        a, b = self.put("a.ani", ani_bytes(b"a")), self.put("b.ani", ani_bytes(b"b"))
        code, out, _ = run_main(ani, a, b, "-o", self.path("out"), "-j", 1)
        self.assertEqual(code, 0)
        self.assertEqual(self.read("out/a.ifp"), ifp_bytes(b"a"))
        self.assertEqual(self.read("out/b.ifp"), ifp_bytes(b"b"))
        self.assertIn("2 converted, 0 failed, 0 skipped", out)

    def test_wildcard(self):
        self.put("a.ani", ani_bytes(b"a"))
        self.put("b.ani", ani_bytes(b"b"))
        code, _, _ = run_main(ani, self.path("*.ani"))
        self.assertEqual(code, 0)
        self.assertTrue(self.exists("a.ifp") and self.exists("b.ifp"))

    def test_folder_needs_direction(self):
        self.put("d/a.ani", ani_bytes())
        code, _, err = run_main(ani, self.path("d"))
        self.assertEqual(code, 2)
        self.assertIn("say which way", err)

    def test_folder_with_direction(self):
        self.put("d/a.ani", ani_bytes(b"a"))
        self.put("d/deep/b.ani", ani_bytes(b"b"))
        self.put("d/skip.txt", b"x")
        code, _, _ = run_main(ani, self.path("d"), "--ifp", "-o", self.path("out"))
        self.assertEqual(code, 0)
        self.assertEqual(self.read("out/a.ifp"), ifp_bytes(b"a"))
        self.assertEqual(self.read("out/deep/b.ifp"), ifp_bytes(b"b"))

    def test_folder_ifp_to_ani_direction(self):
        self.put("d/a.ifp", ifp_bytes(b"a"))
        self.assertEqual(run_main(ani, self.path("d"), "--ani")[0], 0)
        self.assertEqual(self.read("d/a.ani"), ani_bytes(b"a"))

    def test_mixed_extensions_need_direction(self):
        a, b = self.put("a.ani", ani_bytes()), self.put("b.ifp", ifp_bytes())
        code, _, err = run_main(ani, a, b)
        self.assertEqual(code, 2)
        self.assertIn("say which way", err)

    def test_both_directions_rejected(self):
        code, _, err = run_main(ani, "x.ani", "--ifp", "--ani")
        self.assertEqual(code, 2)
        self.assertIn("not allowed", err)

    def test_broken_file_reported(self):
        good = self.put("a.ani", ani_bytes())
        bad = self.put("b.ani", b"ANP3 broken")
        code, out, err = run_main(ani, good, bad, "-j", 1)
        self.assertEqual(code, 1)
        self.assertIn("too small", err)
        self.assertIn("1 converted, 1 failed", out)

    def test_zip_is_not_opened(self):
        self.put("d/p.zip", make_zip({"a.ani": ani_bytes()}))
        code, _, err = run_main(ani, self.path("d"), "--ifp")
        self.assertEqual(code, 1)
        self.assertIn("nothing to convert", err)

    def test_script_entry_point(self):
        src = self.put("walk.ani", ani_bytes())
        code, _, _ = run_script("ani.py", src)
        self.assertEqual(code, 0)
        self.assertTrue(self.exists("walk.ifp"))


# --------------------------------------------------------------------- bpc

def make_bpc(entries, compress=zipfile.ZIP_DEFLATED):
    return naive_xor(make_zip(entries, compress))


def noisy(n, seed=0):
    rnd = random.Random(seed)
    return bytes(rnd.randrange(256) for _ in range(n))


def corrupt_zip_crc(raw):
    """Flip one byte of the first stored entry's data: the CRC no longer matches."""
    pos = raw.index(b"PK\x03\x04")
    name_len, extra_len = struct.unpack_from("<HH", raw, pos + 26)
    at = pos + 30 + name_len + extra_len
    return raw[:at] + bytes([raw[at] ^ 0xFF]) + raw[at + 1:]


def corrupt_zip_deflate(raw):
    """Make the first deflate stream start with an invalid block type."""
    pos = raw.index(b"PK\x03\x04")
    name_len, extra_len = struct.unpack_from("<HH", raw, pos + 26)
    at = pos + 30 + name_len + extra_len
    return raw[:at] + b"\xff" + raw[at + 1:]


class TestXor(unittest.TestCase):
    def tearDown(self):
        bpc._np.clear()

    def test_matches_reference_for_all_alignments(self):
        data = bytes(range(256)) * 3
        for n in (0, 1, 19, 20, 21, 40, 100, 768):
            for offset in (0, 1, 19, 20, 21, 1234):
                with self.subTest(n=n, offset=offset):
                    self.assertEqual(bpc.xor(data[:n], offset), naive_xor(data[:n], offset))

    def test_is_its_own_inverse(self):
        data = noisy(1000)
        self.assertEqual(bpc.xor(bpc.xor(data, 7), 7), data)

    def test_accepts_bytearray_and_memoryview(self):
        data = noisy(50)
        self.assertEqual(bpc.xor(bytearray(data)), naive_xor(data))
        self.assertEqual(bpc.xor(memoryview(data), 3), naive_xor(data, 3))

    def test_big_buffer_uses_numpy(self):
        data = bytes((i * 7 + 3) & 255 for i in range((1 << 20) + 7))
        self.assertIsNotNone(bpc.numpy())
        self.assertEqual(bpc.xor(data, 5), naive_xor(data, 5))

    def test_big_buffer_without_numpy(self):
        data = bytes((i * 11 + 1) & 255 for i in range((1 << 20) + 3))
        bpc._np.clear()
        with mock.patch.dict(sys.modules, {"numpy": None}):
            self.assertIsNone(bpc.numpy())
            self.assertIsNone(bpc.numpy())  # cached
            self.assertEqual(bpc.xor(data, 9), naive_xor(data, 9))

    def test_numpy_is_cached(self):
        first = bpc.numpy()
        self.assertIs(bpc.numpy(), first)


class TestXorStreams(TmpCase):
    def test_writer_encrypts_and_reader_decrypts(self):
        data = noisy(777)
        path = self.path("f.bpc")
        with open(path, "wb") as f:
            w = bpc.XorWriter(f)
            self.assertTrue(w.writable() and w.seekable())
            self.assertEqual(w.write(data[:300]), 300)
            self.assertEqual(w.tell(), 300)
            self.assertEqual(w.write(data[300:]), 477)
            w.flush()
            w.seek(0)
            self.assertEqual(w.tell(), 0)
            w.seek(0, os.SEEK_END)
            w.close()
            w.flush()  # closed: must not raise
        self.assertEqual(self.read("f.bpc"), naive_xor(data))
        with bpc.open_bpc(path) as r:
            self.assertEqual(r.read(), data)

    def test_reader_seek_tell_partial(self):
        data = noisy(200)
        path = self.put("f.bpc", naive_xor(data))
        raw = bpc.XorReader(open(path, "rb"))
        self.assertTrue(raw.readable() and raw.seekable())
        raw.seek(25)
        self.assertEqual(raw.tell(), 25)
        buf = bytearray(10)
        self.assertEqual(raw.readinto(buf), 10)
        self.assertEqual(bytes(buf), data[25:35])
        raw.seek(-5, os.SEEK_END)
        buf = bytearray(10)
        self.assertEqual(raw.readinto(buf), 5)
        self.assertEqual(bytes(buf[:5]), data[-5:])
        raw.close()
        raw.close()  # twice is fine
        self.assertTrue(raw.closed)

    def test_open_zip_is_plain(self):
        path = self.put("f.zip", b"plain")
        with bpc.open_zip(path) as f:
            self.assertEqual(f.read(), b"plain")

    def test_big_stream_through_buffered_reader(self):
        data = bytes((i * 13 + 5) & 255 for i in range(2 * (1 << 20) + 11))
        path = self.put("big.bpc", naive_xor(data))
        with bpc.open_bpc(path) as r:
            self.assertEqual(r.read(), data)


class TestCrypt(TmpCase):
    def test_roundtrip_with_several_chunks(self):
        data = bytes((i * 3 + 1) & 255 for i in range(2 * bpc.CHUNK + 123))
        src = self.put("a.zip", data)
        enc, dec = self.path("a.bpc"), self.path("b.zip")
        bpc.crypt(src, enc)
        bpc.crypt(enc, dec)
        with open(enc, "rb") as f:
            blob = f.read()
        self.assertEqual(blob[bpc.CHUNK:bpc.CHUNK + 50], naive_xor(data[bpc.CHUNK:bpc.CHUNK + 50], bpc.CHUNK))
        self.assertEqual(self.read("b.zip"), data)

    def test_refuses_to_overwrite_input(self):
        src = self.put("a.zip", b"data")
        with self.assertRaises(ValueError) as cm:
            bpc.crypt(src, src)
        self.assertIn("output is the input file", str(cm.exception))
        self.assertEqual(self.read("a.zip"), b"data")

    def test_overwrites_other_existing_output(self):
        src = self.put("a.zip", b"data")
        dst = self.put("a.bpc", b"stale")
        bpc.crypt(src, dst)
        self.assertEqual(self.read("a.bpc"), naive_xor(b"data"))

    def test_check_can_veto(self):
        src = self.put("a.zip", b"data")
        seen = []

        def veto(tmp):
            seen.append(os.path.exists(tmp))
            raise ValueError("nope")

        with self.assertRaises(ValueError):
            bpc.crypt(src, self.path("a.bpc"), veto)
        self.assertEqual(seen, [True])
        self.assertEqual(sorted(os.listdir(self.tmp)), ["a.zip"])

    def test_check_passes(self):
        src = self.put("a.zip", b"data")
        bpc.crypt(src, self.path("a.bpc"), lambda tmp: None)
        self.assertTrue(self.exists("a.bpc"))


class TestValidate(TmpCase):
    def zip_path(self, entries, compress=zipfile.ZIP_DEFLATED, transform=None):
        raw = make_zip(entries, compress)
        return self.put("t.zip", transform(raw) if transform else raw)

    def opener(self, path):
        return lambda: bpc.open_zip(path)

    def test_good_serial(self):
        path = self.zip_path({"a": noisy(100), "b": noisy(100, 1)})
        bpc.validate(self.opener(path))
        bpc.validate(self.opener(path), jobs=4)  # too small for threads, still serial

    def test_single_entry_stays_serial(self):
        path = self.zip_path({"a": noisy(100)})
        with mock.patch.object(bpc, "PAR_MIN", 1):
            bpc.validate(self.opener(path), jobs=4)

    def test_bad_crc_serial(self):
        path = self.zip_path({"a.bin": noisy(100)}, zipfile.ZIP_STORED, corrupt_zip_crc)
        with self.assertRaises(ValueError) as cm:
            bpc.validate(self.opener(path))
        self.assertIn("bad zip entry: a.bin", str(cm.exception))

    def test_bad_deflate_serial(self):
        path = self.zip_path({"a.bin": noisy(300)}, transform=corrupt_zip_deflate)
        with self.assertRaises(ValueError) as cm:
            bpc.validate(self.opener(path))
        self.assertIn("bad zip data", str(cm.exception))

    def test_parallel_good(self):
        entries = {f"f{i}.bin": noisy(500, i) for i in range(5)}
        path = self.zip_path(entries)
        with mock.patch.object(bpc, "PAR_MIN", 1):
            bpc.validate(self.opener(path), jobs=2)

    def test_parallel_bad_crc(self):
        path = self.zip_path({"a.bin": noisy(200), "b.bin": noisy(200, 1)},
                             zipfile.ZIP_STORED, corrupt_zip_crc)
        with mock.patch.object(bpc, "PAR_MIN", 1):
            with self.assertRaises(ValueError) as cm:
                bpc.validate(self.opener(path), jobs=2)
        self.assertIn("bad zip entry: a.bin", str(cm.exception))

    def test_parallel_bad_deflate(self):
        path = self.zip_path({"a.bin": noisy(300), "b.bin": noisy(300, 2)}, transform=corrupt_zip_deflate)
        with mock.patch.object(bpc, "PAR_MIN", 1):
            with self.assertRaises(ValueError) as cm:
                bpc.validate(self.opener(path), jobs=2)
        self.assertIn("bad zip entry", str(cm.exception))

    def test_not_a_zip(self):
        path = self.put("t.zip", b"not a zip at all")
        with self.assertRaises(zipfile.BadZipFile):
            bpc.validate(self.opener(path))


class TestUnpack(TmpCase):
    ENTRIES = {
        "a.txt": b"alpha",
        "sub/": b"",
        "sub/b.bin": noisy(300),
        "empty/": b"",
        "sub/deep/c.txt": b"gamma",
    }

    def test_unpack(self):
        src = self.put("p.bpc", make_bpc(self.ENTRIES))
        n = bpc.unpack(src, self.path("out"), jobs=1)
        self.assertEqual(n, 5)
        self.assertEqual(self.read("out/a.txt"), b"alpha")
        self.assertEqual(self.read("out/sub/b.bin"), self.ENTRIES["sub/b.bin"])
        self.assertEqual(self.read("out/sub/deep/c.txt"), b"gamma")
        self.assertTrue(os.path.isdir(self.path("out", "empty")))

    def test_unpack_without_check(self):
        src = self.put("p.bpc", make_bpc(self.ENTRIES))
        self.assertEqual(bpc.unpack(src, self.path("out"), jobs=1, check=False), 5)
        self.assertEqual(self.read("out/a.txt"), b"alpha")

    def test_check_catches_corruption_before_extracting(self):
        raw = corrupt_zip_crc(make_zip({"a.bin": noisy(100)}, zipfile.ZIP_STORED))
        src = self.put("p.bpc", naive_xor(raw))
        with self.assertRaises(ValueError):
            bpc.unpack(src, self.path("out"), jobs=1)
        self.assertFalse(self.exists("out"))

    def test_parallel_extract(self):
        entries = {f"d{i % 3}/f{i}.bin": noisy(400, i) for i in range(9)}
        src = self.put("p.bpc", make_bpc(entries))
        with mock.patch.object(bpc, "PAR_MIN", 1):
            bpc.unpack(src, self.path("out"), jobs=3)
        for name, data in entries.items():
            self.assertEqual(self.read("out/" + name), data)

    def test_duplicate_names_extract_serially(self):
        import warnings
        buf = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("same.txt", b"first")
                z.writestr("same.txt", b"second")
        src = self.put("p.bpc", naive_xor(buf.getvalue()))
        with mock.patch.object(bpc, "PAR_MIN", 1):
            self.assertEqual(bpc.unpack(src, self.path("out"), jobs=4), 2)
        self.assertEqual(self.read("out/same.txt"), b"second")

    def flaky_extract(self, fail_times):
        real = zipfile.ZipFile.extract
        state = {"left": fail_times}

        def extract(zf, member, path=None, pwd=None):
            name = getattr(member, "filename", member)
            if not name.endswith("/") and state["left"] != 0:
                state["left"] -= 1
                raise FileExistsError("another thread made the folder")
            return real(zf, member, path, pwd)

        return mock.patch.object(zipfile.ZipFile, "extract", extract)

    def test_retries_once_on_file_exists(self):
        src = self.put("p.bpc", make_bpc({"a.txt": b"alpha"}))
        with self.flaky_extract(1):
            bpc.unpack(src, self.path("out"), jobs=1)
        self.assertEqual(self.read("out/a.txt"), b"alpha")

    def test_gives_up_on_second_file_exists(self):
        src = self.put("p.bpc", make_bpc({"a.txt": b"alpha"}))
        with self.flaky_extract(-1):
            with self.assertRaises(FileExistsError):
                bpc.unpack(src, self.path("out"), jobs=1)


class TestPack(TmpCase):
    def tree(self):
        self.put("src/a.txt", b"alpha")
        self.put("src/sub/b.txt", b"beta")
        self.put("src/sub/deep/c.bin", b"gamma")
        os.makedirs(self.path("src", "empty"))
        return self.path("src")

    def zip_of(self, bpc_path):
        with open(bpc_path, "rb") as f:
            return zipfile.ZipFile(io.BytesIO(naive_xor(f.read())))

    def test_pack_layout(self):
        dst = self.path("o.bpc")
        bpc.pack(self.tree(), dst)
        with self.zip_of(dst) as z:
            self.assertEqual(z.namelist(), ["a.txt", "empty/", "sub/", "sub/b.txt", "sub/deep/", "sub/deep/c.bin"])
            self.assertEqual(z.read("sub/deep/c.bin"), b"gamma")
            self.assertEqual(z.getinfo("a.txt").compress_type, zipfile.ZIP_DEFLATED)
            self.assertIsNone(z.testzip())

    def test_pack_meta_layout(self):
        dst = self.path("o.bpc")
        bpc.pack(self.tree(), dst, meta=True)
        with self.zip_of(dst) as z:
            self.assertEqual(z.namelist(), ["a.txt", "sub/b.txt", "sub/deep/c.bin"])
            self.assertTrue(all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist()))
            self.assertEqual(z.read("sub/b.txt"), b"beta")

    def test_pack_then_unpack(self):
        dst = self.path("o.bpc")
        bpc.pack(self.tree(), dst)
        bpc.unpack(dst, self.path("back"), jobs=1)
        for rel in ("a.txt", "sub/b.txt", "sub/deep/c.bin"):
            self.assertEqual(self.read("back/" + rel), self.read("src/" + rel))
        self.assertTrue(os.path.isdir(self.path("back", "empty")))

    def test_failed_pack_leaves_nothing(self):
        dst = self.path("o.bpc")
        with mock.patch.object(zipfile.ZipFile, "write", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                bpc.pack(self.tree(), dst)
        self.assertFalse(os.path.exists(dst))
        self.assertEqual(sorted(os.listdir(self.tmp)), ["src"])


class TestBpcMain(TmpCase):
    def test_folder_is_packed(self):
        self.put("pack/a.txt", b"alpha")
        code, out, err = run_main(bpc, self.path("pack"), "-j", 1)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("pack.bpc", out)
        self.assertEqual(naive_xor(self.read("pack.bpc"))[:2], b"PK")

    def test_folder_with_trailing_slash(self):
        self.put("pack/a.txt", b"alpha")
        self.assertEqual(run_main(bpc, self.path("pack") + os.sep)[0], 0)
        self.assertTrue(self.exists("pack.bpc"))

    def test_bpc_is_unpacked(self):
        self.put("pack.bpc", make_bpc({"a.txt": b"alpha", "s/b.txt": b"beta"}))
        code, _, _ = run_main(bpc, self.path("pack.bpc"), "-j", 1)
        self.assertEqual(code, 0)
        self.assertEqual(self.read("pack/a.txt"), b"alpha")
        self.assertEqual(self.read("pack/s/b.txt"), b"beta")

    def test_bpc_decrypted_to_zip(self):
        raw = make_zip({"a.txt": b"alpha"})
        self.put("pack.bpc", naive_xor(raw))
        code, _, _ = run_main(bpc, self.path("pack.bpc"), "-z")
        self.assertEqual(code, 0)
        self.assertEqual(self.read("pack.zip"), raw)

    def test_zip_is_encrypted(self):
        raw = make_zip({"a.txt": b"alpha"})
        self.put("pack.zip", raw)
        code, _, _ = run_main(bpc, self.path("pack.zip"))
        self.assertEqual(code, 0)
        self.assertEqual(self.read("pack.bpc"), naive_xor(raw))

    def test_meta_pack(self):
        self.put("pack/b.wav", b"bb")
        self.put("pack/a.wav", b"aa")
        self.assertEqual(run_main(bpc, self.path("pack"), "-m")[0], 0)
        with zipfile.ZipFile(io.BytesIO(naive_xor(self.read("pack.bpc")))) as z:
            self.assertEqual(z.namelist(), ["a.wav", "b.wav"])

    def test_output_option(self):
        self.put("pack/a.txt", b"x")
        self.assertEqual(run_main(bpc, self.path("pack"), "-o", self.path("elsewhere", "x.bpc"))[0], 0)
        self.assertTrue(self.exists("elsewhere/x.bpc"))

    def test_several_inputs_to_folder(self):
        self.put("p1/a.txt", b"1")
        self.put("p2/b.txt", b"2")
        code, _, _ = run_main(bpc, self.path("p1"), self.path("p2"), "-o", self.path("out"))
        self.assertEqual(code, 0)
        self.assertTrue(self.exists("out/p1.bpc") and self.exists("out/p2.bpc"))

    def test_wildcard_input(self):
        self.put("p1/a.txt", b"1")
        self.put("p2/a.txt", b"2")
        self.assertEqual(run_main(bpc, self.path("p*"))[0], 0)
        self.assertTrue(self.exists("p1.bpc") and self.exists("p2.bpc"))

    def test_corrupt_zip_is_refused_unless_fast(self):
        raw = corrupt_zip_crc(make_zip({"a.bin": noisy(100)}, zipfile.ZIP_STORED))
        self.put("bad.zip", raw)
        code, _, err = run_main(bpc, self.path("bad.zip"))
        self.assertEqual(code, 1)
        self.assertIn("bad zip entry", err)
        self.assertFalse(self.exists("bad.bpc"))
        code, _, err = run_main(bpc, self.path("bad.zip"), "--fast")
        self.assertEqual(code, 0)
        self.assertTrue(self.exists("bad.bpc"))

    def test_fast_skips_checks_when_decrypting(self):
        raw = corrupt_zip_crc(make_zip({"a.bin": noisy(100)}, zipfile.ZIP_STORED))
        self.put("bad.bpc", naive_xor(raw))
        self.assertEqual(run_main(bpc, self.path("bad.bpc"), "-z")[0], 1)
        self.assertEqual(run_main(bpc, self.path("bad.bpc"), "-z", "--fast")[0], 0)
        self.assertEqual(self.read("bad.zip"), raw)

    def test_fast_unpack(self):
        self.put("p.bpc", make_bpc({"a.txt": b"alpha"}))
        self.assertEqual(run_main(bpc, self.path("p.bpc"), "--fast", "-j", 2)[0], 0)
        self.assertEqual(self.read("p/a.txt"), b"alpha")

    def test_garbage_bpc(self):
        self.put("junk.bpc", b"definitely not a zip, and long enough to look real" * 3)
        code, _, err = run_main(bpc, self.path("junk.bpc"))
        self.assertEqual(code, 1)
        self.assertIn("junk.bpc", err)

    def test_missing_and_wrong_inputs(self):
        wrong = self.put("notes.txt", b"hello")
        good = self.put("p.zip", make_zip({"a.txt": b"a"}))
        code, out, err = run_main(bpc, self.path("missing"), wrong, good)
        self.assertEqual(code, 1)
        self.assertIn("no such file or directory", err)
        self.assertIn("expected a folder, a .bpc or a .zip", err)
        self.assertIn("p.zip", out)
        self.assertTrue(self.exists("p.bpc"))

    def test_zip_with_dash_z_is_not_a_valid_request(self):
        zip_path = self.put("p.zip", make_zip({"a.txt": b"a"}))
        code, _, err = run_main(bpc, zip_path, "-z")
        self.assertEqual(code, 1)
        self.assertIn("expected a folder, a .bpc or a .zip", err)

    def test_script_entry_point(self):
        self.put("pack/a.txt", b"x")
        code, _, _ = run_script("bpc.py", self.path("pack"))
        self.assertEqual(code, 0)
        self.assertTrue(self.exists("pack.bpc"))


# ----------------------------------------------------------------- bpcmeta

def stored_zip(entries):
    return make_zip(entries, zipfile.ZIP_STORED)


class TestMetaFormat(unittest.TestCase):
    def test_entry_header_size(self):
        self.assertEqual(bpcmeta.Entry("abc.wav", 0, 0, 0).header_size, 30 + 7)
        self.assertEqual(bpcmeta.Entry("é.wav", 0, 0, 0).header_size, 30 + 6)  # utf-8 bytes

    def test_serialize_layout(self):
        data = bpcmeta.serialize([bpcmeta.Entry("a.mp3", 35, 10, 1)])
        self.assertEqual(data, struct.pack("<I", 1) + struct.pack("<IIBH", 35, 10, 1, 5) + b"a.mp3")
        self.assertEqual(bpcmeta.serialize([]), struct.pack("<I", 0))

    def test_roundtrip_including_unicode(self):
        entries = [bpcmeta.Entry("a.wav", 35, 5, 0), bpcmeta.Entry("é/ü.mp3", 100, 0, 1)]
        self.assertEqual(bpcmeta.parse(bpcmeta.serialize(entries)), entries)
        self.assertEqual(bpcmeta.parse(bpcmeta.serialize([])), [])

    def test_serialize_errors(self):
        with self.assertRaises(bpcmeta.MetaError) as cm:
            bpcmeta.serialize([bpcmeta.Entry("x" * 0x10000, 0, 0, 0)])
        self.assertIn("name too long", str(cm.exception))
        bpcmeta.serialize([bpcmeta.Entry("x" * 0xFFFF, 0, 0, 0)])
        for offset, size, label in ((2 ** 32, 0, "offset"), (0, 2 ** 32, "size"), (-1, 0, "offset")):
            with self.subTest(label=label):
                with self.assertRaises(bpcmeta.MetaError) as cm:
                    bpcmeta.serialize([bpcmeta.Entry("a.wav", offset, size, 0)])
                self.assertIn(f"{label} ", str(cm.exception))
                self.assertIn("32 bits", str(cm.exception))
        bpcmeta.serialize([bpcmeta.Entry("a.wav", 2 ** 32 - 1, 2 ** 32 - 1, 0)])

    def test_parse_errors(self):
        good = bpcmeta.serialize([bpcmeta.Entry("a.wav", 35, 5, 0)])
        cases = [
            (b"\x01", "file too short"),
            (struct.pack("<I", 1), "record 0: truncated header"),
            (good[:-2], "record 0: truncated name"),
            (struct.pack("<I", 1) + struct.pack("<IIBH", 0, 0, 0, 1) + b"\xff", "not UTF-8"),
            (good + b"zz", "2 unexpected bytes after record 0"),
        ]
        for data, text in cases:
            with self.subTest(text=text):
                with self.assertRaises(bpcmeta.MetaError) as cm:
                    bpcmeta.parse(data)
                self.assertIn(text, str(cm.exception))

    def test_meta_error_is_a_value_error(self):
        self.assertTrue(issubclass(bpcmeta.MetaError, ValueError))

    def test_tag_for(self):
        self.assertEqual(bpcmeta.tag_for("a/b.mp3"), 1)
        self.assertEqual(bpcmeta.tag_for("A.WAV"), 0)
        for name, shown in (("a.txt", "'.txt'"), ("noext", "'(none)'")):
            with self.assertRaises(bpcmeta.MetaError) as cm:
                bpcmeta.tag_for(name)
            self.assertIn(f"no tag for extension {shown}", str(cm.exception))


class TestMetaProblems(unittest.TestCase):
    def entries(self):
        return [bpcmeta.Entry("a.wav", 35, 10, 0), bpcmeta.Entry("b.mp3", 80, 3, 1)]

    def test_clean(self):
        self.assertEqual(bpcmeta.problems(self.entries()), [])
        self.assertEqual(bpcmeta.problems([]), [])

    def test_unsafe_paths(self):
        for name in ("/abs.wav", "a\\b.wav", "a//b.wav", "../x.wav", "a/../b.wav", ""):
            with self.subTest(name=name):
                found = bpcmeta.problems([bpcmeta.Entry(name, len(name.encode()) + 30, 1, 0)])
                self.assertTrue(any("unsafe or empty path" in p for p in found), found)

    def test_duplicate_name(self):
        a = bpcmeta.Entry("a.wav", 35, 10, 0)
        b = bpcmeta.Entry("a.wav", 75, 1, 0)
        self.assertTrue(any("duplicate name" in p for p in bpcmeta.problems([a, b])))

    def test_tag_mismatch(self):
        found = bpcmeta.problems([bpcmeta.Entry("a.wav", 35, 10, 1)])
        self.assertEqual(found, ["#0 a.wav: tag 1 doesn't match the extension"])

    def test_unknown_extension(self):
        found = bpcmeta.problems([bpcmeta.Entry("a.txt", 35, 10, 0)])
        self.assertEqual(len(found), 1)
        self.assertIn("no tag for extension '.txt'", found[0])

    def test_gap_in_offsets(self):
        found = bpcmeta.problems([bpcmeta.Entry("a.wav", 40, 10, 0)])
        self.assertEqual(len(found), 1)
        self.assertIn("offset 40, expected 35", found[0])

    def test_unsorted(self):
        b = bpcmeta.Entry("b.wav", 35, 10, 0)
        a = bpcmeta.Entry("a.wav", 45 + 35, 10, 0)
        self.assertEqual(bpcmeta.problems([b, a]), ["entries are not sorted by name"])


class TestMetaSources(TmpCase):
    FILES = {"b.wav": b"BBBB", "a.mp3": b"aaaaaa", "sub/c.wav": b"cc", "sub/deep/d.mp3": b"d"}

    def folder(self):
        for name, data in self.FILES.items():
            self.put("src/" + name, data)
        return self.path("src")

    def test_data_offset_bad_headers(self):
        info = mock.Mock(header_offset=0, filename="x.wav")
        with self.assertRaises(bpcmeta.MetaError) as cm:
            bpcmeta.data_offset(io.BytesIO(b"PK\x03\x04"), info)  # shorter than a header
        self.assertIn("bad local header", str(cm.exception))
        with self.assertRaises(bpcmeta.MetaError):
            bpcmeta.data_offset(io.BytesIO(b"XXXX" + bytes(40)), info)

    def test_data_offset_counts_extra_field(self):
        head = b"PK\x03\x04" + bytes(22) + struct.pack("<HH", 5, 7) + bytes(30)
        info = mock.Mock(header_offset=0, filename="x.wav")
        self.assertEqual(bpcmeta.data_offset(io.BytesIO(head), info), 30 + 5 + 7)

    def test_scan_sorted_and_recursive(self):
        folder = self.folder()
        self.assertEqual(bpcmeta.scan(folder), sorted((n, len(d)) for n, d in self.FILES.items()))

    def test_scan_ignores_symlinked_folders(self):
        folder = self.folder()
        try:
            os.symlink(self.path("src", "sub"), self.path("src", "link"), target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not available here")
        names = [n for n, _ in bpcmeta.scan(folder)]
        self.assertFalse(any(n.startswith("link/") for n in names))
        self.assertIn("sub/c.wav", names)

    def test_entries_from_folder_chain(self):
        entries = bpcmeta.entries_from_folder(self.folder())
        self.assertEqual([e.name for e in entries], sorted(self.FILES))
        self.assertEqual([e.tag for e in entries], [1, 0, 0, 1])
        self.assertEqual(bpcmeta.problems(entries), [])

    def test_folder_unknown_extension(self):
        self.put("src/a.txt", b"x")
        with self.assertRaises(bpcmeta.MetaError):
            bpcmeta.entries_from_folder(self.path("src"))

    def test_entries_from_zip_skips_directories(self):
        zip_path = self.put("a.zip", stored_zip({"d/": b"", "d/a.wav": b"aaa", "d/b.mp3": b"bb"}))
        entries = bpcmeta.entries_from_zip(lambda: bpc.open_zip(zip_path))
        self.assertEqual([(e.name, e.size, e.tag) for e in entries], [("d/a.wav", 3, 0), ("d/b.mp3", 2, 1)])

    def test_compressed_entries_rejected(self):
        zip_path = self.put("a.zip", make_zip({"a.wav": b"a" * 500}))
        with self.assertRaises(bpcmeta.MetaError) as cm:
            bpcmeta.entries_from_zip(lambda: bpc.open_zip(zip_path))
        self.assertIn("compressed, pack with bpc.py -m", str(cm.exception))

    def test_corrupt_zip_is_a_meta_error_unless_unchecked(self):
        raw = corrupt_zip_crc(stored_zip({"a.wav": noisy(50)}))
        zip_path = self.put("a.zip", raw)
        with self.assertRaises(bpcmeta.MetaError) as cm:
            bpcmeta.entries_from_zip(lambda: bpc.open_zip(zip_path))
        self.assertIn("bad zip entry", str(cm.exception))
        entries = bpcmeta.entries_from_zip(lambda: bpc.open_zip(zip_path), check=False)
        self.assertEqual(len(entries), 1)

    def test_broken_local_header_found_without_check(self):
        raw = stored_zip({"a.wav": b"aaa"})
        zip_path = self.put("a.zip", b"XXXX" + raw[4:])
        with self.assertRaises(bpcmeta.MetaError) as cm:
            bpcmeta.entries_from_zip(lambda: bpc.open_zip(zip_path), check=False)
        self.assertIn("bad local header", str(cm.exception))

    def test_folder_zip_and_bpc_agree_and_point_at_the_data(self):
        folder = self.folder()
        packed = self.path("packed.bpc")
        bpc.pack(folder, packed, meta=True)
        with open(packed, "rb") as f:
            zip_bytes = naive_xor(f.read())
        zip_path = self.put("packed.zip", zip_bytes)

        from_folder, n = bpcmeta.build(folder)
        from_bpc, _ = bpcmeta.build(packed)
        from_zip, _ = bpcmeta.build(zip_path)
        self.assertEqual(n, 4)
        self.assertEqual(from_folder, from_bpc)
        self.assertEqual(from_folder, from_zip)
        for e in bpcmeta.parse(from_bpc):
            self.assertEqual(zip_bytes[e.offset:e.offset + e.size], self.FILES[e.name])

    def test_build_empty_folder(self):
        os.makedirs(self.path("empty"))
        self.assertEqual(bpcmeta.build(self.path("empty")), (struct.pack("<I", 0), 0))

    def test_build_reports_problems(self):
        zip_path = self.put("a.zip", stored_zip({"b.wav": b"b", "a.wav": b"a"}))
        with self.assertRaises(bpcmeta.MetaError) as cm:
            bpcmeta.build(zip_path, check=False)
        self.assertIn("entries are not sorted by name", str(cm.exception))

    def test_build_self_check(self):
        with mock.patch.object(bpcmeta, "parse", return_value=[]):
            with self.assertRaises(bpcmeta.MetaError) as cm:
                bpcmeta.build(self.folder())
        self.assertIn("failed the self check", str(cm.exception))

    def test_build_uppercase_bpc_extension(self):
        folder = self.folder()
        packed = self.path("PACKED.BPC")
        bpc.pack(folder, packed, meta=True)
        self.assertEqual(bpcmeta.build(packed)[0], bpcmeta.build(folder)[0])


class TestMetaMain(TmpCase):
    def meta_bpc(self, rel, files=None):
        files = files or {"a.wav": b"aa", "b.mp3": b"bbb"}
        folder = self.path("_src_" + rel)
        for name, data in files.items():
            self.put(f"_src_{rel}/{name}", data)
        bpc.pack(folder, self.path(rel), meta=True)
        return self.path(rel)

    def test_bpc_to_bpcmeta(self):
        src = self.meta_bpc("pack.bpc")
        code, out, err = run_main(bpcmeta, src)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("(2 files)", out)
        self.assertEqual([e.name for e in bpcmeta.parse(self.read("pack.bpcmeta"))], ["a.wav", "b.mp3"])

    def test_zip_and_folder_sources(self):
        self.put("z.zip", stored_zip({"a.wav": b"x"}))
        self.put("f/a.wav", b"x")
        code, _, _ = run_main(bpcmeta, self.path("z.zip"), self.path("f"), "-j", 2)
        self.assertEqual(code, 0)
        self.assertTrue(self.exists("z.bpcmeta") and self.exists("f.bpcmeta"))

    def test_output_file_and_folder(self):
        src = self.meta_bpc("pack.bpc")
        self.assertEqual(run_main(bpcmeta, src, "-o", self.path("x", "one.bpcmeta"))[0], 0)
        self.assertTrue(self.exists("x/one.bpcmeta"))
        other = self.meta_bpc("other.bpc")
        self.assertEqual(run_main(bpcmeta, src, other, "-o", self.path("metas"))[0], 0)
        self.assertTrue(self.exists("metas/pack.bpcmeta") and self.exists("metas/other.bpcmeta"))

    def test_same_output_twice(self):
        self.put("a.zip", stored_zip({"a.wav": b"1"}))
        self.meta_bpc("a.bpc")
        code, out, err = run_main(bpcmeta, self.path("a.zip"), self.path("a.bpc"))
        self.assertEqual(code, 1)
        self.assertIn("is already made from another source", err)
        self.assertIn("a.zip", out)

    def test_failures_are_reported_per_source(self):
        good = self.meta_bpc("good.bpc")
        compressed = self.put("c.zip", make_zip({"a.wav": b"a" * 400}))
        code, out, err = run_main(bpcmeta, good, self.path("missing.zip"), compressed)
        self.assertEqual(code, 1)
        self.assertIn("good.bpc", out)
        self.assertIn("missing.zip", err)
        self.assertIn("compressed, pack with bpc.py -m", err)
        self.assertTrue(self.exists("good.bpcmeta"))
        self.assertFalse(self.exists("c.bpcmeta"))

    def test_fast_skips_the_crc_check(self):
        raw = corrupt_zip_crc(stored_zip({"a.wav": noisy(40)}))
        self.put("a.zip", raw)
        self.assertEqual(run_main(bpcmeta, self.path("a.zip"))[0], 1)
        self.assertEqual(run_main(bpcmeta, self.path("a.zip"), "--fast")[0], 0)

    def test_wildcard(self):
        self.meta_bpc("a.bpc")
        self.meta_bpc("b.bpc", {"c.wav": b"c"})
        self.assertEqual(run_main(bpcmeta, self.path("*.bpc"), "-j", 1)[0], 0)
        self.assertTrue(self.exists("a.bpcmeta") and self.exists("b.bpcmeta"))

    def test_script_entry_point(self):
        src = self.meta_bpc("pack.bpc")
        self.assertEqual(run_script("bpcmeta.py", src)[0], 0)
        self.assertTrue(self.exists("pack.bpcmeta"))


# --------------------------------------------------------------------- cls

class TestClsHelpers(unittest.TestCase):
    def test_decode_name(self):
        self.assertEqual(cls.decode_name(b"car\0junk\0\0"), "car")
        self.assertEqual(cls.decode_name(b"\0\0"), "")
        self.assertEqual(cls.decode_name(b"full"), "full")
        with self.assertRaises(cls.ClsError) as cm:
            cls.decode_name(b"caf\xe9")
        self.assertIn("not ASCII", str(cm.exception))

    def test_encode_name(self):
        self.assertEqual(cls.encode_name("car", 6), b"car\0\0\0")
        self.assertEqual(cls.encode_name("a" * 5, 6), b"aaaaa\0")
        with self.assertRaises(cls.ClsError) as cm:
            cls.encode_name("a" * 6, 6)
        self.assertIn("name too long (6 bytes, max 5)", str(cm.exception))
        with self.assertRaises(ValueError):
            cls.encode_name("caf\u00e9", 22)

    def test_skip_array(self):
        data = struct.pack("<I", 2) + bytes(40) + b"tail"
        self.assertEqual(cls.skip_array(data, 0, len(data), 20, "sphere"), (44, 2))
        self.assertEqual(cls.skip_array(struct.pack("<I", 0), 0, 4, 20, "x"), (4, 0))
        with self.assertRaises(cls.ClsError) as cm:
            cls.skip_array(b"\0\0", 0, 2, 20, "sphere")
        self.assertIn("truncated sphere count at byte 0", str(cm.exception))
        with self.assertRaises(cls.ClsError) as cm:
            cls.skip_array(data, 0, 43, 20, "box")
        self.assertIn("truncated box array", str(cm.exception))

    def test_sizes_are_consistent(self):
        self.assertEqual(cls.CLS_BODY_SIZE, 112)
        self.assertTrue(issubclass(cls.ClsError, ValueError))


class TestParseCol(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(cls.parse_col(b""), [])
        self.assertEqual(cls.serialize_col([]), b"")

    def test_bounds_only_model(self):
        (m,) = cls.parse_col(col_model(b"car"))
        self.assertEqual(m, cls.Model("car", (1.0, 2.0, 3.0), 2.0, (-1.0, -2.0, -3.0), (1.0, 2.0, 3.0), False))

    def test_geometry_marks_model_solid(self):
        for kwargs in ({"spheres": 1}, {"boxes": 2}, {"verts": 3}, {"faces": 1}):
            with self.subTest(kwargs=kwargs):
                (m,) = cls.parse_col(col_model(**kwargs))
                self.assertTrue(m.solid)

    def test_several_models(self):
        data = col_model(b"a") + col_model(b"b", spheres=1) + col_model(b"c")
        self.assertEqual([(m.name, m.solid) for m in cls.parse_col(data)],
                         [("a", False), ("b", True), ("c", False)])

    def test_errors(self):
        fixed = struct.pack("<H", 0) + bytes(40)
        cases = [
            (b"COLL\0", "model 0: truncated header at byte 0"),
            (col_model(b"a") + b"COL", "model 1: truncated header at byte"),
            (b"COL2" + struct.pack("<I", 0), "unsupported tag"),
            (b"COLL" + struct.pack("<I", 9999) + bytes(10), "bad size 9999"),
            (b"COLL" + struct.pack("<I", 10) + bytes(10), "bad size 10"),
            (col_model(trailing=b"xx"), "2 unexpected trailing bytes"),
            (b"COLL" + struct.pack("<I", 22 + len(fixed)) + b"car".ljust(22, b"\0") + fixed,
             "truncated sphere count"),
            (b"COLL" + struct.pack("<I", 22 + len(fixed) + 4) + b"car".ljust(22, b"\0") + fixed
             + struct.pack("<I", 100), "truncated sphere array"),
            (col_model(name=b"caf\xe9"), "not ASCII"),
        ]
        for data, text in cases:
            with self.subTest(text=text):
                with self.assertRaises(cls.ClsError) as cm:
                    cls.parse_col(data)
                self.assertIn(text, str(cm.exception))

    def test_serialize_roundtrip_is_byte_exact_for_plain_models(self):
        data = col_model(b"a") + col_model(b"bb", bounds=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
        self.assertEqual(cls.serialize_col(cls.parse_col(data)), data)

    def test_serialize_drops_geometry(self):
        (m,) = cls.parse_col(col_model(spheres=2, faces=3))
        self.assertEqual(cls.serialize_col([m]), col_model())


class TestParseCls(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(cls.parse_cls(b""), [])
        self.assertEqual(cls.serialize_cls([]), b"")

    def test_record(self):
        (m,) = cls.parse_cls(cls_record(b"car"))
        self.assertEqual(m, cls.Model("car", (1.0, 2.0, 3.0), 4.0, (-1.0, -2.0, -3.0), (1.0, 2.0, 3.0)))

    def test_roundtrip(self):
        data = cls_record(b"one") + cls_record(b"two")
        self.assertEqual(cls.serialize_cls(cls.parse_cls(data)), data)

    def test_errors(self):
        cases = [
            (b"CLS", "record 0: truncated header at byte 0"),
            (cls_record(tag=b"XXXX"), "unexpected tag"),
            (cls_record(size=99), "unsupported size 99 (expected 112)"),
            (b"CLST" + struct.pack("<I", 112) + bytes(10), "truncated body"),
            (cls_record(kind=b"ABCD"), "unsupported type"),
            (cls_record(tail=b"\0" * 47 + b"\1"), "geometry data is not supported"),
            (cls_record(name=b"caf\xe9"), "not ASCII"),
            (cls_record(b"ok") + b"CLST", "record 1: truncated header"),
        ]
        for data, text in cases:
            with self.subTest(text=text):
                with self.assertRaises(cls.ClsError) as cm:
                    cls.parse_cls(data)
                self.assertIn(text, str(cm.exception))

    def test_col_and_cls_agree_on_the_model(self):
        from_col = cls.parse_col(col_model(b"car"))
        self.assertEqual(cls.parse_cls(cls.serialize_cls(from_col)), from_col)


class TestToClsToCol(unittest.TestCase):
    def test_to_cls(self):
        data, note = cls.to_cls(col_model(b"a") + col_model(b"b"))
        self.assertEqual(note, "2 models")
        same = (1, 2, 3, 2, -1, -2, -3, 1, 2, 3)  # the COL fixture, in CLS field order
        self.assertEqual(data, cls_record(b"a", bounds=same) + cls_record(b"b", bounds=same))
        self.assertEqual([m.name for m in cls.parse_cls(data)], ["a", "b"])

    def test_geometry_is_refused_by_default(self):
        with self.assertRaises(cls.ClsError) as cm:
            cls.to_cls(col_model(b"car", spheres=1) + col_model(b"van", boxes=1))
        self.assertIn("2 model(s) have geometry CLS can't hold yet (-b drops it): car, van", str(cm.exception))

    def test_long_list_of_names_is_cut(self):
        data = b"".join(col_model(b"model_%03d" % i, spheres=1) for i in range(30))
        with self.assertRaises(cls.ClsError) as cm:
            cls.to_cls(data)
        self.assertLess(len(str(cm.exception)), 200)

    def test_bounds_only(self):
        data, note = cls.to_cls(col_model(b"car", spheres=1) + col_model(b"x"), bounds_only=True)
        self.assertEqual(note, "2 models, geometry dropped in 1")
        self.assertEqual([m.name for m in cls.parse_cls(data)], ["car", "x"])

    def test_pool_argument_is_ignored(self):
        self.assertEqual(cls.to_cls(col_model(), object())[1], "1 models")
        self.assertEqual(cls.to_col(cls_record(), object())[1], "1 models")

    def test_to_col(self):
        data, note = cls.to_col(cls_record(b"a") + cls_record(b"b"))
        self.assertEqual(note, "2 models")
        bounds = (4.0, 1, 2, 3, -1, -2, -3, 1, 2, 3)  # the CLS fixture, in COL field order
        self.assertEqual(data, col_model(b"a", bounds=bounds) + col_model(b"b", bounds=bounds))
        self.assertEqual([m.name for m in cls.parse_col(data)], ["a", "b"])

    def test_errors_propagate(self):
        with self.assertRaises(cls.ClsError):
            cls.to_cls(b"junk junk junk")
        with self.assertRaises(cls.ClsError):
            cls.to_col(b"junk junk junk")


class TestClsMain(TmpCase):
    def test_col_to_cls_and_back(self):
        src = self.put("veh.col", col_model(b"a") + col_model(b"b"))
        code, out, err = run_main(cls, src)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("(2 models)", out)
        self.assertEqual(self.read("veh.cls"), cls.serialize_cls(cls.parse_col(self.read("veh.col"))))
        os.remove(src)
        self.assertEqual(run_main(cls, self.path("veh.cls"))[0], 0)
        self.assertEqual(self.read("veh.col"), col_model(b"a") + col_model(b"b"))

    def test_geometry_refused_then_forced(self):
        src = self.put("veh.col", col_model(b"car", spheres=1))
        code, _, err = run_main(cls, src)
        self.assertEqual(code, 1)
        self.assertIn("geometry CLS can't hold", err)
        self.assertFalse(self.exists("veh.cls"))
        code, out, _ = run_main(cls, src, "--cls", "-b")
        self.assertEqual(code, 0)
        self.assertIn("geometry dropped in 1", out)
        self.assertTrue(self.exists("veh.cls"))

    def test_bounds_only_needs_col_direction(self):
        src = self.put("veh.cls", cls_record())
        code, _, err = run_main(cls, src, "-b")
        self.assertEqual(code, 2)
        self.assertIn("-b goes with --cls", err)

    def test_direction_required_for_folders(self):
        self.put("d/a.col", col_model())
        code, _, err = run_main(cls, self.path("d"))
        self.assertEqual(code, 2)
        self.assertIn("say which way", err)

    def test_folder_with_direction_and_output(self):
        self.put("d/a.col", col_model(b"a"))
        self.put("d/sub/b.col", col_model(b"b"))
        code, _, _ = run_main(cls, self.path("d"), "--cls", "-o", self.path("out"), "-j", 1)
        self.assertEqual(code, 0)
        self.assertEqual([m.name for m in cls.parse_cls(self.read("out/a.cls"))], ["a"])
        self.assertEqual([m.name for m in cls.parse_cls(self.read("out/sub/b.cls"))], ["b"])

    def test_cls_folder_to_col(self):
        self.put("d/a.cls", cls_record(b"a"))
        self.assertEqual(run_main(cls, self.path("d"), "--col")[0], 0)
        self.assertEqual(self.read("d/a.col"), col_model(b"a", bounds=(4.0, 1, 2, 3, -1, -2, -3, 1, 2, 3)))

    def test_zip_and_bpc_are_rewritten_in_place(self):
        raw = make_zip({"a.col": col_model(b"a"), "n/b.col": col_model(b"b"), "keep.txt": b"k"})
        self.put("pack.zip", raw)
        self.put("pack.bpc", raw)
        code, out, _ = run_main(cls, self.path("pack.zip"), self.path("pack.bpc"), "--cls", "-j", 1)
        self.assertEqual(code, 0)
        for name in ("pack.zip", "pack.bpc"):
            with zipfile.ZipFile(self.path(name)) as z:
                self.assertEqual(z.namelist(), ["a.cls", "n/b.cls", "keep.txt"])
                self.assertEqual([m.name for m in cls.parse_cls(z.read("a.cls"))], ["a"])
            self.assertEqual(self.read(name + ".bak"), raw)

    def test_zip_member_failures_keep_the_original(self):
        raw = make_zip({"ok.col": col_model(b"a"), "solid.col": col_model(b"s", boxes=1)})
        self.put("pack.zip", raw)
        code, out, err = run_main(cls, self.path("pack.zip"), "--cls", "-j", 1)
        self.assertEqual(code, 1)
        self.assertIn("solid.col", err)
        with zipfile.ZipFile(self.path("pack.zip")) as z:
            self.assertEqual(z.namelist(), ["ok.cls", "solid.col"])

    def test_encrypted_bpc_hint(self):
        self.put("pack.bpc", naive_xor(make_zip({"a.col": col_model()})))
        code, _, err = run_main(cls, self.path("pack.bpc"), "--cls")
        self.assertEqual(code, 1)
        self.assertIn("bpc.py decrypts it", err)

    def test_script_entry_point(self):
        src = self.put("veh.col", col_model())
        self.assertEqual(run_script("cls.py", src)[0], 0)
        self.assertTrue(self.exists("veh.cls"))


# ---------------------------------------------------------------- codecs

def gradient(width, height, alpha=False):
    """A smooth, deterministic RGBA test image as a uint8 array (H, W, 4)."""
    rows = []
    for y in range(height):
        rows.append([
            (
                x * 255 // max(1, width - 1),
                y * 255 // max(1, height - 1),
                (x + y) * 255 // max(1, width + height - 2),
                255 if not alpha else 60 + x * 180 // max(1, width - 1),
            )
            for x in range(width)
        ])
    return np.array(rows, np.uint8)


def flat_image(width, height, colour):
    return np.array([[colour] * width for _ in range(height)], np.uint8)


def void_block(r, g, b, a):
    """An LDR void-extent ASTC block, assembled by hand from the spec."""
    word = 0x1FC | (3 << 10) | (((1 << 52) - 1) << 12)
    for i, c in enumerate((r, g, b, a)):
        word |= (c * 257) << (64 + 16 * i)
    return word.to_bytes(16, "little")


def mean_error(a, b):
    return float(np.abs(a.astype(np.int64) - b.astype(np.int64)).mean())


def threaded_pool():
    return Pool(2)  # threads: no processes, so the work is visible to coverage tools


# -------------------------------------------------------------------- astc

class TestAstcIse(unittest.TestCase):
    def test_ise_bits_known_values(self):
        cases = [(1, 2, 1), (18, 2, 18), (4, 256, 32), (5, 6, 13), (18, 6, 47),
                 (3, 5, 7), (5, 3, 8), (3, 10, 10), (6, 10, 20), (1, 20, 5)]
        for count, rng, bits in cases:
            with self.subTest(count=count, rng=rng):
                self.assertEqual(astc.ise_bits(count, rng), bits)

    def test_bit_reader(self):
        r = astc.BitReader(0b1011_0110, 8)
        self.assertEqual(r.read(3), 0b110)
        self.assertEqual(r.read(2), 0b10)
        self.assertEqual(r.read(3), 0b101)
        self.assertEqual(astc.BitReader(0xFFFF, 4).read(8), 0xF)  # value is cut to nbits

    def test_trit_and_quint_tables(self):
        self.assertEqual(len(astc.TRIT_FORWARD), 256)
        self.assertEqual(len(astc.QUINT_FORWARD), 128)
        self.assertTrue(all(len(t) == 5 and max(t) <= 2 for t in astc.TRIT_FORWARD))
        self.assertTrue(all(len(q) == 3 and max(q) <= 4 for q in astc.QUINT_FORWARD))
        self.assertEqual(len(astc.TRIT_TABLE), 3 ** 5)
        self.assertEqual(len(astc.QUINT_TABLE), 5 ** 3)
        for digits, packed in astc.TRIT_TABLE.items():
            self.assertEqual(astc.TRIT_FORWARD[packed], digits)
        for digits, packed in astc.QUINT_TABLE.items():
            self.assertEqual(astc.QUINT_FORWARD[packed], digits)

    def test_encode_decode_roundtrip_all_ranges(self):
        rnd = random.Random(11)
        for rng in astc.ISE_LEVELS:
            for count in (1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 18, 31):
                values = [rnd.randrange(rng) for _ in range(count)]
                bits = astc.ise_bits(count, rng)
                packed = astc.ise_encode(values, rng)
                self.assertEqual(packed >> bits, 0, (rng, count))
                reader = astc.BitReader(packed, bits)
                self.assertEqual(astc.ise_decode(reader, count, rng), values, (rng, count))

    def test_every_value_of_every_range_roundtrips(self):
        for rng in astc.ISE_LEVELS:
            values = list(range(rng)) * 2
            packed = astc.ise_encode(values, rng)
            bits = astc.ise_bits(len(values), rng)
            self.assertEqual(astc.ise_decode(astc.BitReader(packed, bits), len(values), rng), values)

    def test_decoding_continues_where_the_reader_stopped(self):
        packed = astc.ise_encode([1, 2, 3, 0, 1, 2, 2, 1], 6)
        r = astc.BitReader(packed, astc.ise_bits(8, 6))
        self.assertEqual(astc.ise_decode(r, 5, 6), [1, 2, 3, 0, 1])
        self.assertEqual(astc.ise_decode(r, 3, 6)[:1], [2])


class TestAstcQuantisation(unittest.TestCase):
    def test_color_known_values(self):
        self.assertEqual([astc.unquantize_color(v, 256) for v in (0, 77, 255)], [0, 77, 255])
        self.assertEqual([astc.unquantize_color(v, 16) for v in (0, 1, 15)], [0, 17, 255])
        self.assertEqual([astc.unquantize_color(v, 2) for v in (0, 1)], [0, 255])
        self.assertEqual([astc.unquantize_color(v, 4) for v in range(4)], [0, 85, 170, 255])

    def test_color_ranges_span_0_255(self):
        """Plain bit ranges are monotonic; trit/quint ranges interleave their
        values (0, 255, 51, 204, ...) but still cover 0..255 without repeats."""
        for rng in [2, 4] + astc.COLOR_RANGES:
            values = [astc.unquantize_color(v, rng) for v in range(rng)]
            bits, trits, quints = astc.ISE_LEVELS[rng]
            with self.subTest(rng=rng):
                self.assertEqual((min(values), max(values)), (0, 255))
                self.assertEqual(len(set(values)), rng)
                if not (trits or quints):
                    self.assertEqual(values, sorted(values))

    def test_weight_known_values(self):
        self.assertEqual([astc.unquantize_weight(v, 3) for v in range(3)], [0, 32, 64])
        self.assertEqual([astc.unquantize_weight(v, 5) for v in range(5)], [0, 16, 32, 48, 64])
        self.assertEqual([astc.unquantize_weight(v, 2) for v in range(2)], [0, 64])
        self.assertEqual([astc.unquantize_weight(v, 4) for v in range(4)], [0, 21, 43, 64])
        self.assertEqual([astc.unquantize_weight(v, 8) for v in range(8)], [0, 9, 18, 27, 37, 46, 55, 64])

    def test_weight_ranges_span_0_64(self):
        for rng in astc.WEIGHT_RANGES:
            values = [astc.unquantize_weight(v, rng) for v in range(rng)]
            bits, trits, quints = astc.ISE_LEVELS[rng]
            with self.subTest(rng=rng):
                self.assertEqual((min(values), max(values)), (0, 64))
                self.assertEqual(len(set(values)), rng)
                self.assertNotIn(33, values)  # 32 is followed by 34: the +1 above 32 rule
                if not (trits or quints):
                    self.assertEqual(values, sorted(values))

    def test_trit_and_quint_weights_match_the_reference_tables(self):
        self.assertEqual([astc.unquantize_weight(v, 6) for v in range(6)], [0, 64, 12, 52, 25, 39])
        self.assertEqual([astc.unquantize_weight(v, 10) for v in range(10)],
                         [0, 64, 7, 57, 14, 50, 21, 43, 28, 36])

    def test_luts_agree_with_the_functions(self):
        for rng in (6, 16, 40, 256):
            self.assertEqual(astc._color_lut(rng), tuple(astc.unquantize_color(v, rng) for v in range(len(astc._color_lut(rng)))))
        for rng in (3, 5, 8, 12, 32):
            self.assertEqual(astc._weight_lut(rng), tuple(astc.unquantize_weight(v, rng) for v in range(len(astc._weight_lut(rng)))))

    def test_quant_luts(self):
        index, dequant = astc.quant_luts("c", 256)
        self.assertEqual(index.shape, (256,))
        self.assertEqual(list(index), list(range(256)))
        self.assertEqual(list(dequant), list(range(256)))
        index, dequant = astc.quant_luts("c", 16)
        self.assertEqual(int(dequant[index[100]]), 102)  # nearest multiple of 17
        index, dequant = astc.quant_luts("w", 8)
        self.assertEqual(index.shape, (65,))
        self.assertEqual(int(index[0]), 0)
        self.assertEqual(int(index[64]), 7)


class TestAstcModes(unittest.TestCase):
    def test_every_block_mode(self):
        good, messages = {}, set()
        for mode in range(2048):
            try:
                good[mode] = astc.decode_block_mode(mode)
            except ValueError as e:
                messages.add(str(e))
        self.assertGreater(len(good), 1000)
        # The "bad quant mode" guard in decode_block_mode can not fire for any 11-bit
        # mode: base is always 2..7, so quant_mode is always 0..11. If this ever starts
        # failing the guard became reachable and deserves a test of its own.
        self.assertEqual(messages, {"reserved block mode"})
        for mode, (x, y, dual, rng) in good.items():
            self.assertTrue(2 <= x <= 12 and 2 <= y <= 12, mode)
            self.assertIn(dual, (0, 1))
            self.assertIn(rng, astc.WEIGHT_RANGES)
        sizes = {(x, y) for x, y, dual, _ in good.values() if not dual}
        for want in ((4, 4), (5, 5), (6, 6), (8, 8), (2, 2), (12, 2), (2, 12), (6, 10), (10, 6), (7, 9)):
            self.assertIn(want, sizes)

    def test_reserved_modes(self):
        with self.assertRaises(ValueError):
            astc.decode_block_mode(0)  # low bits zero and k == 0
        with self.assertRaises(ValueError):
            astc.decode_block_mode(452)  # (mode & 3) == 0, k == 1, k2 == 3, k3 == 2

    def test_hash_and_partitions(self):
        seen = set()
        for seed in range(0, 1024, 7):
            for count in (2, 3, 4):
                for small in (False, True):
                    for x, y in ((0, 0), (3, 2), (5, 5), (11, 11)):
                        p = astc.partition_index(seed, x, y, count, small)
                        self.assertTrue(0 <= p < count, (seed, count, p))
                        seen.add((count, p))
        self.assertEqual(seen, {(c, p) for c in (2, 3, 4) for p in range(c)})
        self.assertEqual(astc._hash52(5), astc._hash52(5))
        self.assertNotEqual(astc._hash52(5), astc._hash52(6))
        self.assertLess(astc._hash52(0xFFFFFFFF), 2 ** 32)

    def test_partition_map(self):
        m = astc.partition_map(100, 3, 6, 6)
        self.assertEqual(m.shape, (36,))
        self.assertIs(astc.partition_map(100, 3, 6, 6), m)  # cached
        self.assertTrue(set(m.tolist()) <= {0, 1, 2})
        small = astc.partition_map(100, 2, 4, 4)  # 16 texels: the "small" variant
        self.assertEqual(small.shape, (16,))
        self.assertEqual([astc.partition_index(100, s, t, 2, True) for t in range(4) for s in range(4)], small.tolist())

    def test_infill_matrix(self):
        for bw, bh, nx, ny in ((4, 4, 4, 4), (6, 6, 2, 2), (6, 6, 5, 3), (12, 12, 12, 12), (5, 4, 3, 4)):
            m = astc.infill_matrix(bw, bh, nx, ny)
            with self.subTest(block=(bw, bh), grid=(nx, ny)):
                self.assertEqual(m.shape, (bw * bh, nx * ny))
                self.assertTrue((m >= 0).all())
                self.assertEqual(set(m.sum(1).tolist()), {16})
        self.assertTrue(np.array_equal(astc.infill_matrix(4, 4, 4, 4), np.eye(16, dtype=np.int64) * 16))


class TestAstcEndpoints(unittest.TestCase):
    def test_luminance_modes(self):
        self.assertEqual(astc.endpoints(0, (10, 200)), ((10, 10, 10, 255), (200, 200, 200, 255)))
        self.assertEqual(astc.endpoints(1, (0x40, 0x20)), ((16, 16, 16, 255), (48, 48, 48, 255)))
        self.assertEqual(astc.endpoints(1, (0xFC, 0xFF)), ((255,) * 4, (255,) * 4))  # high end clamped
        self.assertEqual(astc.endpoints(1, (0xFC, 0x3F))[1], (126, 126, 126, 255))
        self.assertEqual(astc.endpoints(4, (1, 2, 3, 4)), ((1, 1, 1, 3), (2, 2, 2, 4)))

    def test_luminance_alpha_offset_mode(self):
        self.assertEqual(astc.endpoints(5, (100, 0x10, 50, 0)), ((50, 50, 50, 25), (58, 58, 58, 25)))
        low, high = astc.endpoints(5, (10, 0x7E, 250, 0x7E))  # negative offsets, clamping
        self.assertTrue(all(0 <= c <= 255 for c in low + high))

    def test_scale_modes(self):
        self.assertEqual(astc.endpoints(6, (100, 150, 200, 128)), ((50, 75, 100, 255), (100, 150, 200, 255)))
        self.assertEqual(astc.endpoints(10, (100, 150, 200, 128, 10, 20)), ((50, 75, 100, 10), (100, 150, 200, 20)))

    def test_rgb_direct_modes(self):
        self.assertEqual(astc.endpoints(8, (10, 200, 20, 210, 30, 220)), ((10, 20, 30, 255), (200, 210, 220, 255)))
        self.assertEqual(astc.endpoints(12, (10, 200, 20, 210, 30, 220, 5, 6)),
                         ((10, 20, 30, 5), (200, 210, 220, 6)))

    def test_rgb_direct_blue_contract(self):
        # endpoint 1 is darker than endpoint 0: swapped and blue-contracted
        self.assertEqual(astc.endpoints(8, (200, 10, 210, 20, 220, 30)),
                         ((20, 25, 30, 255), (210, 215, 220, 255)))
        self.assertEqual(astc.endpoints(12, (200, 10, 210, 20, 220, 30, 7, 9)),
                         ((20, 25, 30, 9), (210, 215, 220, 7)))

    def test_rgb_base_offset_modes(self):
        self.assertEqual(astc.endpoints(9, (100, 0x10, 100, 0x10, 100, 0x10)),
                         ((50, 50, 50, 255), (58, 58, 58, 255)))
        # negative offsets: blue contraction and swap
        self.assertEqual(astc.endpoints(9, (100, 0x7E, 100, 0x7E, 100, 0x7E)),
                         ((49, 49, 49, 255), (50, 50, 50, 255)))
        self.assertEqual(astc.endpoints(13, (100, 0x10, 100, 0x10, 100, 0x10, 100, 0x10)),
                         ((50, 50, 50, 50), (58, 58, 58, 58)))

    def test_hdr_modes_are_refused(self):
        for cem in (2, 3, 7, 11, 14, 15, 16):
            with self.subTest(cem=cem):
                with self.assertRaises(NotImplementedError):
                    astc.endpoints(cem, (0,) * 8)

    def test_helpers(self):
        self.assertEqual(astc._clamp(-3), 0)
        self.assertEqual(astc._clamp(300), 255)
        self.assertEqual(astc._clamp(7), 7)
        self.assertEqual(astc._blue_contract(10, 20, 30), (20, 25, 30))
        self.assertEqual(astc._bit_transfer_signed(0x7E, 100), (-1, 50))
        self.assertEqual(astc._bit_transfer_signed(0x80, 0), (0, 0x80))
        self.assertEqual(astc._field(0b110100, 2, 3), 0b101)


class TestAstcBlocks(unittest.TestCase):
    def test_void_extent(self):
        for bw, bh in ((4, 4), (6, 6), (12, 12)):
            out = astc.decode_block(void_block(10, 20, 30, 40), bw, bh)
            self.assertEqual(out.shape, (bw * bh, 4))
            self.assertEqual(out.dtype, np.uint8)
            self.assertTrue((out == np.array([10, 20, 30, 40], np.uint8)).all())
        out = astc.decode_block(void_block(255, 0, 1, 255), 4, 4)
        self.assertEqual(out[0].tolist(), [255, 0, 1, 255])

    def test_hdr_void_extent_is_refused(self):
        block = bytearray(void_block(1, 2, 3, 4))
        block[1] |= 2  # bit 9: HDR
        with self.assertRaises(NotImplementedError):
            astc.decode_block(bytes(block), 4, 4)

    def test_known_error_blocks(self):
        with self.assertRaises(ValueError):
            astc.decode_block(bytes(16), 6, 6)  # block mode 0 is reserved
        with self.assertRaises(ValueError):
            astc.decode_block(b"\xff" * 16, 4, 4)

    def test_fuzz_random_blocks(self):
        """Random blocks reach almost every path of the decoder (partitions, dual
        plane, all colour modes). A block either decodes or raises one of two errors."""
        rnd = random.Random(2024)
        decoded = refused = 0
        for bw, bh in ((4, 4), (6, 6), (8, 8), (12, 12), (5, 5), (10, 10), (8, 5)):
            for _ in range(5000):
                block = bytes(rnd.randrange(256) for _ in range(16))
                try:
                    out = astc.decode_block(block, bw, bh)
                except (ValueError, NotImplementedError):
                    refused += 1
                    continue
                decoded += 1
                self.assertEqual(out.shape, (bw * bh, 4))
                self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(decoded, 500)
        self.assertGreater(refused, decoded)


class TestAstcLevels(unittest.TestCase):
    BLOCKS = [(4, 4), (5, 4), (6, 6), (8, 8), (10, 5), (12, 12)]

    MAX_ERROR = {(4, 4): 7, (5, 4): 8, (6, 6): 11, (8, 8): 14, (10, 5): 15, (12, 12): 21}

    def test_roundtrip_gradient(self):
        img = gradient(24, 20)
        for bw, bh in self.BLOCKS:
            data = astc.encode_level(img, bw, bh)
            nbx, nby = -(-24 // bw), -(-20 // bh)
            with self.subTest(block=(bw, bh)):
                self.assertEqual(len(data), nbx * nby * 16)
                out = astc.decode_level(data, 24, 20, bw, bh)
                self.assertEqual(out.shape, (20, 24, 4))
                self.assertEqual(out.dtype, np.uint8)
                self.assertLess(mean_error(out, img), self.MAX_ERROR[(bw, bh)])
                self.assertTrue((out[..., 3] == 255).all())

    def test_roundtrip_with_alpha(self):
        img = gradient(16, 12, alpha=True)
        out = astc.decode_level(astc.encode_level(img, 4, 4), 16, 12, 4, 4)
        self.assertLess(mean_error(out, img), 15)
        self.assertLess(mean_error(out[..., 3], img[..., 3]), 15)  # one line fit in 4-D: alpha is the weak spot
        self.assertTrue((out[..., 3] < 255).any())

    def test_flat_image_is_exact(self):
        for colour in ((10, 200, 30, 255), (1, 2, 3, 4)):
            img = flat_image(13, 9, colour)
            data = astc.encode_level(img, 6, 6)
            out = astc.decode_level(data, 13, 9, 6, 6)
            self.assertTrue(np.array_equal(out, img))

    def test_odd_sizes_and_cropping(self):
        img = gradient(13, 7)
        data = astc.encode_level(img, 6, 6)
        self.assertEqual(len(data), 3 * 2 * 16)
        out = astc.decode_level(data, 13, 7, 6, 6)
        self.assertEqual(out.shape, (7, 13, 4))
        one = astc.decode_level(astc.encode_level(gradient(1, 1), 4, 4), 1, 1, 4, 4)
        self.assertEqual(one.shape, (1, 1, 4))

    def test_mixed_flat_and_detailed_blocks(self):
        img = gradient(24, 12)
        img[:, :12] = (200, 100, 50, 255)  # left half is flat: void-extent blocks
        data = astc.encode_level(img, 6, 6)
        out = astc.decode_level(data, 24, 12, 6, 6)
        self.assertTrue(np.array_equal(out[:, :12], img[:, :12]))
        self.assertLess(mean_error(out, img), 10)

    def test_repeated_blocks_hit_the_caches(self):
        tile = gradient(6, 6)
        img = np.tile(tile, (4, 4, 1))
        flat = np.concatenate([np.full((6, 24, 4), 77, np.uint8), img], axis=0)
        data = astc.encode_level(flat, 6, 6)
        blocks = [data[i:i + 16] for i in range(0, len(data), 16)]
        self.assertLess(len(set(blocks)), len(blocks))
        out = astc.decode_level(data, 24, 30, 6, 6)
        self.assertEqual(out.shape, (30, 24, 4))
        self.assertTrue((out[:6] == 77).all())

    def test_cache_limits_do_not_change_results(self):
        img = np.tile(gradient(6, 6), (3, 3, 1))
        data = astc.encode_level(img, 6, 6)
        want = astc.decode_level(data, 18, 18, 6, 6)
        with mock.patch.object(astc, "MEMO_LIMIT", 1):
            self.assertEqual(astc.encode_level(img, 6, 6), data)
            self.assertTrue(np.array_equal(astc.decode_level(data, 18, 18, 6, 6), want))

    def test_not_enough_data(self):
        with self.assertRaises(ValueError) as cm:
            astc.decode_level(bytes(16), 12, 12, 6, 6)
        self.assertIn("not enough data", str(cm.exception))

    def test_parallel_matches_serial(self):
        img = gradient(48, 36, alpha=True)
        img[:6] = (9, 9, 9, 255)
        serial = astc.encode_level(img, 4, 4)
        with mock.patch.object(astc, "PAR_MIN_ASTC_PACK_BLOCKS", 1), \
                mock.patch.object(astc, "PAR_MIN_ASTC_DECODE_BLOCKS", 1), threaded_pool() as pool:
            self.assertEqual(astc.encode_level(img, 4, 4, pool), serial)
            self.assertTrue(np.array_equal(astc.decode_level(serial, 48, 36, 4, 4, pool),
                                           astc.decode_level(serial, 48, 36, 4, 4)))

    def test_small_images_do_not_use_the_pool(self):
        img = gradient(8, 8)
        with threaded_pool() as pool:
            self.assertEqual(astc.encode_level(img, 4, 4, pool), astc.encode_level(img, 4, 4))
            data = astc.encode_level(img, 4, 4)
            self.assertTrue(np.array_equal(astc.decode_level(data, 8, 8, 4, 4, pool),
                                           astc.decode_level(data, 8, 8, 4, 4)))
            self.assertIsNone(pool.ex)  # never started

    def test_no_block_mode_available(self):
        with mock.patch.object(astc, "_block_mode_candidates", return_value=()):
            with self.assertRaises(ValueError) as cm:
                astc.encode_level(gradient(8, 8), 4, 4)
        self.assertIn("no suitable block mode", str(cm.exception))

    def test_choose_config(self):
        for bw, bh, nvals in ((4, 4, 6), (6, 6, 8), (12, 12, 6), (12, 12, 8)):
            best = astc.choose_config(bw, bh, nvals, 100.0)
            with self.subTest(block=(bw, bh), nvals=nvals):
                err, mode, nx, ny, wr, cr = best
                self.assertGreater(err, 0)
                self.assertEqual(astc.decode_block_mode(mode), (nx, ny, 0, wr))
                self.assertLessEqual(nx, bw)
                self.assertLessEqual(ny, bh)
                self.assertGreaterEqual(cr, 16)
        self.assertGreater(len(astc._block_mode_candidates(6, 6, 6)), 10)
        self.assertEqual(astc._block_mode_candidates(2, 2, 6), ())  # grid never below 2x2 weights of 24+ bits

    def test_encoder_output_decodes_block_by_block(self):
        data = astc.encode_level(gradient(12, 12), 6, 6)
        for i in range(0, len(data), 16):
            out = astc.decode_block(data[i:i + 16], 6, 6)
            self.assertEqual(out.shape, (36, 4))


# -------------------------------------------------------------------- etc2

class TestEtc2Helpers(unittest.TestCase):
    def test_expansions(self):
        self.assertEqual((etc2._expand4(0xA), etc2._expand4(0), etc2._expand4(15)), (0xAA, 0, 255))
        self.assertEqual((etc2._expand5(31), etc2._expand5(0), etc2._expand5(1)), (255, 0, 8))
        self.assertEqual((etc2._expand6(63), etc2._expand6(0), etc2._expand6(1)), (255, 0, 4))

    def test_sign_and_clamp(self):
        self.assertEqual([etc2._sign3(v) for v in range(8)], [0, 1, 2, 3, -4, -3, -2, -1])
        self.assertEqual((etc2._clamp255(-5), etc2._clamp255(300), etc2._clamp255(77)), (0, 255, 77))

    def test_index_bits(self):
        self.assertEqual(etc2._etc_indices(bytes(8)), [0] * 16)
        self.assertEqual(etc2._etc_indices(b"\0" * 4 + b"\xff" * 4), [1] * 16)  # raw 3 -> modifier index 1
        only_last = b"\0" * 4 + struct.pack(">I", 0x00010001)  # k = 15 is pixel x=3, y=3
        want = [0] * 15 + [1]
        self.assertEqual(etc2._etc_indices(only_last), want)

    def test_tables_are_sane(self):
        self.assertEqual(len(etc2.ETC_MODIFIERS), 8)
        self.assertTrue(all(m[0] == -m[3] and m[1] == -m[2] for m in etc2.ETC_MODIFIERS))
        self.assertEqual(len(etc2.EAC_MODIFIERS), 16)
        self.assertEqual(sorted(etc2.ETC_INDEX_MAP), [0, 1, 2, 3])
        self.assertEqual([etc2.ETC_INDEX_MAP_INVERSE[etc2.ETC_INDEX_MAP[i]] for i in range(4)], [0, 1, 2, 3])


def classify_etc_block(block):
    """T / H / planar / differential / individual, following the ETC2 spec rules."""
    b0, b1, b2, b3 = block[:4]
    if not (b3 >> 1) & 1:
        return "individual"

    def sign3(v):
        return v - 8 if v & 4 else v

    r, g, b = (b0 >> 3) & 31, (b1 >> 3) & 31, (b2 >> 3) & 31
    if not 0 <= r + sign3(b0 & 7) <= 31:
        return "T"
    if not 0 <= g + sign3(b1 & 7) <= 31:
        return "H"
    if not 0 <= b + sign3(b2 & 7) <= 31:
        return "planar"
    return "differential"


class TestEtc2Decode(unittest.TestCase):
    def test_individual_block_by_hand(self):
        # colours (0x11,0x33,0x55) and (0x22,0x44,0x66), tables 2 and 5, no flip, all indices 0
        block = bytes([0x12, 0x34, 0x56, (2 << 5) | (5 << 2)]) + bytes(4)
        out = etc2._etc_decode_rgb_block(block)
        self.assertEqual(out.shape, (4, 4, 3))
        left = [max(0, 0x11 - 29), 0x33 - 29, 0x55 - 29]
        right = [max(0, 0x22 - 80), max(0, 0x44 - 80), 0x66 - 80]
        for y in range(4):
            for x in range(4):
                self.assertEqual(out[y, x].tolist(), left if x < 2 else right)
        flipped = bytes([0x12, 0x34, 0x56, (2 << 5) | (5 << 2) | 1]) + bytes(4)
        out = etc2._etc_decode_rgb_block(flipped)
        for y in range(4):
            for x in range(4):
                self.assertEqual(out[y, x].tolist(), left if y < 2 else right)

    def test_differential_block_by_hand(self):
        # base (r,g,b) = (8,8,8) with +1 delta, table 0 / 0, no flip, indices 0 (modifier -8)
        block = bytes([(8 << 3) | 1, (8 << 3) | 1, (8 << 3) | 1, 0b10]) + bytes(4)
        out = etc2._etc_decode_rgb_block(block)
        base0, base1 = etc2._expand5(8), etc2._expand5(9)
        self.assertEqual(out[0, 0].tolist(), [base0 - 8] * 3)
        self.assertEqual(out[0, 3].tolist(), [base1 - 8] * 3)

    def test_block_length_is_checked(self):
        with self.assertRaises(ValueError):
            etc2._etc_decode_rgb_block(bytes(7))
        with self.assertRaises(ValueError):
            etc2._eac_decode_alpha(bytes(9))

    def test_every_mode_matches_the_vectorised_decoder(self):
        rnd = random.Random(99)
        blocks = [bytes(rnd.randrange(256) for _ in range(8)) for _ in range(3000)]
        kinds = {classify_etc_block(b) for b in blocks}
        self.assertEqual(kinds, {"individual", "differential", "T", "H", "planar"})
        arr8 = np.array([list(b) for b in blocks], np.uint8)
        vec = etc2._etc_decode_rgb_vec(arr8)
        self.assertEqual(vec.shape, (3000, 16, 3))
        for i, block in enumerate(blocks):
            scalar = etc2._etc_decode_rgb_block(block).reshape(16, 3)
            self.assertTrue(np.array_equal(scalar, vec[i]), (i, classify_etc_block(block)))

    def test_t_and_h_blocks_use_at_most_four_colours(self):
        rnd = random.Random(5)
        checked = {"T": 0, "H": 0}
        while min(checked.values()) < 20:
            block = bytes(rnd.randrange(256) for _ in range(8))
            kind = classify_etc_block(block)
            if kind in checked:
                pixels = {tuple(p) for p in etc2._etc_decode_rgb_block(block).reshape(16, 3).tolist()}
                self.assertLessEqual(len(pixels), 4)
                checked[kind] += 1

    def test_planar_blocks_are_smooth(self):
        rnd = random.Random(6)
        found = 0
        while found < 20:
            block = bytes(rnd.randrange(256) for _ in range(8))
            if classify_etc_block(block) != "planar":
                continue
            out = etc2._etc_decode_rgb_block(block).astype(int)
            found += 1
            for y in range(4):
                for c in range(3):
                    row = out[y, :, c]
                    if 0 < row.min() and row.max() < 255:  # not clamped: steps differ by at most 1
                        steps = row[1:] - row[:-1]
                        self.assertLessEqual(int(steps.max() - steps.min()), 1)

    def test_eac_block_by_hand(self):
        block = bytes([100, 0x20]) + bytes(6)  # base 100, multiplier 2, table 0, all indices 0 (-3)
        self.assertTrue((etc2._eac_decode_alpha(block) == 94).all())
        high = bytes([250, 0xF0]) + b"\xff" * 6  # multiplier 15, all indices 7: clamps at 255
        self.assertTrue((etc2._eac_decode_alpha(high) == 255).all())
        low = bytes([2, 0xF0]) + bytes(6)  # 2 + (-3 * 15) clamps at 0
        self.assertTrue((etc2._eac_decode_alpha(low) == 0).all())

    def test_eac_vectorised_matches_scalar(self):
        rnd = random.Random(8)
        blocks = [bytes(rnd.randrange(256) for _ in range(8)) for _ in range(1500)]
        vec = etc2._eac_decode_vec(np.array([list(b) for b in blocks], np.uint8))
        for i, block in enumerate(blocks):
            self.assertTrue(np.array_equal(etc2._eac_decode_alpha(block).reshape(16), vec[i]), i)

    def test_decode_level_sizes(self):
        data = bytes(range(256)) * 2
        rgb = etc2.etc2_decode_level(data, 8, 8, rgba=False)
        self.assertEqual(rgb.shape, (8, 8, 4))
        self.assertTrue((rgb[..., 3] == 255).all())
        rgba = etc2.etc2_decode_level(data, 8, 8, rgba=True)
        self.assertEqual(rgba.shape, (8, 8, 4))
        self.assertEqual(etc2.etc2_decode_level(data, 5, 3, True).shape, (3, 5, 4))
        self.assertEqual(etc2.etc2_decode_level(data, 1, 1, False).shape, (1, 1, 4))

    def test_decode_level_is_row_major_in_blocks(self):
        # four different void-like blocks laid out 2x2: block order is row-major
        blocks = [bytes([v << 4 | v, v << 4 | v, v << 4 | v, 0]) + bytes(4) for v in (0, 5, 10, 15)]
        out = etc2.etc2_decode_level(b"".join(blocks), 8, 8, rgba=False)
        corner = [int(out[0, 0, 0]), int(out[0, 4, 0]), int(out[4, 0, 0]), int(out[4, 4, 0])]
        self.assertEqual(corner, [0, 77, 162, 247])  # expand4(v) - 8, clamped, per block

    def test_decode_level_too_short(self):
        with self.assertRaises(ValueError) as cm:
            etc2.etc2_decode_level(bytes(15), 4, 4, rgba=True)
        self.assertIn("ETC2 level is too short: 15 < 16", str(cm.exception))

    def test_decode_level_in_chunks(self):
        rnd = random.Random(2)
        data = bytes(rnd.randrange(256) for _ in range(16 * 20))
        want = etc2.etc2_decode_level(data, 20, 16, True)
        with mock.patch.object(etc2, "ETC2_CHUNK_BLOCKS", 3):
            self.assertTrue(np.array_equal(etc2.etc2_decode_level(data, 20, 16, True), want))


class TestEtc2Encode(unittest.TestCase):
    def test_roundtrip_rgb(self):
        img = gradient(16, 16)
        data = etc2.etc2_encode_level(img, False)
        self.assertEqual(len(data), 16 * 8)
        out = etc2.etc2_decode_level(data, 16, 16, False)
        self.assertEqual(out.shape, (16, 16, 4))
        self.assertLess(mean_error(out[..., :3], img[..., :3]), 13)
        self.assertTrue((out[..., 3] == 255).all())

    def test_roundtrip_rgba(self):
        img = gradient(16, 12, alpha=True)
        data = etc2.etc2_encode_level(img, True)
        self.assertEqual(len(data), 4 * 3 * 16)
        out = etc2.etc2_decode_level(data, 16, 12, True)
        self.assertLess(mean_error(out[..., :3], img[..., :3]), 15)
        self.assertLess(mean_error(out[..., 3], img[..., 3]), 3)  # EAC alpha is very accurate

    def test_opaque_alpha_is_exact(self):
        img = gradient(8, 8)
        out = etc2.etc2_decode_level(etc2.etc2_encode_level(img, True), 8, 8, True)
        self.assertTrue((out[..., 3] == 255).all())

    def test_flat_colour_is_close(self):
        img = flat_image(8, 8, (102, 51, 204, 255))
        out = etc2.etc2_decode_level(etc2.etc2_encode_level(img, False), 8, 8, False)
        self.assertLessEqual(mean_error(out[..., :3], img[..., :3]), 4)

    def test_padding_of_odd_sizes(self):
        img = np.ascontiguousarray(gradient(32, 32)[:3, :5])  # a smooth 5x3 crop
        data = etc2.etc2_encode_level(img, False)
        self.assertEqual(len(data), 2 * 1 * 8)
        out = etc2.etc2_decode_level(data, 5, 3, False)
        self.assertEqual(out.shape, (3, 5, 4))
        self.assertLess(mean_error(out[..., :3], img[..., :3]), 7)
        self.assertEqual(len(etc2.etc2_encode_level(gradient(1, 1), True)), 16)

    def test_chunking_gives_identical_output(self):
        img = gradient(24, 16, alpha=True)
        want = etc2.etc2_encode_level(img, True)
        with mock.patch.object(etc2, "ETC2_CHUNK_BLOCKS", 5), mock.patch.object(etc2, "EAC_UNIQUE_CHUNK", 2):
            self.assertEqual(etc2.etc2_encode_level(img, True), want)

    def test_parallel_matches_serial(self):
        img = gradient(32, 24, alpha=True)
        serial_rgb, serial_rgba = etc2.etc2_encode_level(img, False), etc2.etc2_encode_level(img, True)
        with mock.patch.object(etc2, "PAR_MIN_ETC2_ENCODE_BLOCKS", 1), threaded_pool() as pool:
            self.assertEqual(etc2.etc2_encode_level(img, False, pool), serial_rgb)
            self.assertEqual(etc2.etc2_encode_level(img, True, pool), serial_rgba)

    def test_small_images_do_not_use_the_pool(self):
        with threaded_pool() as pool:
            etc2.etc2_encode_level(gradient(8, 8), False, pool)
            self.assertIsNone(pool.ex)

    def test_repeated_alpha_patterns_are_encoded_once(self):
        img = gradient(16, 16, alpha=True)
        img[..., 3] = np.tile(np.array([[10, 20, 30, 40]], np.uint8), (16, 4))
        blocks = img.reshape(4, 4, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(-1, 4, 4, 4)
        coded = etc2._eac_encode_vec(blocks)
        self.assertEqual(coded.shape, (16, 8))
        self.assertEqual(len({bytes(row) for row in coded.tolist()}), 1)
        decoded = etc2._eac_decode_vec(coded)  # decoded in row-major pixel order
        want = blocks[..., 3].reshape(16, 16)
        self.assertLessEqual(int(np.abs(decoded.astype(int) - want).max()), 2)

    def test_rgb_block_encoder_shapes(self):
        pix = np.array([[gradient(4, 4)[y, x] for x in range(4)] for y in range(4)], np.uint8)[None]
        out = etc2._etc_encode_rgb_vec(pix)
        self.assertEqual(out.shape, (1, 8))
        self.assertEqual((out[0, 3] >> 1) & 1, 0)  # individual mode: the diff bit stays clear


# --------------------------------------------------------------------- btx

KTX_ID = b"\xABKTX 11\xBB\r\n\x1A\n"


def ktx(internal, width=4, height=4, levels=(bytes(16),), gl_type=0, gl_format=0x1908,
        kv=b"", mips=None, prefix=b"", endian=0x04030201, depth=0, arrays=0, faces=1, tail=b""):
    """A KTX 1.1 file assembled by hand (independent of btx.build_btx)."""
    head = KTX_ID + struct.pack(
        "<13I", endian, gl_type, 1, gl_format, internal, internal, width, height,
        depth, arrays, faces, len(levels) if mips is None else mips, len(kv))
    body = b"".join(struct.pack("<I", len(v)) + v + b"\0" * ((4 - len(v) % 4) % 4) for v in levels)
    return prefix + head + kv + body + tail


ASTC_6X6 = 0x93B0 + btx.ASTC_BLOCKS.index((6, 6))


def btx_file(fmt="astc", width=16, height=16, alpha=False, block=(6, 6), mips=1, seed=0):
    """A real .btx made with the tool's own encoders."""
    png = make_png(width, height, alpha, seed)
    data, _ = btx.png_to_btx(png, block if fmt == "astc" else None, mips, fmt)
    return data


def only_chunks(png, **chunks):
    """Rebuild a PNG's pixels with just the given private chunks."""
    info = PngImagePlugin.PngInfo()
    for name, payload in chunks.items():
        info.add(name.encode(), payload)
    buf = io.BytesIO()
    Image.open(io.BytesIO(png)).convert("RGBA").save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def edit_pixel(png):
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    r, g, b, a = img.getpixel((0, 0))
    img.putpixel((0, 0), ((r + 100) % 256, g, b, a))
    buf = io.BytesIO()
    img.save(buf, "PNG", pnginfo=None)
    return buf.getvalue()


class TestParseBtx(unittest.TestCase):
    def test_astc(self):
        level = bytes(range(16))
        t = btx.parse_btx(ktx(ASTC_6X6, 6, 6, [level]))
        self.assertEqual((t.kind, t.block, t.srgb, t.mips), ("astc", (6, 6), False, 1))
        self.assertEqual((t.width, t.height, t.levels, t.prefix, t.tail), (6, 6, [level], b"", b""))
        self.assertEqual(t.internal, ASTC_6X6)

    def test_all_astc_block_sizes_and_srgb(self):
        for i, block in enumerate(btx.ASTC_BLOCKS):
            for base, srgb in ((btx.ASTC_UNORM_BASE, False), (btx.ASTC_SRGB_BASE, True)):
                with self.subTest(block=block, srgb=srgb):
                    t = btx.parse_btx(ktx(base + i))
                    self.assertEqual((t.block, t.srgb, t.kind), (block, srgb, "astc"))

    def test_just_outside_the_astc_ranges(self):
        for internal in (btx.ASTC_UNORM_BASE + len(btx.ASTC_BLOCKS), btx.ASTC_SRGB_BASE + len(btx.ASTC_BLOCKS),
                         btx.ASTC_UNORM_BASE - 1):
            with self.assertRaises(ValueError):
                btx.parse_btx(ktx(internal))

    def test_etc2_and_rgba8(self):
        self.assertEqual(btx.parse_btx(ktx(btx.ETC2_RGB8)).kind, "etc2")
        t = btx.parse_btx(ktx(btx.ETC2_RGBA8))
        self.assertEqual((t.kind, t.internal), ("etc2", btx.ETC2_RGBA8))
        t = btx.parse_btx(ktx(btx.RGBA8, gl_type=btx.GL_UNSIGNED_BYTE, gl_format=btx.GL_RGBA))
        self.assertEqual((t.kind, t.gl_type, t.gl_format), ("rgba8", 0x1401, 0x1908))

    def test_unsupported_format(self):
        with self.assertRaises(ValueError) as cm:
            btx.parse_btx(ktx(0x1234))
        self.assertIn("unsupported KTX format: internal=0x1234", str(cm.exception))
        with self.assertRaises(ValueError):  # RGBA8 but not unsigned byte / RGBA
            btx.parse_btx(ktx(btx.RGBA8, gl_type=0, gl_format=0))

    def test_stray_prefix(self):
        t = btx.parse_btx(ktx(ASTC_6X6, prefix=b"\0" * 16))
        self.assertEqual(t.prefix, b"\0" * 16)
        with self.assertRaises(ValueError) as cm:
            btx.parse_btx(ktx(ASTC_6X6, prefix=b"\0" * 17))
        self.assertIn("signature not found", str(cm.exception))
        with self.assertRaises(ValueError):
            btx.parse_btx(b"nothing like a ktx file")

    def test_header_problems(self):
        with self.assertRaises(ValueError) as cm:
            btx.parse_btx(KTX_ID + bytes(20))
        self.assertIn("truncated KTX header", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            btx.parse_btx(ktx(ASTC_6X6, endian=0x01020304))
        self.assertIn("big-endian", str(cm.exception))
        for kwargs in ({"depth": 2}, {"arrays": 1}, {"faces": 6}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError) as cm:
                    btx.parse_btx(ktx(ASTC_6X6, **kwargs))
                self.assertIn("only standard 2D", str(cm.exception))
        btx.parse_btx(ktx(ASTC_6X6, depth=1))

    def test_mip_chain_padding_kv_and_tail(self):
        levels = [b"A" * 16, b"B" * 7, b"C" * 5]
        data = ktx(ASTC_6X6, 12, 12, levels, kv=b"kv-data", tail=b"TAIL")
        t = btx.parse_btx(data)
        self.assertEqual((t.mips, t.levels, t.tail), (3, levels, b"TAIL"))
        self.assertEqual(t.header, data[:64 + len(b"kv-data")])

    def test_zero_mips_means_one(self):
        t = btx.parse_btx(ktx(ASTC_6X6, mips=0))
        self.assertEqual((t.mips, len(t.levels)), (1, 1))

    def test_truncated_mips(self):
        good = ktx(ASTC_6X6)
        with self.assertRaises(ValueError) as cm:
            btx.parse_btx(good[:64])
        self.assertIn("truncated mip header", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            btx.parse_btx(good[:-3])
        self.assertIn("truncated mip data", str(cm.exception))
        with self.assertRaises(ValueError):  # two mips declared, one present
            btx.parse_btx(ktx(ASTC_6X6, mips=2))


class TestBuildBtx(unittest.TestCase):
    def test_fresh_headers(self):
        lv = [bytes(range(16)), b"xyz"]
        for fmt, kwargs, internal in (
                ("astc", {"block": (8, 5)}, 0x93B0 + btx.ASTC_BLOCKS.index((8, 5))),
                ("astc", {"block": (4, 4), "srgb": True}, 0x93D0),
                ("etc2", {}, btx.ETC2_RGB8),
                ("etc2", {"etc2_alpha": True}, btx.ETC2_RGBA8),
                ("rgba8", {}, btx.RGBA8)):
            with self.subTest(fmt=fmt, kwargs=kwargs):
                out = btx.build_btx(None, 7, 9, lv, fmt, **kwargs)
                t = btx.parse_btx(out)
                self.assertEqual((t.width, t.height, t.mips, t.levels), (7, 9, 2, lv))
                self.assertEqual(t.internal, internal)
                self.assertEqual(t.prefix, b"\0" * 4)
                self.assertEqual(t.kind, fmt)

    def test_unknown_format(self):
        with self.assertRaises(ValueError) as cm:
            btx.build_btx(None, 4, 4, [bytes(16)], "dxt")
        self.assertIn("unknown BTX format: dxt", str(cm.exception))

    def test_same_kind_template_is_reused(self):
        template = btx.parse_btx(ktx(ASTC_6X6, 6, 6, kv=b"keep-me!", prefix=b"\1\2", tail=b"END!"))
        out = btx.build_btx(template, 12, 12, [bytes(32), bytes(16)], "astc", block=(4, 4))
        t = btx.parse_btx(out)
        self.assertEqual((t.width, t.height, t.mips, t.block), (12, 12, 2, (4, 4)))
        self.assertEqual((t.prefix, t.tail), (b"\1\2", b"END!"))
        self.assertIn(b"keep-me!", t.header)
        srgb = btx.build_btx(template, 6, 6, [bytes(16)], "astc", block=(6, 6), srgb=True)
        self.assertTrue(btx.parse_btx(srgb).srgb)

    def test_rgba8_template_keeps_its_format_words(self):
        template = btx.parse_btx(ktx(btx.RGBA8, 2, 2, [bytes(16)], btx.GL_UNSIGNED_BYTE, btx.GL_RGBA))
        out = btx.build_btx(template, 4, 4, [bytes(64)], "rgba8")
        t = btx.parse_btx(out)
        self.assertEqual((t.kind, t.width, t.height), ("rgba8", 4, 4))

    def test_etc2_template_switches_alpha(self):
        template = btx.parse_btx(ktx(btx.ETC2_RGB8, 4, 4, tail=b"T"))
        out = btx.build_btx(template, 4, 4, [bytes(16)], "etc2", etc2_alpha=True)
        t = btx.parse_btx(out)
        self.assertEqual((t.internal, t.tail), (btx.ETC2_RGBA8, b"T"))

    def test_other_kind_template_gives_a_fresh_header(self):
        template = btx.parse_btx(ktx(btx.ETC2_RGB8, 4, 4, kv=b"12345678", prefix=b"\x09", tail=b"TAIL"))
        out = btx.build_btx(template, 6, 6, [bytes(16)], "astc", block=(6, 6))
        t = btx.parse_btx(out)
        self.assertEqual((t.prefix, t.tail), (b"\0" * 4, b"TAIL"))
        self.assertEqual(len(t.header), 64)  # no key/value block carried over

    def test_tail_rules(self):
        astc_t = btx.parse_btx(ktx(ASTC_6X6, tail=b"TAIL"))
        etc2_t = btx.parse_btx(ktx(btx.ETC2_RGB8, tail=b"TAIL"))
        # astc/rgba8 output keeps whatever tail the template had; etc2 only from an etc2 template
        self.assertEqual(btx.parse_btx(btx.build_btx(etc2_t, 4, 4, [bytes(16)], "astc", block=(6, 6))).tail, b"TAIL")
        self.assertEqual(btx.parse_btx(btx.build_btx(astc_t, 4, 4, [bytes(16)], "rgba8")).tail, b"TAIL")
        self.assertEqual(btx.parse_btx(btx.build_btx(astc_t, 4, 4, [bytes(8)], "etc2")).tail, b"")
        self.assertEqual(btx.parse_btx(btx.build_btx(etc2_t, 4, 4, [bytes(8)], "etc2")).tail, b"TAIL")


class TestPngChunks(unittest.TestCase):
    def test_get_chunk(self):
        png = only_chunks(make_png(4, 4), btXt=b"payload", btXh=b"hash")
        self.assertEqual(btx.png_get_chunk(png), b"payload")
        self.assertEqual(btx.png_get_chunk(png, btx.PNG_HASH_CHUNK), b"hash")
        self.assertIsNone(btx.png_get_chunk(png, b"zzZz"))

    def test_get_chunk_on_junk(self):
        self.assertIsNone(btx.png_get_chunk(b""))
        self.assertIsNone(btx.png_get_chunk(b"\x89PNG\r\n\x1a\n" + b"\0" * 5))
        self.assertIsNone(btx.png_get_chunk(make_png(4, 4)))

    def test_digest(self):
        a = np.zeros((2, 3, 4), np.uint8)
        b = a.copy()
        b[1, 2, 3] = 1
        self.assertEqual(len(btx._pixel_digest(a)), 16)
        self.assertEqual(btx._pixel_digest(a), btx._pixel_digest(a.copy()))
        self.assertNotEqual(btx._pixel_digest(a), btx._pixel_digest(b))
        self.assertNotEqual(btx._pixel_digest(a), btx._pixel_digest(np.zeros((3, 2, 4), np.uint8)))


class TestDecodeLevel(unittest.TestCase):
    def test_astc_with_mip_sizes(self):
        data = btx_file("astc", 16, 16, mips=3)
        t = btx.parse_btx(data)
        for level, size in enumerate((16, 8, 4)):
            out = btx.btx_decode_level(t, level)
            self.assertEqual(out.shape, (size, size, 4))
            self.assertTrue(np.array_equal(out, astc.decode_level(t.levels[level], size, size, 6, 6)))

    def test_etc2_rgb_and_rgba(self):
        for alpha in (False, True):
            t = btx.parse_btx(btx_file("etc2", 12, 8, alpha=alpha))
            out = btx.btx_decode_level(t, 0)
            self.assertEqual(out.shape, (8, 12, 4))
            self.assertEqual(bool((out[..., 3] < 255).any()), alpha)

    def test_rgba8_is_exact_and_independent_of_the_source(self):
        png = make_png(8, 8, alpha=True)
        t = btx.parse_btx(btx.png_to_btx(png, fmt="rgba8")[0])
        out = btx.btx_decode_level(t, 0)
        self.assertEqual(out.tobytes(), png_pixels(png)[1])
        out[0, 0, 0] ^= 255  # a copy: the texture is untouched
        self.assertEqual(btx.btx_decode_level(t, 0).tobytes(), png_pixels(png)[1])

    def test_short_and_unknown(self):
        import dataclasses
        t = btx.parse_btx(ktx(btx.RGBA8, 4, 4, [bytes(10)], btx.GL_UNSIGNED_BYTE, btx.GL_RGBA))
        with self.assertRaises(ValueError) as cm:
            btx.btx_decode_level(t, 0)
        self.assertIn("RGBA8 mip is too short", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            btx.btx_decode_level(dataclasses.replace(t, kind="bogus"), 0)
        self.assertIn("unknown BTX type", str(cm.exception))

    def test_tiny_mips_never_reach_zero(self):
        t = btx.parse_btx(ktx(btx.RGBA8, 1, 1, [bytes(4), bytes(4)], btx.GL_UNSIGNED_BYTE, btx.GL_RGBA))
        self.assertEqual(btx.btx_decode_level(t, 1).shape, (1, 1, 4))


class TestBtxToPng(unittest.TestCase):
    def test_png_carries_the_original_and_a_hash(self):
        for fmt in ("astc", "etc2", "rgba8"):
            data = btx_file(fmt, 16, 12)
            png = btx.btx_to_png(data)
            with self.subTest(fmt=fmt):
                self.assertEqual(zlib.decompress(btx.png_get_chunk(png)), data)
                self.assertEqual(len(btx.png_get_chunk(png, btx.PNG_HASH_CHUNK)), 16)
                size, pixels = png_pixels(png)
                self.assertEqual(size, (16, 12))
                t = btx.parse_btx(data)
                self.assertEqual(pixels, btx.btx_decode_level(t, 0).tobytes())

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            btx.btx_to_png(b"not a ktx")

    def test_wrappers(self):
        data = btx_file("rgba8", 4, 4)
        self.assertEqual(btx.decode(data), btx.btx_to_png(data))
        png = make_png(8, 8)
        self.assertEqual(btx.encode(png, fmt="rgba8"), btx.png_to_btx(png, fmt="rgba8"))
        self.assertEqual(btx.encode(png, fmt="etc2", verify=False)[1], "encoded ETC2 RGB8 (mip=1)")


class TestPngToBtx(unittest.TestCase):
    def test_fresh_png_every_format(self):
        png = make_png(16, 12, seed=3)
        out, note = btx.png_to_btx(png)
        t = btx.parse_btx(out)
        self.assertEqual((t.kind, t.block, t.mips, t.srgb), ("astc", (6, 6), 1, False))
        self.assertEqual(note, "re-encoded (astc 6x6, mip=1)")
        out, note = btx.png_to_btx(png, fmt="etc2")
        self.assertEqual(note, "encoded ETC2 RGB8 (mip=1)")
        self.assertEqual(btx.parse_btx(out).internal, btx.ETC2_RGB8)
        out, note = btx.png_to_btx(png, fmt="rgba8")
        self.assertEqual(note, "re-encoded (rgba8, mip=1)")
        self.assertEqual(btx.btx_decode_level(btx.parse_btx(out), 0).tobytes(), png_pixels(png)[1])

    def test_block_size_and_mips(self):
        png = make_png(16, 16)
        out, note = btx.png_to_btx(png, (4, 4), 3)
        t = btx.parse_btx(out)
        self.assertEqual((t.block, t.mips), ((4, 4), 3))
        self.assertEqual([len(v) for v in t.levels], [256, 64, 16])  # 16, 4 and 1 blocks of 16 bytes
        self.assertEqual(note, "re-encoded (astc 4x4, mip=3)")

    def test_rgba8_mips_are_box_filtered(self):
        png = make_png(8, 8, alpha=True)
        t = btx.parse_btx(btx.png_to_btx(png, mips=3, fmt="rgba8")[0])
        self.assertEqual([len(v) for v in t.levels], [256, 64, 16])
        want = Image.open(io.BytesIO(png)).convert("RGBA").resize((4, 4), btx._BOX).tobytes()
        self.assertEqual(btx.btx_decode_level(t, 1).tobytes(), want)

    def test_etc2_alpha_is_detected(self):
        out, note = btx.png_to_btx(make_png(8, 8, alpha=True), fmt="etc2")
        self.assertEqual(note, "encoded ETC2 RGBA8 (mip=1)")
        self.assertEqual(btx.parse_btx(out).internal, btx.ETC2_RGBA8)

    def test_unknown_format_and_skipped_verify(self):
        with self.assertRaises(ValueError) as cm:
            btx.png_to_btx(make_png(4, 4), fmt="dxt")
        self.assertIn("unknown format: dxt", str(cm.exception))
        out, _ = btx.png_to_btx(make_png(8, 8), fmt="astc", verify=False)
        btx.parse_btx(out)

    def test_not_a_png(self):
        with self.assertRaises(Exception):
            btx.png_to_btx(b"definitely not a png")

    def test_unedited_png_restores_the_original_byte_for_byte(self):
        for fmt in ("astc", "etc2", "rgba8"):
            data = btx_file(fmt, 16, 16)
            with self.subTest(fmt=fmt):
                out, note = btx.png_to_btx(btx.btx_to_png(data), fmt=fmt)
                self.assertEqual((out, note), (data, "original restored 1:1"))

    def test_png_without_hash_is_compared_by_decoding(self):
        data = btx_file("astc", 16, 16)
        full = btx.btx_to_png(data)
        png = only_chunks(full, btXt=btx.png_get_chunk(full))  # the hash chunk is gone
        self.assertIsNone(btx.png_get_chunk(png, btx.PNG_HASH_CHUNK))
        self.assertEqual(btx.png_to_btx(png, fmt="astc"), (data, "original restored 1:1"))
        edited = only_chunks(edit_pixel(png), btXt=btx.png_get_chunk(png))
        out, note = btx.png_to_btx(edited, fmt="astc")
        self.assertNotEqual(out, data)
        self.assertTrue(note.startswith("re-encoded (astc 6x6"))

    def test_hash_mismatch_means_edited(self):
        data = btx_file("etc2", 16, 16)
        png = btx.btx_to_png(data)
        chunks = {"btXt": btx.png_get_chunk(png), "btXh": btx.png_get_chunk(png, btx.PNG_HASH_CHUNK)}
        out, note = btx.png_to_btx(only_chunks(edit_pixel(png), **chunks), fmt="etc2")
        self.assertEqual(note, "encoded ETC2 RGB8 (mip=1)")
        self.assertNotEqual(out, data)

    def test_template_decode_failure_falls_back_to_reencoding(self):
        # parses fine, but the single 16-byte level is too small for an 8x8 6x6-ASTC texture
        broken = ktx(ASTC_6X6, 8, 8, [bytes(16)])
        png = only_chunks(make_png(8, 8), btXt=zlib.compress(broken))
        out, note = btx.png_to_btx(png, fmt="astc")
        self.assertEqual(note, "re-encoded (astc 6x6, mip=1)")
        self.assertEqual(len(btx.parse_btx(out).levels[0]), 4 * 16)

    def test_bad_template_chunks_are_ignored(self):
        for payload in (b"not zlib at all", zlib.compress(b"zlib but not a ktx")):
            with self.subTest(payload=payload[:6]):
                png = only_chunks(make_png(8, 8), btXt=payload)
                out, note = btx.png_to_btx(png, fmt="astc")
                self.assertEqual(note, "re-encoded (astc 6x6, mip=1)")

    def test_template_of_another_size_is_ignored(self):
        small = btx_file("astc", 8, 8, mips=2)
        png = only_chunks(make_png(16, 16), btXt=zlib.compress(small))
        out, note = btx.png_to_btx(png, fmt="astc")
        self.assertEqual(btx.parse_btx(out).mips, 1)

    def test_edited_png_keeps_block_srgb_and_mips_of_the_template(self):
        pixels = make_png(16, 16)
        template = ktx(0x93D0 + btx.ASTC_BLOCKS.index((5, 5)), 16, 16, [astc.encode_level(
            np.array(Image.open(io.BytesIO(pixels)).convert("RGBA")), 5, 5)] * 2, kv=b"keepkeep")
        png = only_chunks(make_png(16, 16, seed=9), btXt=zlib.compress(template))
        out, note = btx.png_to_btx(png, fmt="astc")
        t = btx.parse_btx(out)
        self.assertEqual((t.block, t.srgb, t.mips), ((5, 5), True, 2))
        self.assertIn(b"keepkeep", t.header)
        self.assertEqual(note, "re-encoded (astc 5x5, mip=2)")
        out, _ = btx.png_to_btx(png, block=(8, 8), mips=1, fmt="astc")
        t = btx.parse_btx(out)
        self.assertEqual((t.block, t.mips, t.srgb), ((8, 8), 1, True))

    def test_edited_rgba8_png_keeps_mips(self):
        template = btx_file("rgba8", 8, 8, mips=2)
        png = only_chunks(make_png(8, 8, seed=4), btXt=zlib.compress(template))
        out, note = btx.png_to_btx(png, fmt="rgba8")
        self.assertEqual(btx.parse_btx(out).mips, 2)
        self.assertEqual(note, "re-encoded (rgba8, mip=2)")

    def test_edited_etc2_png_keeps_rgba_internal_format(self):
        template = btx_file("etc2", 8, 8, alpha=True)
        png = only_chunks(make_png(8, 8, seed=5), btXt=zlib.compress(template))  # opaque pixels now
        out, note = btx.png_to_btx(png, fmt="etc2")
        self.assertEqual(note, "encoded ETC2 RGBA8 (mip=1)")
        self.assertEqual(btx.parse_btx(out).internal, btx.ETC2_RGBA8)

    def test_other_format_than_the_template(self):
        template = btx_file("etc2", 8, 8, mips=2)
        png = only_chunks(make_png(8, 8), btXt=zlib.compress(template))
        out, note = btx.png_to_btx(png, fmt="astc")
        t = btx.parse_btx(out)
        self.assertEqual((t.kind, t.mips), ("astc", 1))  # the template's mips only count for its own kind
        out, _ = btx.png_to_btx(png, mips=2, fmt="astc")
        self.assertEqual(btx.parse_btx(out).mips, 2)


class TestBlockArgs(unittest.TestCase):
    def test_parse_block(self):
        self.assertEqual(btx.parse_block("6x6"), (6, 6))
        self.assertEqual(btx.parse_block("10X5"), (10, 5))
        for text, message in (("6", "invalid block size"), ("axb", "invalid block size"),
                              ("6x6x6", "invalid block size"), ("7x7", "unknown block size 7x7"),
                              ("", "invalid block size")):
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as cm:
                    btx.parse_block(text)
                self.assertIn(message, str(cm.exception))

    def test_block_arg(self):
        self.assertEqual(btx.block_arg("4x4"), (4, 4))
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            btx.block_arg("3x3")
        self.assertIn("unknown block size", str(cm.exception))


class TestBtxMain(TmpCase):
    def png(self, rel="a.png", **kwargs):
        return self.put(rel, make_png(**kwargs) if kwargs else make_png(16, 16))

    def test_png_to_btx_and_back(self):
        self.png("a.png")
        code, out, err = run_main(btx, self.path("a.png"), "--astc", "-j", 1)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("re-encoded (astc 6x6, mip=1)", out)
        self.assertEqual(btx.parse_btx(self.read("a.btx")).kind, "astc")
        os.remove(self.path("a.png"))
        self.assertEqual(run_main(btx, self.path("a.btx"), "-j", 1)[0], 0)
        self.assertEqual(png_pixels(self.read("a.png"))[0], (16, 16))

    def test_other_formats_and_options(self):
        self.png("a.png")
        self.assertEqual(run_main(btx, self.path("a.png"), "--etc2", "-o", self.path("e.btx"), "-j", 1)[0], 0)
        self.assertEqual(btx.parse_btx(self.read("e.btx")).kind, "etc2")
        self.assertEqual(run_main(btx, self.path("a.png"), "-r", "-m", 2, "-o", self.path("r.btx"))[0], 0)
        t = btx.parse_btx(self.read("r.btx"))
        self.assertEqual((t.kind, t.mips), ("rgba8", 2))
        self.assertEqual(run_main(btx, self.path("a.png"), "-a", "-b", "4x4", "--fast", "-o", self.path("f.btx"))[0], 0)
        self.assertEqual(btx.parse_btx(self.read("f.btx")).block, (4, 4))

    def test_argument_errors(self):
        self.png("a.png")
        src = self.path("a.png")
        cases = [
            ((src, "--etc2", "-b", "4x4"), "-b only goes with --astc"),
            ((src, "-b", "4x4"), "-b and -m need a format"),
            ((src, "-m", "2"), "-b and -m need a format"),
            ((src,), "PNG -> BTX needs a format"),
            ((src, "--astc", "-b", "9x9"), "unknown block size"),
            ((src, "--astc", "--etc2"), "not allowed"),
        ]
        for args, text in cases:
            with self.subTest(args=args[1:]):
                code, _, err = run_main(btx, *args)
                self.assertEqual(code, 2)
                self.assertIn(text, err)

    def test_folder_and_zip(self):
        self.put("d/a.png", make_png(8, 8))
        self.put("d/s/b.png", make_png(8, 8, seed=1))
        self.put("p.zip", make_zip({"in.png": make_png(8, 8), "keep.txt": b"k"}))
        code, out, _ = run_main(btx, self.path("d"), self.path("p.zip"), "--rgba8", "-j", 1)
        self.assertEqual(code, 0)
        self.assertIn("3 converted", out)
        self.assertTrue(self.exists("d/a.btx") and self.exists("d/s/b.btx"))
        with zipfile.ZipFile(self.path("p.zip")) as z:
            self.assertEqual(z.namelist(), ["in.btx", "keep.txt"])
        code, _, _ = run_main(btx, self.path("p.zip"), "-j", 1)
        self.assertEqual(code, 0)
        with zipfile.ZipFile(self.path("p.zip")) as z:
            self.assertEqual(z.namelist(), ["in.png", "keep.txt"])

    def test_broken_btx_is_reported(self):
        self.put("bad.btx", b"junk")
        code, _, err = run_main(btx, self.path("bad.btx"), "-j", 1)
        self.assertEqual(code, 1)
        self.assertIn("KTX signature not found", err)

    def test_worker_processes(self):
        for i in range(3):
            self.put(f"d/{i}.png", make_png(8, 8, seed=i))
        with mock.patch.dict(os.environ):
            code, out, err = run_main(btx, self.path("d"), "--astc", "-j", 2)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("3 converted", out)
        for i in range(3):
            self.assertEqual(btx.parse_btx(self.read(f"d/{i}.btx")).kind, "astc")

    def test_script_entry_point(self):
        self.png("a.png")
        code, _, _ = run_script("btx.py", self.path("a.png"), "--rgba8", "-j", 1)
        self.assertEqual(code, 0)
        self.assertTrue(self.exists("a.btx"))



# ------------------------------------------------------------------- badge

UNITTEST_OK = """test_a (mod.T.test_a) ... ok
test_b (mod.T.test_b) ... ok

----------------------------------------------------------------------
Ran 323 tests in 13.591s

OK
"""

COVERAGE_REPORT = """Name        Stmts   Miss  Cover   Missing
-----------------------------------------
astc.py       689      1    99%   351
btx.py        295      0   100%
-----------------------------------------
TOTAL        2202      1    99%
"""


class TestBadgeParsing(unittest.TestCase):
    def test_passed(self):
        self.assertEqual(badge.tests_badge(UNITTEST_OK), ("323 passed", badge.GREEN))

    def test_skipped_are_not_passed(self):
        text = "Ran 10 tests in 1s\n\nOK (skipped=3)\n"
        self.assertEqual(badge.tests_badge(text), ("7 passed, 3 skipped", badge.GREEN))

    def test_only_skipped_means_no_tests(self):
        self.assertEqual(badge.tests_badge("Ran 2 tests in 0s\n\nOK (skipped=2)\n"), ("no tests", badge.GREY))

    def test_failed_counts_failures_and_errors(self):
        text = "Ran 50 tests in 2s\n\nFAILED (failures=2, errors=1)\n"
        self.assertEqual(badge.tests_badge(text), ("3 failed", badge.RED))
        self.assertEqual(badge.tests_badge("Ran 5 tests in 1s\n\nFAILED (errors=4)\n"), ("4 failed", badge.RED))

    def test_the_last_run_wins(self):
        text = "Ran 3 tests in 1s\n\nFAILED (failures=1)\n\nRan 4 tests in 1s\n\nOK\n"
        self.assertEqual(badge.tests_badge(text), ("4 passed", badge.GREEN))

    def test_a_failing_test_line_is_not_the_verdict(self):
        text = "test_x (mod.T.test_x) ... FAIL\nRan 1 test in 0s\n\nFAILED (failures=1)\n"
        self.assertEqual(badge.tests_badge(text), ("1 failed", badge.RED))

    def test_unreadable_output(self):
        for text in ("", "Traceback (most recent call last):\nImportError: no numpy\n", "Ran 3 tests in 1s\n"):
            with self.subTest(text=text[:20]):
                self.assertEqual(badge.tests_badge(text), ("unknown", badge.GREY))

    def test_coverage_total(self):
        self.assertEqual(badge.coverage_badge(COVERAGE_REPORT), ("99%", badge.LIME))
        with_branches = "TOTAL     2202      1    400     12    99.95%\n"
        self.assertEqual(badge.coverage_badge(with_branches), ("99.95%", badge.LIME))

    def test_coverage_colours(self):
        for pct, colour in (("100", badge.GREEN), ("95", badge.LIME), ("90", badge.LIME), ("80", badge.YELLOW),
                            ("75", badge.YELLOW), ("60", badge.ORANGE), ("50", badge.ORANGE), ("49", badge.RED),
                            ("0", badge.RED)):
            with self.subTest(pct=pct):
                self.assertEqual(badge.coverage_badge(f"TOTAL   10   0   {pct}%\n"), (pct + "%", colour))

    def test_no_coverage_report(self):
        self.assertEqual(badge.coverage_badge(UNITTEST_OK), ("n/a", badge.GREY))
        self.assertEqual(badge.coverage_badge(""), ("n/a", badge.GREY))

    def test_svg(self):
        out = badge.svg("tests", "323 passed", badge.GREEN)
        self.assertTrue(out.startswith("<svg "))
        self.assertIn('aria-label="tests: 323 passed"', out)
        self.assertIn(f'fill="{badge.GREEN}"', out)
        self.assertIn(">323 passed</text>", out)
        width = int(out.split('width="')[1].split('"')[0])
        self.assertEqual(width, (10 + 7 * 5) + (10 + 7 * 10))

    def test_svg_escapes_text(self):
        out = badge.svg("a<b", "x & y", badge.GREY)
        self.assertIn("a&lt;b", out)
        self.assertIn("x &amp; y", out)
        self.assertNotIn("a<b", out)


class TestBadgeFiles(TmpCase):
    def test_read_utf8_bom_and_utf16(self):
        for name, raw in (("plain", UNITTEST_OK.encode()), ("bom", b"\xef\xbb\xbf" + UNITTEST_OK.encode()),
                          ("le", UNITTEST_OK.encode("utf-16")), ("be", b"\xfe\xff" + UNITTEST_OK.encode("utf-16-be"))):
            with self.subTest(name=name):
                self.assertEqual(badge.read(self.put(name + ".txt", raw)), UNITTEST_OK)

    def test_read_survives_bad_bytes(self):
        text = badge.read(self.put("odd.txt", b"Ran 1 test in 0s\n\xff\xfe junk\nOK\n"))
        self.assertIn("OK", text)

    def test_main_writes_both_badges_next_to_the_input(self):
        src = self.put("out/test_result.txt", (UNITTEST_OK + COVERAGE_REPORT).encode())
        code, out, err = run_main(badge, src)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("tests.svg: 323 passed", out)
        self.assertIn("coverage.svg: 99%", out)
        self.assertIn(">323 passed<", self.read("out/tests.svg").decode())
        self.assertIn(">99%<", self.read("out/coverage.svg").decode())

    def test_main_without_a_coverage_report(self):
        src = self.put("r.txt", UNITTEST_OK.encode())
        self.assertEqual(run_main(badge, src)[0], 0)
        self.assertIn(">n/a<", self.read("coverage.svg").decode())

    def test_main_other_folder_and_default_name(self):
        self.put("r.txt", UNITTEST_OK.encode())
        self.assertEqual(run_main(badge, self.path("r.txt"), "-o", self.path("docs", "img"))[0], 0)
        self.assertTrue(self.exists("docs/img/tests.svg") and self.exists("docs/img/coverage.svg"))
        self.assertFalse(self.exists("tests.svg"))
        self.put("test_result.txt", b"Ran 1 test in 0s\n\nFAILED (failures=1)\n")
        here = os.getcwd()
        try:
            os.chdir(self.tmp)
            code, out, _ = run_main(badge)
        finally:
            os.chdir(here)
        self.assertEqual(code, 0)
        self.assertIn("tests.svg: 1 failed", out)
        self.assertIn(badge.RED, self.read("tests.svg").decode())

    def test_missing_input(self):
        code, out, err = run_main(badge, self.path("nothing.txt"))
        self.assertEqual((code, out), (1, ""))
        self.assertIn("nothing.txt", err)
        self.assertEqual(os.listdir(self.tmp), [])

    def test_badges_from_a_real_run_of_unittest(self):
        """Feed it what unittest really prints, not just text we made up."""
        import subprocess
        code = ("import unittest\n"
                "class T(unittest.TestCase):\n"
                "    def test_a(self): pass\n"
                "    def test_b(self): self.assertEqual(1, 2)\n"
                "unittest.main(argv=['x', '-v'])\n")
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(badge.tests_badge(run.stderr), ("1 failed", badge.RED))

    def test_script_entry_point(self):
        src = self.put("r.txt", UNITTEST_OK.encode())
        code, out, _ = run_script("badge.py", src)
        self.assertEqual(code, 0)
        self.assertIn("tests.svg: 323 passed", out)


if __name__ == "__main__":
    unittest.main()
