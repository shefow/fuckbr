#!/usr/bin/env python3
"""cls.py - COL1 collision files <-> CLS. Names and bounds only, no geometry.

    cls.py model.col               COL -> CLS
    cls.py model.cls               CLS -> COL
    cls.py models/ --cls -o out/   folder, zip or .bpc: say the direction with --cls or --col

Folders and zips (.bpc is just a renamed zip) are searched at any depth, by
extension only. Without -o results land next to the sources; a zip is
rewritten in place and the old one kept as FILE.bak.
"""

import argparse
import struct
import sys
from dataclasses import dataclass
from functools import partial

from common import expand, infer, run

COL_TAG = b"COLL"
COL_NAME_SIZE = 22
COL_HEAD = struct.Struct("<4sI")
COL_ID = struct.Struct("<H")
COL_BOUNDS = struct.Struct("<10f")
COL_COUNT = struct.Struct("<I")
COL_SPHERE_SIZE = 20
COL_BOX_SIZE = 28
COL_VERTEX_SIZE = 12
COL_FACE_SIZE = 16

CLS_TAG = b"CLST"
CLS_TYPE = b"CED2"
CLS_NAME_SIZE = 20
CLS_HEAD = struct.Struct("<4sI")
CLS_BOUNDS = struct.Struct("<10f")
CLS_TAIL_SIZE = 48
CLS_BODY_SIZE = CLS_NAME_SIZE + len(CLS_TYPE) + CLS_BOUNDS.size + CLS_TAIL_SIZE


class ClsError(ValueError):
    pass


@dataclass(frozen=True)
class Model:
    name: str
    center: tuple
    radius: float
    low: tuple
    high: tuple
    solid: bool = False  # has geometry we would have to drop


def decode_name(raw):
    try:
        return raw.split(b"\0", 1)[0].decode("ascii")
    except UnicodeDecodeError as e:
        raise ClsError(f"name is not ASCII: {e}") from None


def encode_name(name, size):
    raw = name.encode("ascii", "strict")
    if len(raw) >= size:
        raise ClsError(f"name too long ({len(raw)} bytes, max {size - 1}): {name}")
    return raw.ljust(size, b"\0")


def skip_array(data, pos, limit, item_size, label):
    """Skip a counted array, return (position after it, item count)."""
    if pos + COL_COUNT.size > limit:
        raise ClsError(f"truncated {label} count at byte {pos}")
    (count,) = COL_COUNT.unpack_from(data, pos)
    end = pos + COL_COUNT.size + count * item_size
    if end > limit:
        raise ClsError(f"truncated {label} array at byte {pos}")
    return end, count


def parse_col(data):
    models = []
    pos = 0
    while pos < len(data):
        if pos + COL_HEAD.size > len(data):
            raise ClsError(f"model {len(models)}: truncated header at byte {pos}")
        tag, size = COL_HEAD.unpack_from(data, pos)
        if tag != COL_TAG:
            raise ClsError(f"model {len(models)}: unsupported tag {tag!r} (only COL1)")
        start = pos + COL_HEAD.size
        end = start + size
        fixed = COL_NAME_SIZE + COL_ID.size + COL_BOUNDS.size
        if end > len(data) or size < fixed:
            raise ClsError(f"model {len(models)}: bad size {size}")

        name = decode_name(data[start:start + COL_NAME_SIZE])
        radius, cx, cy, cz, x0, y0, z0, x1, y1, z1 = COL_BOUNDS.unpack_from(
            data, start + COL_NAME_SIZE + COL_ID.size)

        cursor, solid = start + fixed, False
        for item_size, label in ((COL_SPHERE_SIZE, "sphere"), (0, "unknown"), (COL_BOX_SIZE, "box"),
                                 (COL_VERTEX_SIZE, "vertex"), (COL_FACE_SIZE, "face")):
            cursor, count = skip_array(data, cursor, end, item_size, label)
            solid = solid or (count > 0 and label != "unknown")
        if cursor != end:
            raise ClsError(f"model {len(models)}: {end - cursor} unexpected trailing bytes")

        models.append(Model(name, (cx, cy, cz), radius, (x0, y0, z0), (x1, y1, z1), solid))
        pos = end
    return models


def serialize_col(models):
    parts = []
    for m in models:
        body = b"".join((
            encode_name(m.name, COL_NAME_SIZE),
            COL_ID.pack(0),
            COL_BOUNDS.pack(m.radius, *m.center, *m.low, *m.high),
            COL_COUNT.pack(0) * 5,
        ))
        parts += [COL_HEAD.pack(COL_TAG, len(body)), body]
    return b"".join(parts)


def parse_cls(data):
    models = []
    pos = 0
    while pos < len(data):
        i = len(models)
        if pos + CLS_HEAD.size > len(data):
            raise ClsError(f"record {i}: truncated header at byte {pos}")
        tag, size = CLS_HEAD.unpack_from(data, pos)
        if tag != CLS_TAG:
            raise ClsError(f"record {i}: unexpected tag {tag!r} at byte {pos}")
        if size != CLS_BODY_SIZE:
            raise ClsError(f"record {i}: unsupported size {size} (expected {CLS_BODY_SIZE})")
        start = pos + CLS_HEAD.size
        end = start + size
        if end > len(data):
            raise ClsError(f"record {i}: truncated body at byte {start}")

        body = data[start:end]
        kind = body[CLS_NAME_SIZE:CLS_NAME_SIZE + len(CLS_TYPE)]
        if kind != CLS_TYPE:
            raise ClsError(f"record {i}: unsupported type {kind!r}")
        if any(body[-CLS_TAIL_SIZE:]):
            raise ClsError(f"record {i}: geometry data is not supported")

        name = decode_name(body[:CLS_NAME_SIZE])
        cx, cy, cz, radius, x0, y0, z0, x1, y1, z1 = CLS_BOUNDS.unpack_from(
            body, CLS_NAME_SIZE + len(CLS_TYPE))
        models.append(Model(name, (cx, cy, cz), radius, (x0, y0, z0), (x1, y1, z1)))
        pos = end
    return models


def serialize_cls(models):
    parts = []
    for m in models:
        body = b"".join((
            encode_name(m.name, CLS_NAME_SIZE),
            CLS_TYPE,
            CLS_BOUNDS.pack(*m.center, m.radius, *m.low, *m.high),
            b"\0" * CLS_TAIL_SIZE,
        ))
        parts += [CLS_HEAD.pack(CLS_TAG, len(body)), body]
    return b"".join(parts)


def to_cls(data, pool=None, bounds_only=False):
    models = parse_col(data)
    solid = [m.name for m in models if m.solid]
    if solid and not bounds_only:
        raise ClsError(f"{len(solid)} model(s) have geometry CLS can't hold yet "
                       f"(-b drops it): {', '.join(solid)[:80]}")
    note = f"{len(models)} models" + (f", geometry dropped in {len(solid)}" if solid else "")
    return serialize_cls(models), note


def to_col(data, pool=None):
    models = parse_cls(data)
    return serialize_col(models), f"{len(models)} models"


def main():
    ap = argparse.ArgumentParser(
        usage="cls.py INPUT... [--cls | --col] [-b] [-o OUT] [-j N]",
        description="COL1 <-> CLS, names and bounds only. The direction comes from the "
                    "extension of plain files; for folders and zips give --cls or --col.",
        epilog="INPUT is a file, a folder, a zip or a .bpc (a renamed, not encrypted zip).")
    ap.add_argument("paths", nargs="+", metavar="INPUT")
    way = ap.add_mutually_exclusive_group()
    way.add_argument("--cls", action="store_true", help="COL -> CLS")
    way.add_argument("--col", action="store_true", help="CLS -> COL")
    ap.add_argument("-b", "--bounds-only", action="store_true",
                    help="with --cls: drop spheres, boxes and meshes instead of failing")
    ap.add_argument("-o", "--out", metavar="PATH", help="output file, or folder for several inputs")
    ap.add_argument("-j", "--jobs", type=int, default=0, metavar="N", help="workers, default: all cores")
    a = ap.parse_args()

    paths = expand(a.paths)
    src = ".col" if a.cls else ".cls" if a.col else infer(paths, (".col", ".cls"))
    if not src:
        ap.error("say which way: --cls (COL -> CLS) or --col (CLS -> COL)")
    if a.bounds_only and src != ".col":
        ap.error("-b goes with --cls")
    if src == ".col":
        rule = (".col", ".cls", partial(to_cls, bounds_only=a.bounds_only))
    else:
        rule = (".cls", ".col", to_col)
    sys.exit(run(paths, rule, a.out, a.jobs))


if __name__ == "__main__":
    main()
