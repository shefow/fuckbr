#!/usr/bin/env python3
"""ani.py - ANP3 .ani <-> .ifp animations (only the 36 byte header differs).

    ani.py walk.ani                walk.ani -> walk.ifp
    ani.py walk.ifp                walk.ifp -> walk.ani
    ani.py anims/ --ifp -o out/    folder: say the direction with --ifp or --ani

Folders are searched at any depth, by extension only. Without -o results
land next to the sources.
"""

import argparse
import struct
import sys

from common import expand, infer, run

MAGIC = b"ANP3"
HEADER = 36
TYPE = struct.pack("<I", 0x126)

# ani: 00 magic | 04 name (24) | 28 type | 32 size
# ifp: 00 magic | 04 size      | 08 name (24) | 32 type
ANI = dict(type=28, size=32, label="ANI")
IFP = dict(type=32, size=4, label="IFP")


def check(data, kind):
    if len(data) < HEADER:
        raise ValueError("file is too small")
    if data[:4] != MAGIC:
        raise ValueError("not an ANP3 file")
    if data[kind["type"]:kind["type"] + 4] != TYPE:
        raise ValueError(f"not an {kind['label']} file (wrong type field)")
    (size,) = struct.unpack_from("<I", data, kind["size"])
    if size != len(data) - 8:
        raise ValueError(f"wrong size field: header says {size}, file needs {len(data) - 8}")


def to_ifp(data, pool=None):
    check(data, ANI)
    out = bytearray(data)
    out[4:8] = data[32:36]
    out[8:32] = data[4:28]
    out[28:32] = bytes(4)  # the last 4 name bytes don't survive, same as always
    out[32:36] = data[28:32]
    return bytes(out)


def to_ani(data, pool=None):
    check(data, IFP)
    out = bytearray(data)
    out[4:28] = data[8:32]
    out[28:32] = data[32:36]
    out[32:36] = data[4:8]
    return bytes(out)


def main():
    ap = argparse.ArgumentParser(
        usage="ani.py INPUT... [--ifp | --ani] [-o OUT] [-j N]",
        description="ANP3 .ani <-> .ifp. The direction comes from the extension of plain "
                    "files; for folders give --ifp or --ani.")
    ap.add_argument("paths", nargs="+", metavar="INPUT")
    way = ap.add_mutually_exclusive_group()
    way.add_argument("--ifp", action="store_true", help="ANI -> IFP")
    way.add_argument("--ani", action="store_true", help="IFP -> ANI")
    ap.add_argument("-o", "--out", metavar="PATH", help="output file, or folder for several inputs")
    ap.add_argument("-j", "--jobs", type=int, default=0, metavar="N", help="workers, default: all cores")
    a = ap.parse_args()

    paths = expand(a.paths)
    src = ".ani" if a.ifp else ".ifp" if a.ani else infer(paths, (".ani", ".ifp"))
    if not src:
        ap.error("say which way: --ifp (ANI -> IFP) or --ani (IFP -> ANI)")
    rule = (".ani", ".ifp", to_ifp) if src == ".ani" else (".ifp", ".ani", to_ani)
    sys.exit(run(paths, rule, a.out, a.jobs, archives=False))


if __name__ == "__main__":
    main()
