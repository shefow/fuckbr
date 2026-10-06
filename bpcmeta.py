#!/usr/bin/env python3
"""bpcmeta.py - build .bpcmeta index files

    bpcmeta.py pack.bpc            -> pack.bpcmeta
    bpcmeta.py pack.zip            -> pack.bpcmeta
    bpcmeta.py folder/             -> folder.bpcmeta, for the folder packed with bpc.py -m
    bpcmeta.py a.bpc b.bpc -o metas/

The archive has to be stored (not compressed): bpc.py pack -m makes one.
"""

import argparse
import os
import struct
import sys
import zipfile
import zlib
from dataclasses import dataclass

import bpc
from common import Pool, cpus, expand, write

LOCAL_HEADER = 30
RECORD = struct.Struct("<IIBH")  # offset, size, tag, name length
COUNT = struct.Struct("<I")
TAGS = {".mp3": 1, ".wav": 0}


class MetaError(ValueError):
    pass


@dataclass(frozen=True)
class Entry:
    name: str
    offset: int
    size: int
    tag: int

    @property
    def header_size(self):
        return LOCAL_HEADER + len(self.name.encode("utf-8"))


def serialize(entries):
    parts = [COUNT.pack(len(entries))]
    for e in entries:
        raw = e.name.encode("utf-8")
        if len(raw) > 0xFFFF:
            raise MetaError(f"name too long: {e.name[:40]}...")
        for label, value in (("offset", e.offset), ("size", e.size)):
            if not 0 <= value <= 0xFFFFFFFF:
                raise MetaError(f"{e.name}: {label} {value} doesn't fit in 32 bits")
        parts += [RECORD.pack(e.offset, e.size, e.tag, len(raw)), raw]
    return b"".join(parts)


def parse(data):
    if len(data) < COUNT.size:
        raise MetaError("file too short")
    (count,) = COUNT.unpack_from(data, 0)
    entries, pos = [], COUNT.size
    for i in range(count):
        if pos + RECORD.size > len(data):
            raise MetaError(f"record {i}: truncated header at byte {pos}")
        offset, size, tag, length = RECORD.unpack_from(data, pos)
        pos += RECORD.size
        if pos + length > len(data):
            raise MetaError(f"record {i}: truncated name at byte {pos}")
        try:
            name = data[pos:pos + length].decode("utf-8")
        except UnicodeDecodeError as e:
            raise MetaError(f"record {i}: name is not UTF-8: {e}") from None
        pos += length
        entries.append(Entry(name, offset, size, tag))
    if pos != len(data):
        raise MetaError(f"{len(data) - pos} unexpected bytes after record {count - 1}")
    return entries


def tag_for(name):
    ext = os.path.splitext(name)[1].lower()
    if ext not in TAGS:
        raise MetaError(f"{name}: no tag for extension {ext or '(none)'!r}")
    return TAGS[ext]


def data_offset(f, info):
    """Where an entry's data really starts, read from its local header."""
    f.seek(info.header_offset)
    head = f.read(LOCAL_HEADER)
    if len(head) < LOCAL_HEADER or head[:4] != b"PK\x03\x04":
        raise MetaError(f"{info.filename}: bad local header at {info.header_offset}")
    name_len, extra_len = struct.unpack_from("<HH", head, 26)
    return info.header_offset + LOCAL_HEADER + name_len + extra_len


def entries_from_zip(opener, check=True, jobs=1):
    """opener() returns a fresh seekable file with the (decrypted) zip."""
    if check:
        try:
            bpc.validate(opener, jobs)
        except ValueError as e:
            raise MetaError(str(e)) from None
    entries = []
    with opener() as zf, opener() as f, zipfile.ZipFile(zf) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            if info.compress_type != zipfile.ZIP_STORED:
                raise MetaError(f"{info.filename}: compressed, pack with bpc.py -m")
            entries.append(Entry(info.filename, data_offset(f, info), info.file_size, tag_for(info.filename)))
    return entries


def scan(folder):
    """(relative path, size) of every file below folder, sorted by path."""
    found, stack = [], [(folder, "")]
    while stack:
        path, prefix = stack.pop()
        with os.scandir(path) as it:
            for item in it:
                if item.is_dir():
                    if not item.is_symlink():
                        stack.append((item.path, prefix + item.name + "/"))
                else:
                    found.append((prefix + item.name, item.stat().st_size))
    return sorted(found)


def entries_from_folder(folder):
    """The layout of folder packed as a stored zip without dir entries."""
    entries, cursor = [], 0
    for name, size in scan(folder):
        offset = cursor + LOCAL_HEADER + len(name.encode("utf-8"))
        entries.append(Entry(name, offset, size, tag_for(name)))
        cursor = offset + size
    return entries


def problems(entries):
    """Things wrong with the index itself."""
    found, seen, end = [], set(), 0
    for i, e in enumerate(entries):
        where = f"#{i} {e.name}"
        parts = e.name.split("/")
        if not e.name or e.name.startswith("/") or "\\" in e.name or "" in parts or ".." in parts:
            found.append(f"{where}: unsafe or empty path")
        if e.name in seen:
            found.append(f"{where}: duplicate name")
        seen.add(e.name)
        try:
            if e.tag != tag_for(e.name):
                found.append(f"{where}: tag {e.tag} doesn't match the extension")
        except MetaError as err:
            found.append(str(err))
        if e.offset != end + e.header_size:
            found.append(f"{where}: offset {e.offset}, expected {end + e.header_size} for a contiguous stored zip")
        end = e.offset + e.size
    names = [e.name for e in entries]
    if names != sorted(names):
        found.append("entries are not sorted by name")
    return found


def build(src, check=True, jobs=1):
    if os.path.isdir(src):
        entries = entries_from_folder(src)
    elif src.lower().endswith(".bpc"):
        entries = entries_from_zip(lambda: bpc.open_bpc(src), check, jobs)
    else:
        entries = entries_from_zip(lambda: bpc.open_zip(src), check, jobs)
    bad = problems(entries)
    if bad:
        raise MetaError("; ".join(bad[:3]))
    data = serialize(entries)
    if parse(data) != entries:
        raise MetaError("generated index failed the self check")
    return data, len(entries)


def main():
    ap = argparse.ArgumentParser(
        usage="bpcmeta.py SOURCE... [-o OUT] [-j N] [--fast]",
        description="Write a .bpcmeta next to each source: a .bpc (encrypted), a .zip, "
                    "or a folder to be packed with bpc.py -m.")
    ap.add_argument("paths", nargs="+", metavar="SOURCE")
    ap.add_argument("-o", "--out", metavar="PATH", help="output file, or folder for several sources")
    ap.add_argument("-j", "--jobs", type=int, default=0, metavar="N", help="threads, default: all cores")
    ap.add_argument("--fast", action="store_true", help="skip the CRC check of the archive")
    a = ap.parse_args()

    paths = expand(a.paths)
    jobs = a.jobs or min(cpus(), 8)
    tasks, taken, failed = [], set(), 0
    for p in paths:
        base = os.path.normpath(p)
        if base.lower().endswith((".bpc", ".zip")):
            base = base[:-4]
        dst = base + ".bpcmeta"
        if a.out:
            dst = a.out if len(paths) == 1 else os.path.join(a.out, os.path.basename(dst))
        if dst in taken:
            print(f"{p}: {dst} is already made from another source", file=sys.stderr)
            failed += 1
            continue
        taken.add(dst)
        tasks.append((p, dst))

    def work(src, dst):
        try:
            data, n = build(src, not a.fast, 1 if len(tasks) > 1 else jobs)
            write(dst, data)
            return True, f"{src} -> {dst} ({n} files)"
        except (OSError, ValueError, zipfile.BadZipFile, zlib.error) as e:
            return False, f"{src}: {e}"

    with Pool(jobs) as pool:
        for ok, msg in pool.map(work, tasks):
            print(msg, file=sys.stdout if ok else sys.stderr)
            failed += not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
