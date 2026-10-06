#!/usr/bin/env python3
"""bpc.py - .bpc archives (a zip XORed with a fixed key)

    bpc.py pack.bpc          unpack to pack/
    bpc.py folder/           pack to folder.bpc
    bpc.py pack.bpc -z       decrypt to pack.zip
    bpc.py pack.zip          encrypt to pack.bpc

The XOR is obfuscation, not security.
"""

import argparse
import io
import os
import sys
import zipfile
import zlib

from common import Pool, atomic, cpus, expand

KEY = b"1cK1a5UF2tU8*G2lW#&%"
KEY_LEN = len(KEY)
CHUNK = KEY_LEN * 52428  # about 1 MB, a multiple of the key so chunks stay aligned
BUF = 1 << 20
PAR_MIN = 4 << 20  # don't bother with threads for less than this many bytes

_tables = {}
_np = []  # [numpy or None], filled on first big buffer


def numpy():
    if not _np:
        try:
            import numpy as m
        except ImportError:
            m = None
        _np.append(m)
    return _np[0]


def xor(data, offset=0):
    """XOR data with the repeating key as if it started at `offset` in the file."""
    n = len(data)
    shift = offset % KEY_LEN
    np = numpy() if n >= 1 << 20 else None
    if np:
        key = np.frombuffer(KEY * (n // KEY_LEN + 2), np.uint8)[shift:shift + n]
        return np.bitwise_xor(np.frombuffer(data, np.uint8), key).tobytes()
    data = bytes(data)
    out = bytearray(n)
    for j in range(min(KEY_LEN, n)):  # one translate table per key byte
        k = KEY[(shift + j) % KEY_LEN]
        if k not in _tables:
            _tables[k] = bytes(b ^ k for b in range(256))
        out[j::KEY_LEN] = data[j::KEY_LEN].translate(_tables[k])
    return bytes(out)


class XorReader(io.RawIOBase):
    """Decrypts a file while it is read, so zipfile can use a .bpc directly."""

    def __init__(self, f):
        super().__init__()
        self.f = f

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.f.tell()

    def seek(self, offset, whence=io.SEEK_SET):
        return self.f.seek(offset, whence)

    def readinto(self, buf):
        pos = self.f.tell()
        chunk = self.f.read(len(buf))
        buf[:len(chunk)] = xor(chunk, pos)
        return len(chunk)

    def close(self):
        if not self.closed:
            self.f.close()
        super().close()


class XorWriter(io.RawIOBase):
    """Encrypts everything written to it (seekable, zipfile goes back to patch headers)."""

    def __init__(self, f):
        super().__init__()
        self.f = f

    def writable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.f.tell()

    def seek(self, offset, whence=io.SEEK_SET):
        return self.f.seek(offset, whence)

    def write(self, data):
        self.f.write(xor(data, self.f.tell()))
        return len(data)

    def flush(self):
        if not self.closed:
            self.f.flush()


def open_bpc(path):
    return io.BufferedReader(XorReader(open(path, "rb")), BUF)


def open_zip(path):
    return open(path, "rb")


def crypt(src, dst, check=None):
    """Encrypt or decrypt, it's the same thing. check(tmp) can veto the result."""
    if os.path.exists(dst) and os.path.samefile(src, dst):
        raise ValueError("output is the input file")
    with atomic(dst) as tmp:
        with open(src, "rb") as i, open(tmp, "wb") as o:
            pos = 0
            while chunk := i.read(CHUNK):
                o.write(xor(chunk, pos))
                pos += len(chunk)
        if check:
            check(tmp)


def validate(opener, jobs=1):
    """Read every entry of a zip so the CRCs get checked. opener() gives a fresh file."""
    try:
        with opener() as f, zipfile.ZipFile(f) as z:
            infos = z.infolist()
            if jobs < 2 or len(infos) < 2 or sum(i.file_size for i in infos) < PAR_MIN:
                if bad := z.testzip():
                    raise ValueError(f"bad zip entry: {bad}")
                return
    except zlib.error as e:
        raise ValueError(f"bad zip data: {e}") from None

    def check(idx):
        with opener() as f, zipfile.ZipFile(f) as z:
            entries = z.infolist()
            for k in idx:
                try:
                    with z.open(entries[k]) as s:
                        while s.read(BUF):
                            pass
                except (zipfile.BadZipFile, zlib.error):
                    return entries[k].filename

    with Pool(jobs) as pool:
        for bad in pool.map(check, [(range(k, len(infos), jobs),) for k in range(jobs)]):
            if bad:
                raise ValueError(f"bad zip entry: {bad}")


def unpack(src, folder, jobs=1, check=True):
    """Decrypt src and extract it into folder."""
    def opener():
        return open_bpc(src)

    if check:
        validate(opener, jobs)
    os.makedirs(folder, exist_ok=True)
    with opener() as f, zipfile.ZipFile(f) as z:
        infos = z.infolist()
        for i in infos:
            if i.is_dir():
                z.extract(i, folder)
    files = [k for k, i in enumerate(infos) if not i.is_dir()]

    def extract(idx):
        with opener() as f, zipfile.ZipFile(f) as z:
            entries = z.infolist()
            for k in idx:
                for attempt in (0, 1):
                    try:
                        z.extract(entries[k], folder)
                        break
                    except FileExistsError:  # two threads made the same folder
                        if attempt:
                            raise

    same_name = len({infos[k].filename for k in files}) < len(files)
    big = sum(infos[k].file_size for k in files) >= PAR_MIN
    if jobs < 2 or len(files) < 2 or same_name or not big:
        extract(files)
    else:
        with Pool(jobs) as pool:
            list(pool.map(extract, [(files[k::jobs],) for k in range(jobs)]))
    return len(infos)


def pack(folder, dst, meta=False):
    """Zip folder and encrypt it. meta: stored, no dir entries, sorted - what bpcmeta.py wants."""
    comp = zipfile.ZIP_STORED if meta else zipfile.ZIP_DEFLATED
    with atomic(dst) as tmp:
        with open(tmp, "wb") as f, io.BufferedWriter(XorWriter(f), BUF) as out:
            with zipfile.ZipFile(out, "w", comp) as z:
                if meta:
                    names = []
                    for root, _, files in os.walk(folder):
                        rel = os.path.relpath(root, folder)
                        names += [os.path.normpath(os.path.join(rel, n)).replace(os.sep, "/") for n in files]
                    for name in sorted(names):
                        z.write(os.path.join(folder, *name.split("/")), name)
                else:
                    for root, dirs, files in os.walk(folder):
                        dirs.sort()
                        rel = os.path.relpath(root, folder)
                        if rel != ".":
                            z.write(root, rel.replace(os.sep, "/") + "/")
                        for name in sorted(files):
                            path = os.path.join(root, name)
                            z.write(path, os.path.relpath(path, folder).replace(os.sep, "/"))


def main():
    ap = argparse.ArgumentParser(
        usage="bpc.py INPUT... [-z] [-m] [-o OUT] [-j N]",
        description="What happens depends on the input: a .bpc is unpacked, a folder is "
                    "packed, a .zip is encrypted. -z decrypts a .bpc to a zip instead.")
    ap.add_argument("paths", nargs="+", metavar="INPUT")
    ap.add_argument("-z", "--zip", action="store_true", help="decrypt a .bpc to a zip, don't unpack")
    ap.add_argument("-m", "--meta", action="store_true",
                    help="pack as stored zip, no dir entries, sorted (the layout bpcmeta.py expects)")
    ap.add_argument("-o", "--out", metavar="PATH", help="output path, or folder for several inputs")
    ap.add_argument("-j", "--jobs", type=int, default=0, metavar="N", help="threads, default: all cores")
    ap.add_argument("--fast", action="store_true", help="skip the zip integrity checks")
    a = ap.parse_args()

    jobs = a.jobs or min(cpus(), 8)
    paths = expand(a.paths)
    failed = 0
    for p in paths:
        low = p.lower()
        base = os.path.normpath(p)
        try:
            if os.path.isdir(p):
                dst = base + ".bpc"
            elif low.endswith(".bpc"):
                dst = base[:-4] + (".zip" if a.zip else "")
            elif low.endswith(".zip") and not a.zip:
                dst = base[:-4] + ".bpc"
            elif os.path.exists(p):
                raise ValueError("expected a folder, a .bpc or a .zip")
            else:
                raise FileNotFoundError("no such file or directory")
            if a.out:
                dst = a.out if len(paths) == 1 else os.path.join(a.out, os.path.basename(dst))

            if os.path.isdir(p):
                pack(p, dst, a.meta)
            elif low.endswith(".bpc") and not a.zip:
                unpack(p, dst, jobs, not a.fast)
            elif low.endswith(".bpc"):
                crypt(p, dst, None if a.fast else lambda tmp: validate(lambda: open_zip(tmp), jobs))
            else:
                if not a.fast:
                    validate(lambda: open_zip(p), jobs)
                crypt(p, dst, None if a.fast else lambda tmp: validate(lambda: open_bpc(tmp), jobs))
            print(f"{p} -> {dst}")
        except (OSError, ValueError, zipfile.BadZipFile, zlib.error) as e:
            print(f"{p}: {e}", file=sys.stderr)
            failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
