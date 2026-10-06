#!/usr/bin/env python3
"""btx.py - KTX 1.1 .btx textures (ASTC / ETC2 / RGBA8) to PNG and back.

    btx.py tex.btx                 BTX -> PNG
    btx.py tex.png --astc          PNG -> BTX, ASTC (6x6, -b 4x4 for another block)
    btx.py tex.png --etc2          PNG -> BTX, ETC2
    btx.py textures/ --etc2 -o out/  a folder, a zip or a .bpc works as a root too

Folders and zips (.bpc is just a renamed zip) are searched at any depth, by
extension only. Without -o results land next to the sources; a zip is
rewritten in place and the old one kept as FILE.bak.
"""

import argparse
import hashlib
import io
import os
import struct
import sys
import zlib
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Sequence, Tuple

from PIL import Image, PngImagePlugin

from arr import np
from astc import Block, decode_level, encode_level
from common import ARCHIVES, expand, run
from etc2 import etc2_decode_level, etc2_encode_level

PNG_CHUNK = b"btXt"       # the original btx, stored inside the png
PNG_HASH_CHUNK = b"btXh"  # hash of its pixels, to spot an unedited png
ETC2_RGB8 = 0x9274
ETC2_RGBA8 = 0x9278
RGBA8 = 0x8058
KTX_ID = b"\xABKTX 11\xBB\r\n\x1A\n"
KTX_ENDIANNESS = 0x04030201
GL_UNSIGNED_BYTE = 0x1401
GL_RGBA = 0x1908
FORMATS = ("astc", "etc2", "rgba8")

# pillow >= 9.1 moved the filters to Image.Resampling
_BOX = getattr(getattr(Image, "Resampling", Image), "BOX")


ASTC_BLOCKS: List[Block] = [
    (4, 4),
    (5, 4),
    (5, 5),
    (6, 5),
    (6, 6),
    (8, 5),
    (8, 6),
    (8, 8),
    (10, 5),
    (10, 6),
    (10, 8),
    (10, 10),
    (12, 10),
    (12, 12),
]
ASTC_UNORM_BASE = 0x93B0
ASTC_SRGB_BASE = 0x93D0

_OFF_INTERNAL_FORMAT = 28
_OFF_BASE_INTERNAL_FORMAT = 32
_OFF_WIDTH = 36
_OFF_MIP_COUNT = 56

@dataclass(frozen=True)
class Texture:
    """A parsed KTX 1.1 texture."""

    prefix: bytes
    header: bytes
    width: int
    height: int
    block: Block
    srgb: bool
    mips: int
    levels: List[bytes]
    tail: bytes
    kind: str
    internal: int
    gl_type: int
    gl_format: int

def parse_btx(data: bytes) -> Texture:
    """Parse a KTX 1.1 file (optionally preceded by up to 16 stray bytes)."""
    p = data.find(KTX_ID)
    if p < 0 or p > 16:
        raise ValueError("KTX signature not found")
    if len(data) < p + 64:
        raise ValueError("truncated KTX header")

    (endian, gl_type, _, gl_format, internal, _, width, height, depth, arrays,
     faces, mips, kv_size) = struct.unpack("<13I", data[p + 12 : p + 64])
    if endian != KTX_ENDIANNESS:
        raise ValueError("big-endian is not supported")
    if depth not in (0, 1) or arrays != 0 or faces != 1:
        raise ValueError("only standard 2D KTX textures are supported")

    srgb = False
    block: Block = (4, 4)
    if ASTC_UNORM_BASE <= internal < ASTC_UNORM_BASE + len(ASTC_BLOCKS):
        kind = "astc"
        block = ASTC_BLOCKS[internal - ASTC_UNORM_BASE]
    elif ASTC_SRGB_BASE <= internal < ASTC_SRGB_BASE + len(ASTC_BLOCKS):
        kind = "astc"
        block = ASTC_BLOCKS[internal - ASTC_SRGB_BASE]
        srgb = True
    elif internal in (ETC2_RGB8, ETC2_RGBA8):
        kind = "etc2"
    elif internal == RGBA8 and gl_type == GL_UNSIGNED_BYTE and gl_format == GL_RGBA:
        kind = "rgba8"
    else:
        raise ValueError(
            f"unsupported KTX format: internal={internal:#x}, "
            f"type={gl_type:#x}, format={gl_format:#x}"
        )

    offset = p + 64 + kv_size
    levels = []
    for _ in range(max(mips, 1)):
        if offset + 4 > len(data):
            raise ValueError("truncated mip header")
        (size,) = struct.unpack("<I", data[offset : offset + 4])
        if offset + 4 + size > len(data):
            raise ValueError("truncated mip data")
        levels.append(data[offset + 4 : offset + 4 + size])
        offset += 4 + size + ((4 - (size & 3)) & 3)

    return Texture(
        prefix=data[:p],
        header=data[p : p + 64 + kv_size],
        width=width,
        height=height,
        block=block,
        srgb=srgb,
        mips=max(mips, 1),
        levels=levels,
        tail=data[offset:],
        kind=kind,
        internal=internal,
        gl_type=gl_type,
        gl_format=gl_format,
    )

def build_btx(
    template: Optional[Texture],
    width: int,
    height: int,
    levels: Sequence[bytes],
    fmt: str,
    *,
    block: Block = (4, 4),
    srgb: bool = False,
    etc2_alpha: bool = False,
) -> bytes:
    """Assemble a KTX file, reusing the prefix/header/tail of ``template``.

    The template's header is only reused when it describes the same texture
    kind as ``fmt``; otherwise a fresh header is written.
    """
    if fmt == "astc":
        internal = (ASTC_SRGB_BASE if srgb else ASTC_UNORM_BASE) + ASTC_BLOCKS.index(block)
        gl_type, gl_format = 0, GL_RGBA
    elif fmt == "etc2":
        internal = ETC2_RGBA8 if etc2_alpha else ETC2_RGB8
        gl_type, gl_format = 0, 0
    elif fmt == "rgba8":
        internal = RGBA8
        gl_type, gl_format = GL_UNSIGNED_BYTE, GL_RGBA
    else:
        raise ValueError(f"unknown BTX format: {fmt}")

    if template is not None and template.kind == fmt:
        header = bytearray(template.header)
        if fmt != "rgba8":
            struct.pack_into("<I", header, _OFF_INTERNAL_FORMAT, internal)
            struct.pack_into("<I", header, _OFF_BASE_INTERNAL_FORMAT, internal)
        struct.pack_into("<II", header, _OFF_WIDTH, width, height)
        struct.pack_into("<I", header, _OFF_MIP_COUNT, len(levels))
        prefix, header = template.prefix, bytes(header)
    else:
        prefix = b"\x00" * 4
        header = KTX_ID + struct.pack(
            "<13I",
            KTX_ENDIANNESS, gl_type, 1, gl_format, internal, internal,
            width, height, 0, 0, 1, len(levels), 0,
        )

    body = b"".join(
        struct.pack("<I", len(level)) + level + b"\x00" * ((4 - (len(level) & 3)) & 3)
        for level in levels
    )

    keep_tail = template is not None and (fmt != "etc2" or template.kind == "etc2")
    return prefix + header + body + (template.tail if keep_tail else b"")

def png_get_chunk(png: bytes, name: bytes = PNG_CHUNK) -> Optional[bytes]:
    """Return the payload of the first PNG chunk called ``name``, if any."""
    pos = 8
    while pos + 8 <= len(png):
        length, chunk_type = struct.unpack(">I4s", png[pos : pos + 8])
        if chunk_type == name:
            return png[pos + 8 : pos + 8 + length]
        pos += 12 + length
    return None

def btx_decode_level(texture: Texture, level: int, pool=None):
    """Decode one mip level of ``texture`` to an (H, W, 4) uint8 array."""
    width = max(1, texture.width >> level)
    height = max(1, texture.height >> level)
    data = texture.levels[level]

    if texture.kind == "astc":
        return decode_level(data, width, height, *texture.block, pool=pool)
    if texture.kind == "etc2":
        return etc2_decode_level(data, width, height, texture.internal == ETC2_RGBA8)
    if texture.kind == "rgba8":
        need = width * height * 4
        if len(data) < need:
            raise ValueError("RGBA8 mip is too short")
        return np.frombuffer(data[:need], np.uint8).reshape(height, width, 4).copy()
    raise ValueError("unknown BTX type")

def _pixel_digest(rgba) -> bytes:
    """Digest of decoded pixels, used to detect an unedited PNG without decoding."""
    h, w = rgba.shape[:2]
    digest = hashlib.blake2b(digest_size=16)
    digest.update(struct.pack("<II", w, h))
    digest.update(np.ascontiguousarray(rgba).tobytes())
    return digest.digest()

def btx_to_png(data: bytes, pool=None) -> bytes:
    """Decode a BTX to PNG, embedding the original BTX in a ``btXt`` chunk."""
    texture = parse_btx(data)
    rgba = btx_decode_level(texture, 0, pool)

    info = PngImagePlugin.PngInfo()
    info.add(PNG_CHUNK, zlib.compress(data, 6))
    info.add(PNG_HASH_CHUNK, _pixel_digest(rgba))
    buffer = io.BytesIO()
    Image.fromarray(rgba).save(buffer, "PNG", pnginfo=info)
    return buffer.getvalue()

def png_to_btx(
    png: bytes,
    block: Optional[Block] = None,
    mips: Optional[int] = None,
    fmt: str = "astc",
    verify: bool = True,
    pool=None,
) -> Tuple[bytes, str]:
    """Convert PNG bytes to BTX bytes. Returns (btx, human-readable summary)."""
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    rgba = np.asarray(img)

    template: Optional[Texture] = None
    original = b""
    chunk = png_get_chunk(png)
    if chunk is not None:
        try:
            original = zlib.decompress(chunk)
            template = parse_btx(original)
        except Exception:
            template = None

    srgb = False
    if template is not None and (template.width, template.height) == img.size:
        if template.kind == fmt:
            stored = png_get_chunk(png, PNG_HASH_CHUNK)
            if stored is not None:
                # PNG written by this tool: compare digests, no decoding needed.
                if stored == _pixel_digest(rgba):
                    return original, "original restored 1:1"
            else:
                try:
                    if np.array_equal(btx_decode_level(template, 0, pool), rgba):
                        return original, "original restored 1:1"
                except Exception:
                    pass
            if fmt == "astc":
                block = template.block if block is None else block
                srgb = template.srgb
            if mips is None:
                mips = template.mips
            if fmt == "rgba8":
                block = (4, 4)
        elif mips is None:
            mips = 1

    block = block or (6, 6)
    mips = mips or 1

    has_alpha = bool((rgba[..., 3] < 255).any())
    etc2_alpha = has_alpha or (
        template is not None and template.kind == "etc2" and template.internal == ETC2_RGBA8
    )

    width, height = img.size
    levels = []
    for level in range(mips):
        lw, lh = max(1, width >> level), max(1, height >> level)
        pixels = rgba if level == 0 else np.asarray(img.resize((lw, lh), _BOX))
        pixels = np.ascontiguousarray(pixels)
        if fmt == "astc":
            levels.append(encode_level(pixels, *block, pool=pool))
        elif fmt == "etc2":
            levels.append(etc2_encode_level(pixels, etc2_alpha, pool))
        elif fmt == "rgba8":
            levels.append(pixels.tobytes())
        else:
            raise ValueError(f"unknown format: {fmt}")

    out = build_btx(
        template, width, height, levels, fmt,
        block=block, srgb=srgb, etc2_alpha=etc2_alpha,
    )
    checked = parse_btx(out)
    if verify:
        for level in range(len(checked.levels)):
            btx_decode_level(checked, level, pool)

    if fmt == "etc2":
        kind = "RGBA8" if etc2_alpha else "RGB8"
        return out, f"encoded ETC2 {kind} (mip={mips})"
    detail = f" {block[0]}x{block[1]}" if fmt == "astc" else ""
    return out, f"re-encoded ({fmt}{detail}, mip={mips})"

def parse_block(text: str) -> Block:
    """Parse ``"6x6"`` into ``(6, 6)``, validating against known ASTC sizes."""
    try:
        width, height = text.lower().split("x")
        block = (int(width), int(height))
    except ValueError:
        raise ValueError(f"invalid block size {text!r} (expected WxH)") from None
    if block not in ASTC_BLOCKS:
        raise ValueError(f"unknown block size {text}")
    return block

def decode(data, pool=None):
    return btx_to_png(data, pool)


def encode(data, pool=None, fmt="astc", block=None, mips=None, verify=True):
    return png_to_btx(data, block, mips, fmt, verify, pool)


def block_arg(text):
    try:
        return parse_block(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def main():
    ap = argparse.ArgumentParser(
        usage="btx.py INPUT... [--astc | --etc2 | --rgba8] [-b WxH] [-m N] [-o OUT] [-j N]",
        description="KTX 1.1 .btx <-> PNG. No format flag: BTX -> PNG. "
                    "A format flag: PNG -> BTX (mandatory then).",
        epilog="INPUT is a file, a folder, a zip or a .bpc (a renamed, not encrypted zip). "
               "Env: BTX_ARRAY_BACKEND=auto|rsnumpy|numpy.")
    ap.add_argument("paths", nargs="+", metavar="INPUT")
    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument("-a", "--astc", dest="fmt", action="store_const", const="astc", help="PNG -> ASTC")
    fmt.add_argument("-e", "--etc2", dest="fmt", action="store_const", const="etc2", help="PNG -> ETC2")
    fmt.add_argument("-r", "--rgba8", dest="fmt", action="store_const", const="rgba8", help="PNG -> RGBA8")
    ap.add_argument("-b", "--block", type=block_arg, metavar="WxH", help="ASTC block size, default 6x6")
    ap.add_argument("-m", "--mips", type=int, metavar="N", help="mip levels")
    ap.add_argument("-o", "--out", metavar="PATH", help="output file, or folder for several inputs")
    ap.add_argument("-j", "--jobs", type=int, default=0, metavar="N", help="workers, default: all cores")
    ap.add_argument("--fast", action="store_true", help="skip the self check after encoding")
    a = ap.parse_args()

    paths = expand(a.paths)
    if a.fmt:
        if a.block and a.fmt != "astc":
            ap.error("-b only goes with --astc")
        rule = (".png", ".btx", partial(encode, fmt=a.fmt, block=a.block, mips=a.mips, verify=not a.fast))
    else:
        if a.block or a.mips:
            ap.error("-b and -m need a format: --astc, --etc2 or --rgba8")
        if any(p.lower().endswith(".png") for p in paths):
            ap.error("PNG -> BTX needs a format: --astc, --etc2 or --rgba8")
        rule = (".btx", ".png", decode)
    # processes: the codecs are pure python, threads would not help
    sys.exit(run(paths, rule, a.out, a.jobs, procs=True, env={"BTX_ARRAY_THREADS": "1"}))


if __name__ == "__main__":
    main()
