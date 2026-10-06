"""ETC2 RGB8 / RGBA8 (EAC alpha): scalar reference decoder, vectorised
decoder and encoder."""

from typing import Dict, List, Tuple

from arr import np
from common import bands

PAR_MIN_ETC2_ENCODE_BLOCKS = 4096
ETC2_CHUNK_BLOCKS = 16384

ETC_MODIFIERS = (
    (-8, -2, 2, 8),
    (-17, -5, 5, 17),
    (-29, -9, 9, 29),
    (-42, -13, 13, 42),
    (-60, -18, 18, 60),
    (-80, -24, 24, 80),
    (-106, -33, 33, 106),
    (-183, -47, 47, 183),
)
ETC_DIST = (3, 6, 11, 16, 23, 32, 41, 64)
EAC_MODIFIERS = (
    (-3, -6, -9, -15, 2, 5, 8, 14),
    (-3, -7, -10, -13, 2, 6, 9, 12),
    (-2, -5, -8, -13, 1, 4, 7, 12),
    (-2, -4, -6, -13, 1, 3, 5, 12),
    (-3, -6, -8, -12, 2, 5, 7, 11),
    (-3, -7, -9, -11, 2, 6, 8, 10),
    (-4, -7, -8, -11, 3, 6, 7, 10),
    (-3, -5, -8, -11, 2, 4, 7, 10),
    (-2, -6, -8, -10, 1, 5, 7, 9),
    (-2, -5, -8, -10, 1, 4, 7, 9),
    (-2, -4, -8, -10, 1, 3, 7, 9),
    (-2, -5, -7, -10, 1, 4, 6, 9),
    (-3, -4, -7, -10, 2, 3, 6, 9),
    (-1, -2, -3, -10, 0, 1, 2, 9),
    (-4, -6, -8, -9, 3, 5, 7, 8),
    (-3, -5, -7, -9, 2, 4, 6, 8),
)
ETC_INDEX_MAP = (0, 2, 3, 1)
ETC_INDEX_MAP_INVERSE = {decoded: raw for raw, decoded in enumerate(ETC_INDEX_MAP)}

def _expand4(v: int) -> int:
    return (v << 4) | v

def _expand5(v: int) -> int:
    return (v << 3) | (v >> 2)

def _expand6(v: int) -> int:
    return (v << 2) | (v >> 4)

def _sign3(v: int) -> int:
    return v - 8 if v & 4 else v

def _clamp255(v: int) -> int:
    return 0 if v < 0 else 255 if v > 255 else v

def _etc_indices(block: bytes) -> List[int]:
    """Per-pixel modifier indices of an ETC block, in row-major order."""
    low = int.from_bytes(block, "big") & 0xFFFFFFFF
    msb = (low >> 16) & 0xFFFF
    lsb = low & 0xFFFF
    out = [0] * 16
    for x in range(4):
        for y in range(4):
            k = y + 4 * x
            raw = (((msb >> (15 - k)) & 1) << 1) | ((lsb >> (15 - k)) & 1)
            out[y * 4 + x] = ETC_INDEX_MAP[raw]
    return out

def _etc_decode_rgb_block(block: bytes):
    """Decode one 8-byte ETC2 RGB block into a (4, 4, 3) uint8 array.

    NOTE: the T, H and planar bit layouts below are kept exactly as in the
    original implementation and have not been validated against a reference
    decoder (see "Known limitations" in the README).
    """
    if len(block) != 8:
        raise ValueError("ETC2 RGB block must be 8 bytes")

    u = int.from_bytes(block, "big")
    b0, b1, b2, b3 = block[:4]
    out = np.empty((4, 4, 3), np.uint8)

    if not (b3 >> 1) & 1:
        colors = [
            (_expand4(b0 >> 4), _expand4(b1 >> 4), _expand4(b2 >> 4)),
            (_expand4(b0 & 15), _expand4(b1 & 15), _expand4(b2 & 15)),
        ]
        tables = [(b3 >> 5) & 7, (b3 >> 2) & 7]
        flip = b3 & 1
        idx = _etc_indices(block)
        for y in range(4):
            for x in range(4):
                sub = (y >= 2) if flip else (x >= 2)
                mod = ETC_MODIFIERS[tables[sub]][idx[y * 4 + x]]
                c = colors[sub]
                out[y, x] = [_clamp255(c[0] + mod), _clamp255(c[1] + mod), _clamp255(c[2] + mod)]
        return out

    r = (b0 >> 3) & 31
    g = (b1 >> 3) & 31
    b = (b2 >> 3) & 31
    rr = r + _sign3(b0 & 7)
    gg = g + _sign3(b1 & 7)
    bb = b + _sign3(b2 & 7)
    idx = _etc_indices(block)

    if rr < 0 or rr > 31:
        r0, g0, bl0 = (u >> 58) & 15, (u >> 54) & 15, (u >> 50) & 15
        r1, g1, bl1 = (u >> 46) & 15, (u >> 42) & 15, (u >> 38) & 15
        dist = (u >> 32) & 7
        c0 = (_expand4(r0), _expand4(g0), _expand4(bl0))
        c1 = (_expand4(r1), _expand4(g1), _expand4(bl1))
        d = ETC_DIST[dist]
        palette = [
            c0,
            tuple(_clamp255(c1[i] + d) for i in range(3)),
            c1,
            tuple(_clamp255(c1[i] - d) for i in range(3)),
        ]
        for y in range(4):
            for x in range(4):
                out[y, x] = palette[idx[y * 4 + x]]
        return out

    if gg < 0 or gg > 31:
        r0, g0, bl0 = (u >> 54) & 15, (u >> 50) & 15, (u >> 46) & 15
        r1, g1, bl1 = (u >> 42) & 15, (u >> 38) & 15, (u >> 34) & 15
        key0 = (r0 << 8) | (g0 << 4) | bl0
        key1 = (r1 << 8) | (g1 << 4) | bl1
        dist = (((u >> 33) & 3) << 1) | (1 if key0 >= key1 else 0)
        c0 = (_expand4(r0), _expand4(g0), _expand4(bl0))
        c1 = (_expand4(r1), _expand4(g1), _expand4(bl1))
        d = ETC_DIST[dist]
        palette = [
            tuple(_clamp255(c0[i] + d) for i in range(3)),
            tuple(_clamp255(c0[i] - d) for i in range(3)),
            tuple(_clamp255(c1[i] + d) for i in range(3)),
            tuple(_clamp255(c1[i] - d) for i in range(3)),
        ]
        for y in range(4):
            for x in range(4):
                out[y, x] = palette[idx[y * 4 + x]]
        return out

    if bb < 0 or bb > 31:
        o_r, o_g, o_b = (u >> 57) & 63, (u >> 50) & 127, (u >> 44) & 63
        h_r, h_g, h_b = (u >> 38) & 63, (u >> 31) & 127, (u >> 25) & 63
        v_r, v_g, v_b = (u >> 19) & 63, (u >> 12) & 127, (u >> 6) & 63
        origin = (_expand6(o_r), (o_g << 1) | (o_g >> 6), (o_b << 2) | (o_b >> 4))
        horiz = (_expand6(h_r), (h_g << 1) | (h_g >> 6), (h_b << 2) | (h_b >> 4))
        vert = (_expand6(v_r), (v_g << 1) | (v_g >> 6), (v_b << 2) | (v_b >> 4))
        for y in range(4):
            for x in range(4):
                out[y, x] = [
                    _clamp255(
                        (x * (horiz[c] - origin[c]) + y * (vert[c] - origin[c]) + 4 * origin[c] + 2)
                        >> 2
                    )
                    for c in range(3)
                ]
        return out

    c0 = (_expand5(r), _expand5(g), _expand5(b))
    c1 = (_expand5(rr), _expand5(gg), _expand5(bb))
    tables = [(b3 >> 5) & 7, (b3 >> 2) & 7]
    flip = b3 & 1
    for y in range(4):
        for x in range(4):
            sub = (y >= 2) if flip else (x >= 2)
            base = c1 if sub else c0
            mod = ETC_MODIFIERS[tables[sub]][idx[y * 4 + x]]
            out[y, x] = [_clamp255(v + mod) for v in base]
    return out

def _eac_decode_alpha(block: bytes):
    """Decode one 8-byte EAC alpha block into a (4, 4) uint8 array."""
    if len(block) != 8:
        raise ValueError("EAC alpha block must be 8 bytes")

    base = block[0]
    multiplier = block[1] >> 4
    table = block[1] & 15
    bits = int.from_bytes(block[2:8], "big")
    modifiers = EAC_MODIFIERS[table]
    out = np.empty((4, 4), np.uint8)
    for x in range(4):
        for y in range(4):
            k = y + 4 * x
            index = (bits >> (45 - 3 * k)) & 7
            out[y, x] = _clamp255(base + modifiers[index] * multiplier)
    return out


# ---------------------------------------------------------------- ETC2 decode
# Vectorised over blocks.  Individual / differential ETC1 blocks (the vast
# majority) are decoded with array maths; T, H and planar blocks fall back to
# the scalar reference decoder above, so results are identical.

_ETC_MOD_ARR = np.array(ETC_MODIFIERS, np.int64)
_EAC_MOD_ARR = np.array(EAC_MODIFIERS, np.int64)
_ETC_MAP_ARR = np.array(ETC_INDEX_MAP, np.int64)
# k = y + 4*x is the bit position used by the format for pixel (x, y);
# arrays below are indexed in row-major pixel order p = y*4 + x.
_ETC_K = np.array([y + 4 * x for y in range(4) for x in range(4)], np.int64)
_ETC_SUB = np.array(
    [
        [1 if x >= 2 else 0 for y in range(4) for x in range(4)],  # flip = 0
        [1 if y >= 2 else 0 for y in range(4) for x in range(4)],  # flip = 1
    ],
    np.int64,
)

def _etc_decode_rgb_vec(blocks):
    """Decode (N, 8) uint8 ETC2 colour blocks into (N, 16, 3) uint8."""
    n = blocks.shape[0]
    b = blocks.astype(np.int64)
    b0, b1, b2, b3 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    diff = (b3 >> 1) & 1
    r5, g5, bl5 = (b0 >> 3) & 31, (b1 >> 3) & 31, (b2 >> 3) & 31
    d3 = [v & 7 for v in (b0, b1, b2)]
    rr, gg, bb = (
        base + np.where(d & 4, d - 8, d) for base, d in zip((r5, g5, bl5), d3)
    )
    in_range = (rr >= 0) & (rr <= 31) & (gg >= 0) & (gg <= 31) & (bb >= 0) & (bb <= 31)
    simple = (diff == 0) | in_range

    def ex4(v):
        return (v << 4) | v

    def ex5(v):
        return (v << 3) | (v >> 2)

    ind0 = np.stack([ex4(b0 >> 4), ex4(b1 >> 4), ex4(b2 >> 4)], 1)
    ind1 = np.stack([ex4(b0 & 15), ex4(b1 & 15), ex4(b2 & 15)], 1)
    dif0 = np.stack([ex5(r5), ex5(g5), ex5(bl5)], 1)
    dif1 = np.stack([ex5(rr), ex5(gg), ex5(bb)], 1)
    is_diff = (diff == 1)[:, None]
    c0 = np.where(is_diff, dif0, ind0)
    c1 = np.where(is_diff, dif1, ind1)

    low = (b[:, 4] << 24) | (b[:, 5] << 16) | (b[:, 6] << 8) | b[:, 7]
    msb = (low >> 16) & 0xFFFF
    lsb = low & 0xFFFF
    shifts = 15 - _ETC_K
    raw = (((msb[:, None] >> shifts[None, :]) & 1) << 1) | ((lsb[:, None] >> shifts[None, :]) & 1)
    idx = _ETC_MAP_ARR[raw]

    sub = _ETC_SUB[b3 & 1]
    tables = np.where(sub == 1, ((b3 >> 2) & 7)[:, None], ((b3 >> 5) & 7)[:, None])
    mod = _ETC_MOD_ARR[tables, idx]
    colour = np.where((sub == 1)[:, :, None], c1[:, None, :], c0[:, None, :])
    out = np.clip(colour + mod[:, :, None], 0, 255).astype(np.uint8)

    special = np.nonzero(~simple)[0]
    for i in special.tolist():
        out[i] = _etc_decode_rgb_block(bytes(blocks[i])).reshape(16, 3)
    return out

def _eac_decode_vec(blocks):
    """Decode (N, 8) uint8 EAC alpha blocks into (N, 16) uint8."""
    b = blocks.astype(np.int64)
    base = b[:, 0]
    mult = b[:, 1] >> 4
    table = b[:, 1] & 15
    bits = np.zeros(b.shape[0], np.int64)
    for i in range(6):
        bits |= b[:, 2 + i] << (8 * (5 - i))
    shifts = 45 - 3 * _ETC_K
    idx = (bits[:, None] >> shifts[None, :]) & 7
    value = base[:, None] + _EAC_MOD_ARR[table[:, None], idx] * mult[:, None]
    return np.clip(value, 0, 255).astype(np.uint8)

def etc2_decode_level(data: bytes, width: int, height: int, rgba: bool = True):
    nbx = (width + 3) // 4
    nby = (height + 3) // 4
    stride = 16 if rgba else 8
    count = nbx * nby
    need = count * stride
    if len(data) < need:
        raise ValueError(f"ETC2 level is too short: {len(data)} < {need}")

    raw = np.frombuffer(data, np.uint8, count=need).reshape(count, stride)
    dec = np.empty((count, 16, 4), np.uint8)
    for start in range(0, count, ETC2_CHUNK_BLOCKS):
        part = raw[start : start + ETC2_CHUNK_BLOCKS]
        dec[start : start + part.shape[0], :, :3] = _etc_decode_rgb_vec(part[:, :8])
        if rgba:
            dec[start : start + part.shape[0], :, 3] = _eac_decode_vec(part[:, 8:16])
        else:
            dec[start : start + part.shape[0], :, 3] = 255

    out = (
        dec.reshape(nby, nbx, 4, 4, 4)
        .transpose(0, 2, 1, 3, 4)
        .reshape(nby * 4, nbx * 4, 4)
    )
    return out[:height, :width]

# ---------------------------------------------------------------- ETC2 encode
# Same algorithm (and byte-identical output) as the original per-block
# encoder, but evaluated for many blocks at once.

_ETC_INV = np.array([ETC_INDEX_MAP_INVERSE[i] for i in range(4)], np.int64)
_EAC_MULT_MODS = (
    np.array(EAC_MODIFIERS, np.int64)[None, :, :] * np.arange(16, dtype=np.int64)[:, None, None]
)
EAC_UNIQUE_CHUNK = 32

def _etc_encode_rgb_vec(pix):
    """Encode (N, 4, 4, >=3) uint8 blocks as ETC1 individual mode -> (N, 8) uint8."""
    n = pix.shape[0]
    rgb = pix[..., :3].astype(np.int64)
    halves = (rgb[:, :, 0:2, :].reshape(n, 8, 3), rgb[:, :, 2:4, :].reshape(n, 8, 3))

    fits = []
    for sub in halves:
        lo = sub.min(1).astype(np.float64)
        hi = sub.max(1).astype(np.float64)
        q = np.clip(np.rint(((lo + hi) * 0.5) / 17.0), 0, 15).astype(np.int64)
        base = q * 17
        best_err = best_table = best_choice = None
        for table in range(8):
            mods = np.array(ETC_MODIFIERS[table], np.int64)
            palette = np.clip(base[:, None, :] + mods[None, :, None], 0, 255)
            delta = sub[:, :, None, :] - palette[:, None, :, :]
            dist = (delta * delta).sum(3)
            choice = dist.argmin(2)
            err = dist.min(2).sum(1)
            if best_err is None:
                best_err, best_table, best_choice = err, np.full(n, table, np.int64), choice
            else:
                better = err < best_err  # strict: the first best table wins, as before
                best_err = np.where(better, err, best_err)
                best_table = np.where(better, table, best_table)
                best_choice = np.where(better[:, None], choice, best_choice)
        fits.append((q, best_table, best_choice))

    (q0, t0, c0), (q1, t1, c1) = fits
    out = np.zeros((n, 8), np.uint8)
    out[:, 0] = (q0[:, 0] << 4) | q1[:, 0]
    out[:, 1] = (q0[:, 1] << 4) | q1[:, 1]
    out[:, 2] = (q0[:, 2] << 4) | q1[:, 2]
    out[:, 3] = (t0 << 5) | (t1 << 2)

    msb = np.zeros(n, np.int64)
    lsb = np.zeros(n, np.int64)
    for s, choices in enumerate((c0, c1)):
        for y in range(4):
            for xx in range(2):
                k = y + 4 * (s * 2 + xx)
                raw = _ETC_INV[choices[:, y * 2 + xx]]
                msb |= ((raw >> 1) & 1) << (15 - k)
                lsb |= (raw & 1) << (15 - k)
    out[:, 4] = msb >> 8
    out[:, 5] = msb & 255
    out[:, 6] = lsb >> 8
    out[:, 7] = lsb & 255
    return out

def _eac_encode_unique(patterns):
    """Encode (U, 16) int64 alpha patterns (index k = y + 4*x) -> (U, 8) uint8."""
    total_u = patterns.shape[0]
    out = np.empty((total_u, 8), np.uint8)
    for s in range(0, total_u, EAC_UNIQUE_CHUNK):
        a = patterns[s : s + EAC_UNIQUE_CHUNK]
        m = a.shape[0]
        base = np.rint(a.sum(1) / 16.0).astype(np.int64)
        levels = np.clip(base[:, None, None, None] + _EAC_MULT_MODS[None], 0, 255)
        delta = a[:, None, None, :, None] - levels[:, :, :, None, :]
        cost = delta * delta
        best_index = cost.argmin(4)
        total = cost.min(4).sum(3)
        flat_arg = total.reshape(m, 256).argmin(1)
        mult = flat_arg // 16
        table = flat_arg % 16
        sel = best_index.reshape(m, 256, 16)[np.arange(m), flat_arg]
        bits = np.zeros(m, np.int64)
        for k in range(16):
            bits |= sel[:, k].astype(np.int64) << (45 - 3 * k)
        out[s : s + m, 0] = base & 255
        out[s : s + m, 1] = ((mult & 15) << 4) | (table & 15)
        for i in range(6):
            out[s : s + m, 2 + i] = (bits >> (8 * (5 - i))) & 255
    return out

def _eac_encode_vec(pix):
    """Encode the alpha of (N, 4, 4, 4) uint8 blocks -> (N, 8) uint8."""
    n = pix.shape[0]
    alpha = np.ascontiguousarray(pix[..., 3].transpose(0, 2, 1).reshape(n, 16)).astype(np.uint8)
    raw = alpha.tobytes()
    seen: Dict[bytes, int] = {}
    uniques: List[bytes] = []
    inverse = [0] * n
    for i in range(n):
        key = raw[i * 16 : i * 16 + 16]
        j = seen.get(key)
        if j is None:
            j = len(uniques)
            seen[key] = j
            uniques.append(key)
        inverse[i] = j
    patterns = np.frombuffer(b"".join(uniques), np.uint8).reshape(-1, 16).astype(np.int64)
    return _eac_encode_unique(patterns)[np.array(inverse, np.int64)]

def _etc2_encode_band(raw: bytes, shape: Tuple[int, int, int], rgba: bool) -> bytes:
    """Encode a horizontal band (rows*4, nbx*4, 4) of an already padded image."""
    band = np.frombuffer(raw, np.uint8).reshape(shape)
    rows, cols = shape[0] // 4, shape[1] // 4
    blocks = band.reshape(rows, 4, cols, 4, 4).transpose(0, 2, 1, 3, 4).reshape(-1, 4, 4, 4)
    parts = []
    for start in range(0, blocks.shape[0], ETC2_CHUNK_BLOCKS):
        chunk = blocks[start : start + ETC2_CHUNK_BLOCKS]
        colour = _etc_encode_rgb_vec(chunk)
        if rgba:
            colour = np.concatenate([colour, _eac_encode_vec(chunk)], axis=1)
        parts.append(colour.tobytes())
    return b"".join(parts)

def etc2_encode_level(img, rgba: bool, pool=None) -> bytes:
    height, width = img.shape[:2]
    nbx = (width + 3) // 4
    nby = (height + 3) // 4
    padded = np.pad(img, ((0, nby * 4 - height), (0, nbx * 4 - width), (0, 0)), mode="edge")
    padded = np.ascontiguousarray(padded)

    if pool is not None and pool.jobs > 1 and nbx * nby >= PAR_MIN_ETC2_ENCODE_BLOCKS:
        tasks = []
        for r0, r1 in bands(nby, pool.jobs * 2):
            band = np.ascontiguousarray(padded[r0 * 4 : r1 * 4])
            tasks.append((band.tobytes(), tuple(band.shape), rgba))
        return b"".join(pool.map(_etc2_encode_band, tasks))
    return _etc2_encode_band(padded.tobytes(), tuple(padded.shape), rgba)
