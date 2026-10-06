"""ASTC (LDR) block decoder and a simple encoder, numpy based."""

from functools import lru_cache
from typing import Any, Dict, List, Sequence, Tuple

from arr import np
from common import bands

# below these sizes a pool costs more than it saves
PAR_MIN_ASTC_DECODE_BLOCKS = 3000
PAR_MIN_ASTC_PACK_BLOCKS = 8000
MEMO_LIMIT = 65536

Block = Tuple[int, int]

ISE_LEVELS: Dict[int, Tuple[int, int, int]] = {
    2: (1, 0, 0),
    3: (0, 1, 0),
    4: (2, 0, 0),
    5: (0, 0, 1),
    6: (1, 1, 0),
    8: (3, 0, 0),
    10: (1, 0, 1),
    12: (2, 1, 0),
    16: (4, 0, 0),
    20: (2, 0, 1),
    24: (3, 1, 0),
    32: (5, 0, 0),
    40: (3, 0, 1),
    48: (4, 1, 0),
    64: (6, 0, 0),
    80: (4, 0, 1),
    96: (5, 1, 0),
    128: (7, 0, 0),
    160: (5, 0, 1),
    192: (6, 1, 0),
    256: (8, 0, 0),
}
WEIGHT_RANGES = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]
COLOR_RANGES = [r for r in sorted(ISE_LEVELS) if r >= 6]

def _field(value: int, low: int, width: int) -> int:
    """Extract ``width`` bits of ``value`` starting at bit ``low``."""
    return (value >> low) & ((1 << width) - 1)

def ise_bits(count: int, rng: int) -> int:
    """Number of bits needed to store ``count`` values in range ``rng``."""
    bits, trits, quints = ISE_LEVELS[rng]
    total = count * bits
    if trits:
        total += (8 * count + 4) // 5
    if quints:
        total += (7 * count + 2) // 3
    return total

class BitReader:
    """Sequential little-endian bit reader over an integer."""

    def __init__(self, value: int, nbits: int) -> None:
        self._value = value & ((1 << nbits) - 1)
        self._pos = 0

    def read(self, count: int) -> int:
        result = (self._value >> self._pos) & ((1 << count) - 1)
        self._pos += count
        return result

def _decode_trit_block(packed: int) -> List[int]:
    """Unpack 8 bits into five trits."""
    if _field(packed, 2, 3) == 7:
        c = (_field(packed, 5, 3) << 2) | _field(packed, 0, 2)
        t4 = t3 = 2
    else:
        c = _field(packed, 0, 5)
        if _field(packed, 5, 2) == 3:
            t4 = 2
            t3 = _field(packed, 7, 1)
        else:
            t4 = _field(packed, 7, 1)
            t3 = _field(packed, 5, 2)

    if _field(c, 0, 2) == 3:
        t2 = 2
        t1 = _field(c, 4, 1)
        t0 = (_field(c, 3, 1) << 1) | (_field(c, 2, 1) & (~_field(c, 3, 1) & 1))
    elif _field(c, 2, 2) == 3:
        t2 = t1 = 2
        t0 = _field(c, 0, 2)
    else:
        t2 = _field(c, 4, 1)
        t1 = _field(c, 2, 2)
        t0 = (_field(c, 1, 1) << 1) | (_field(c, 0, 1) & (~_field(c, 1, 1) & 1))
    return [t0, t1, t2, t3, t4]

def _decode_quint_block(packed: int) -> List[int]:
    """Unpack 7 bits into three quints."""
    if _field(packed, 1, 2) == 3 and _field(packed, 5, 2) == 0:
        q2 = (
            (_field(packed, 0, 1) << 2)
            | ((_field(packed, 4, 1) & (~_field(packed, 0, 1) & 1)) << 1)
            | (_field(packed, 3, 1) & (~_field(packed, 0, 1) & 1))
        )
        return [4, 4, q2]

    if _field(packed, 1, 2) == 3:
        q2 = 4
        c = (
            (_field(packed, 3, 2) << 3)
            | (((~_field(packed, 5, 2)) & 3) << 1)
            | _field(packed, 0, 1)
        )
    else:
        q2 = _field(packed, 5, 2)
        c = _field(packed, 0, 5)

    if (c & 7) == 5:
        q1 = 4
        q0 = (c >> 3) & 3
    else:
        q1 = (c >> 3) & 3
        q0 = c & 7
    return [q0, q1, q2]

def _build_inverse_table(decode, size: int) -> Dict[Tuple[int, ...], int]:
    """Map every digit tuple back to the first packed value that yields it."""
    table: Dict[Tuple[int, ...], int] = {}
    for packed in range(size):
        table.setdefault(tuple(decode(packed)), packed)
    return table

TRIT_FORWARD = tuple(tuple(_decode_trit_block(p)) for p in range(256))
QUINT_FORWARD = tuple(tuple(_decode_quint_block(p)) for p in range(128))
TRIT_TABLE = _build_inverse_table(_decode_trit_block, 256)
QUINT_TABLE = _build_inverse_table(_decode_quint_block, 128)

TRIT_LAYOUT = [
    ("m", 0),
    ("t", 0, 2, 1),
    ("m", 1),
    ("t", 2, 2, 2),
    ("m", 2),
    ("t", 4, 1, 3),
    ("m", 3),
    ("t", 5, 2, 4),
    ("m", 4),
    ("t", 7, 1, 5),
]
QUINT_LAYOUT = [
    ("m", 0),
    ("t", 0, 3, 1),
    ("m", 1),
    ("t", 3, 2, 2),
    ("m", 2),
    ("t", 5, 2, 3),
]

def ise_decode(stream: BitReader, count: int, rng: int) -> List[int]:
    """Read ``count`` ISE values of range ``rng`` from ``stream``."""
    bits, trits, quints = ISE_LEVELS[rng]
    value = stream._value
    pos = stream._pos
    mask = (1 << bits) - 1
    out: List[int] = []

    if not (trits or quints):
        for _ in range(count):
            out.append((value >> pos) & mask)
            pos += bits
        stream._pos = pos
        return out

    layout, group_size, forward = (
        (TRIT_LAYOUT, 5, TRIT_FORWARD) if trits else (QUINT_LAYOUT, 3, QUINT_FORWARD)
    )
    while len(out) < count:
        take = min(group_size, count - len(out))
        plain = [0] * group_size
        packed = 0
        for seg in layout:
            if seg[0] == "m":
                plain[seg[1]] = (value >> pos) & mask
                pos += bits
            else:
                width = seg[2]
                packed |= ((value >> pos) & ((1 << width) - 1)) << seg[1]
                pos += width
        digits = forward[packed]
        out.extend((digits[i] << bits) | plain[i] for i in range(take))
    stream._pos = pos
    return out

def ise_encode(values: Sequence[int], rng: int) -> int:
    """Pack ISE ``values`` into an integer bitstream (least-significant first)."""
    bits, trits, quints = ISE_LEVELS[rng]
    out = 0
    pos = 0
    mask = (1 << bits) - 1

    if not (trits or quints):
        for value in values:
            out |= value << pos
            pos += bits
        return out

    layout, group_size, table = (
        (TRIT_LAYOUT, 5, TRIT_TABLE) if trits else (QUINT_LAYOUT, 3, QUINT_TABLE)
    )
    for start in range(0, len(values), group_size):
        group = values[start : start + group_size]
        count = len(group)
        pad = [0] * (group_size - count)
        plain = [v & mask for v in group] + pad
        packed = table[tuple([v >> bits for v in group] + pad)]
        for seg in layout:
            if seg[0] == "m":
                if seg[1] < count:
                    out |= plain[seg[1]] << pos
                    pos += bits
            elif count >= seg[3]:
                out |= ((packed >> seg[1]) & ((1 << seg[2]) - 1)) << pos
                pos += seg[2]
    return out

def _bit_pattern(pattern: str, m: int) -> int:
    """Build a value from bits of ``m`` selected by letters in ``pattern``."""
    value = 0
    for ch in pattern:
        value <<= 1
        if ch != "0":
            value |= (m >> (ord(ch) - ord("a"))) & 1
    return value

COLOR_QUANT = {
    6: (204, "000000000"),
    12: (93, "b000b0b00"),
    24: (44, "cb000cbcb"),
    48: (22, "dcb000dcb"),
    96: (11, "edcb000ed"),
    192: (5, "fedcb000f"),
    10: (113, "000000000"),
    20: (54, "b0000bb00"),
    40: (26, "cb0000cbc"),
    80: (13, "dcb0000dc"),
    160: (6, "edcb0000e"),
}
WEIGHT_QUANT = {
    6: (50, "0000000"),
    12: (23, "b000b0b"),
    24: (11, "cb000cb"),
    10: (28, "0000000"),
    20: (13, "b0000b0"),
}
WEIGHTS_DIRECT = {3: [0, 32, 64], 5: [0, 16, 32, 48, 64]}

def unquantize_color(value: int, rng: int) -> int:
    """Map a quantised colour index to an 8-bit endpoint value."""
    bits, trits, quints = ISE_LEVELS[rng]
    if not (trits or quints):
        if bits == 8:
            return value
        result = 0
        shift = 8
        while shift > 0:
            shift -= bits
            result |= (value << shift) if shift >= 0 else (value >> -shift)
        return result & 255

    multiplier, pattern = COLOR_QUANT[rng]
    high = value >> bits
    low = value & ((1 << bits) - 1)
    a = 0x1FF if (low & 1) else 0
    t = (high * multiplier + _bit_pattern(pattern, low)) ^ a
    return (a & 0x80) | (t >> 2)

def unquantize_weight(value: int, rng: int) -> int:
    """Map a quantised weight index to the 0..64 interpolation weight."""
    if rng in WEIGHTS_DIRECT:
        return WEIGHTS_DIRECT[rng][value]

    bits, trits, quints = ISE_LEVELS[rng]
    if not (trits or quints):
        weight = value
        if bits == 1:
            weight = 63 * value
        elif bits == 2:
            weight = value * 21
        elif bits == 3:
            weight = value * 9
        elif bits == 4:
            weight = (value << 2) | (value >> 2)
        elif bits == 5:
            weight = (value << 1) | (value >> 4)
    else:
        multiplier, pattern = WEIGHT_QUANT[rng]
        high = value >> bits
        low = value & ((1 << bits) - 1)
        a = 0x7F if (low & 1) else 0
        t = (high * multiplier + _bit_pattern(pattern, low)) ^ a
        weight = (a & 0x20) | (t >> 2)
    return weight + 1 if weight > 32 else weight

def decode_block_mode(mode: int) -> Tuple[int, int, int, int]:
    """Decode an 11-bit block mode into (grid_x, grid_y, dual_plane, weight_range)."""
    base = (mode >> 4) & 1
    high_prec = (mode >> 9) & 1
    dual = (mode >> 10) & 1
    a = (mode >> 5) & 3

    if (mode & 3) != 0:
        base |= (mode & 3) << 1
        b = (mode >> 7) & 3
        k = (mode >> 2) & 3
        if k == 0:
            x, y = b + 4, a + 2
        elif k == 1:
            x, y = b + 8, a + 2
        elif k == 2:
            x, y = a + 2, b + 8
        else:
            b &= 1
            if mode & 0x100:
                x, y = b + 2, a + 2
            else:
                x, y = a + 2, b + 6
    else:
        k = (mode >> 2) & 3
        if k == 0:
            raise ValueError("reserved block mode")
        base |= k << 1
        b = (mode >> 9) & 3
        k2 = (mode >> 7) & 3
        if k2 == 0:
            x, y = 12, a + 2
        elif k2 == 1:
            x, y = a + 2, 12
        elif k2 == 2:
            x, y = a + 6, b + 6
            dual = 0
            high_prec = 0
        else:
            k3 = (mode >> 5) & 3
            if k3 == 0:
                x, y = 6, 10
            elif k3 == 1:
                x, y = 10, 6
            else:
                raise ValueError("reserved block mode")

    quant_mode = (base - 2) + 6 * high_prec
    if quant_mode < 0 or quant_mode > 11:
        raise ValueError("bad quant mode")
    return x, y, dual, WEIGHT_RANGES[quant_mode]

def _hash52(p: int) -> int:
    mask = 0xFFFFFFFF
    p ^= p >> 15
    p = (p * 0xEEDE0891) & mask
    p ^= p >> 5
    p = (p + (p << 16)) & mask
    p ^= p >> 7
    p ^= p >> 3
    p = (p ^ (p << 6)) & mask
    p ^= p >> 17
    return p

def partition_index(seed: int, x: int, y: int, count: int, small: bool) -> int:
    """Partition (0..3) that texel (x, y) belongs to."""
    if small:
        x <<= 1
        y <<= 1
    seed += (count - 1) * 1024
    r = _hash52(seed & 0xFFFFFFFF)

    s = [(r >> (4 * i)) & 15 for i in range(8)]
    s += [
        (r >> 18) & 15,
        (r >> 22) & 15,
        (r >> 26) & 15,
        ((r >> 30) | (r << 2)) & 15,
    ]
    s = [v * v for v in s]

    if seed & 1:
        sh1 = 4 if seed & 2 else 5
        sh2 = 6 if count == 3 else 5
    else:
        sh1 = 6 if count == 3 else 5
        sh2 = 4 if seed & 2 else 5
    sh3 = sh1 if seed & 0x10 else sh2
    s = [v >> h for v, h in zip(s, [sh1, sh2] * 4 + [sh3] * 4)]

    a = (s[0] * x + s[1] * y + (r >> 14)) & 63
    b = (s[2] * x + s[3] * y + (r >> 10)) & 63
    c = (s[4] * x + s[5] * y + (r >> 6)) & 63
    d = (s[6] * x + s[7] * y + (r >> 2)) & 63
    if count < 4:
        d = 0
    if count < 3:
        c = 0

    if a >= b and a >= c and a >= d:
        return 0
    if b >= c and b >= d:
        return 1
    if c >= d:
        return 2
    return 3

@lru_cache(maxsize=None)
def partition_map(seed: int, count: int, bw: int, bh: int):
    small = bw * bh < 31
    return np.array(
        [
            partition_index(seed, s, t, count, small)
            for t in range(bh)
            for s in range(bw)
        ]
    )

def _clamp(v: int) -> int:
    return 0 if v < 0 else 255 if v > 255 else v

def _blue_contract(r: int, g: int, b: int) -> Tuple[int, int, int]:
    return (r + b) >> 1, (g + b) >> 1, b

def _bit_transfer_signed(a: int, b: int) -> Tuple[int, int]:
    b = (b >> 1) | (a & 0x80)
    a = (a >> 1) & 0x3F
    if a & 0x20:
        a -= 0x40
    return a, b

def endpoints(cem: int, v: Sequence[int]):
    """Decode colour endpoint mode ``cem`` into two RGBA endpoints."""
    if cem == 0:
        return (v[0],) * 3 + (255,), (v[1],) * 3 + (255,)

    if cem == 1:
        low = (v[0] >> 2) | (v[1] & 0xC0)
        high = min(255, low + (v[1] & 0x3F))
        return (low,) * 3 + (255,), (high,) * 3 + (255,)

    if cem == 4:
        return (v[0],) * 3 + (v[2],), (v[1],) * 3 + (v[3],)

    if cem == 5:
        o0, b0 = _bit_transfer_signed(v[1], v[0])
        o1, b1 = _bit_transfer_signed(v[3], v[2])
        return (
            (b0,) * 3 + (b1,),
            (_clamp(b0 + o0),) * 3 + (_clamp(b1 + o1),),
        )

    if cem in (6, 10):
        e1 = (v[0], v[1], v[2])
        scale = v[3]
        e0 = ((v[0] * scale) >> 8, (v[1] * scale) >> 8, (v[2] * scale) >> 8)
        a0, a1 = (255, 255) if cem == 6 else (v[4], v[5])
        return e0 + (a0,), e1 + (a1,)

    if cem in (8, 12):
        r0, r1, g0, g1, b0, b1 = v[:6]
        a0, a1 = (255, 255) if cem == 8 else (v[6], v[7])
        if r1 + g1 + b1 >= r0 + g0 + b0:
            return (r0, g0, b0, a0), (r1, g1, b1, a1)
        return _blue_contract(r1, g1, b1) + (a1,), _blue_contract(r0, g0, b0) + (a0,)

    if cem in (9, 13):
        o_r, r0 = _bit_transfer_signed(v[1], v[0])
        o_g, g0 = _bit_transfer_signed(v[3], v[2])
        o_b, b0 = _bit_transfer_signed(v[5], v[4])
        if cem == 9:
            a0 = a1 = 255
        else:
            o_a, a0 = _bit_transfer_signed(v[7], v[6])
            a1 = _clamp(a0 + o_a)
        e0 = (r0, g0, b0)
        e1 = (_clamp(r0 + o_r), _clamp(g0 + o_g), _clamp(b0 + o_b))
        if o_r + o_g + o_b >= 0:
            return e0 + (a0,), e1 + (a1,)
        return _blue_contract(*e1) + (a1,), _blue_contract(*e0) + (a0,)

    raise NotImplementedError(f"CEM {cem} (HDR) is not supported")

@lru_cache(maxsize=None)
def infill_matrix(bw: int, bh: int, nx: int, ny: int):
    """Matrix mapping an nx*ny weight grid to bw*bh texel weights (x16)."""
    ds = (1024 + bw // 2) // (bw - 1)
    dt = (1024 + bh // 2) // (bh - 1)
    matrix = np.zeros((bw * bh, nx * ny), np.int64)
    for t in range(bh):
        for s in range(bw):
            gs = (ds * s * (nx - 1) + 32) >> 6
            gt = (dt * t * (ny - 1) + 32) >> 6
            js, fs, jt, ft = gs >> 4, gs & 15, gt >> 4, gt & 15
            w11 = (fs * ft + 8) >> 4
            w10 = ft - w11
            w01 = fs - w11
            w00 = 16 - fs - ft + w11
            for i, j, w in (
                (js, jt, w00),
                (js + 1, jt, w01),
                (js, jt + 1, w10),
                (js + 1, jt + 1, w11),
            ):
                matrix[t * bw + s, min(j, ny - 1) * nx + min(i, nx - 1)] += w
    return matrix

# Bit-reversal table: reverses the bits of a 128-bit ASTC block at C speed
# via bytes.translate (replaces a format()/int() string round trip).
_REV8 = bytes(int(format(i, "08b")[::-1], 2) for i in range(256))

@lru_cache(maxsize=None)
def _weight_lut(rng: int):
    size = ISE_LEVELS[rng]
    bits, trits, quints = size
    count = (1 << bits) * (3 if trits else 5 if quints else 1)
    return tuple(unquantize_weight(v, rng) for v in range(count))

@lru_cache(maxsize=None)
def _color_lut(rng: int):
    bits, trits, quints = ISE_LEVELS[rng]
    count = (1 << bits) * (3 if trits else 5 if quints else 1)
    return tuple(unquantize_color(v, rng) for v in range(count))

def decode_block(block: bytes, bw: int, bh: int):
    """Decode one 16-byte ASTC block into a (bw*bh, 4) uint8 array."""
    b = int.from_bytes(block, "little")

    if _field(b, 0, 9) == 0x1FC:
        if _field(b, 9, 1):
            raise NotImplementedError("HDR void-extent")
        colour = np.array([_field(b, 64 + 16 * i, 16) >> 8 for i in range(4)], np.uint8)
        return np.tile(colour, (bh * bw, 1))

    nx, ny, dual, weight_range = decode_block_mode(_field(b, 0, 11))
    partitions = _field(b, 11, 2) + 1
    weight_count = nx * ny * (2 if dual else 1)
    weight_bits = ise_bits(weight_count, weight_range)
    if (
        nx > bw
        or ny > bh
        or weight_count > 64
        or weight_bits < 24
        or weight_bits > 96
    ):
        raise ValueError("bad weights")

    reversed_bits = int.from_bytes(block[::-1].translate(_REV8), "little")
    weight_lut = _weight_lut(weight_range)
    weights = [
        weight_lut[v]
        for v in ise_decode(BitReader(reversed_bits, weight_bits), weight_count, weight_range)
    ]

    extra = 0
    seed = 0
    if partitions == 1:
        modes = [_field(b, 13, 4)]
        color_start = 17
    else:
        seed = _field(b, 13, 10)
        mode_class = _field(b, 23, 2)
        color_start = 29
        if mode_class == 0:
            modes = [_field(b, 25, 4)] * partitions
        else:
            extra = 3 * partitions - 4
            packed = _field(b, 25, 4) | (_field(b, 128 - weight_bits - extra, extra) << 4)
            class_bits = [(packed >> i) & 1 for i in range(partitions)]
            mode_bits = [(packed >> (partitions + 2 * i)) & 3 for i in range(partitions)]
            modes = [
                ((mode_class - 1 + class_bits[i]) << 2) | mode_bits[i]
                for i in range(partitions)
            ]

    plane_channel = _field(b, 128 - weight_bits - extra - 2, 2) if dual else 0
    value_count = sum(((m >> 2) + 1) * 2 for m in modes)
    if value_count > 18:
        raise ValueError("too many color values")

    available = 128 - weight_bits - extra - (2 if dual else 0) - color_start
    color_range = next(
        (r for r in reversed(COLOR_RANGES) if ise_bits(value_count, r) <= available),
        None,
    )
    if color_range is None:
        raise ValueError("no color range")

    color_bits = ise_bits(value_count, color_range)
    color_lut = _color_lut(color_range)
    values = [
        color_lut[v]
        for v in ise_decode(BitReader(b >> color_start, color_bits), value_count, color_range)
    ]

    eps = []
    k = 0
    for mode in modes:
        n = ((mode >> 2) + 1) * 2
        eps.append(endpoints(mode, values[k : k + n]))
        k += n

    matrix = infill_matrix(bw, bh, nx, ny)
    weights = np.array(weights, np.int64)
    if dual:
        w0 = (matrix @ weights[0::2] + 8) >> 4
        w1 = (matrix @ weights[1::2] + 8) >> 4
        texel_w = np.repeat(w0[:, None], 4, 1)
        texel_w[:, plane_channel] = w1
    else:
        texel_w = np.repeat(((matrix @ weights + 8) >> 4)[:, None], 4, 1)

    part = partition_map(seed, partitions, bw, bh) if partitions > 1 else np.zeros(bw * bh, int)
    e0 = np.array([e[0] for e in eps], np.int64)[part]
    e1 = np.array([e[1] for e in eps], np.int64)[part]
    mixed = (e0 * 257 * (64 - texel_w) + e1 * 257 * texel_w + 32) >> 6
    return (mixed >> 8).astype(np.uint8)

def _decode_rows(data: bytes, nbx: int, nby: int, bw: int, bh: int):
    """Decode ``nby`` rows of ASTC blocks into a (nby*bh, nbx*bw, 4) array.

    Constant-colour (void-extent) blocks are filled in bulk and identical
    blocks are decoded only once.
    """
    count = nbx * nby
    raw = np.frombuffer(data, np.uint8, count=count * 16).reshape(count, 16)
    texels = bw * bh
    dec = np.empty((count, texels, 4), np.uint8)

    # LDR void-extent: low 9 bits == 0x1FC and bit 9 (HDR flag) == 0.
    void = (raw[:, 0] == 0xFC) & ((raw[:, 1] & 3) == 1)
    if void.any():
        colours = raw[:, 9:16:2][void]
        dec[void] = colours[:, None, :]
        todo = np.nonzero(~void)[0].tolist()
    else:
        todo = range(count)

    memo: Dict[bytes, Any] = {}
    for i in todo:
        key = data[i * 16 : i * 16 + 16]
        texel = memo.get(key)
        if texel is None:
            texel = decode_block(key, bw, bh)
            if len(memo) < MEMO_LIMIT:
                memo[key] = texel
        dec[i] = texel

    return (
        dec.reshape(nby, nbx, bh, bw, 4)
        .transpose(0, 2, 1, 3, 4)
        .reshape(nby * bh, nbx * bw, 4)
    )

def _decode_rows_task(data: bytes, nbx: int, nby: int, bw: int, bh: int):
    return _decode_rows(data, nbx, nby, bw, bh)

def decode_level(data: bytes, width: int, height: int, bw: int, bh: int, pool=None):
    """Decode one ASTC mip level into a (height, width, 4) uint8 array."""
    nbx = (width + bw - 1) // bw
    nby = (height + bh - 1) // bh
    if len(data) < nbx * nby * 16:
        raise ValueError("not enough data for image dimensions")

    if pool is not None and pool.jobs > 1 and nbx * nby >= PAR_MIN_ASTC_DECODE_BLOCKS:
        tasks = [(data[r0 * nbx * 16 : r1 * nbx * 16], nbx, r1 - r0, bw, bh)
                 for r0, r1 in bands(nby, pool.jobs * 2)]
        out = np.concatenate(list(pool.map(_decode_rows_task, tasks)), axis=0)
    else:
        out = _decode_rows(data, nbx, nby, bw, bh)
    return out[:height, :width]


@lru_cache(maxsize=None)
def quant_luts(kind: str, rng: int):
    """Return (nearest-index table, dequantised values) for a colour/weight range."""
    if kind == "c":
        dequant = np.array([unquantize_color(v, rng) for v in range(rng)])
        xs = np.arange(256)
    else:
        dequant = np.array([unquantize_weight(v, rng) for v in range(rng)])
        xs = np.arange(65)
    index = np.abs(xs[:, None] - dequant[None, :]).argmin(1)
    return index, dequant

@lru_cache(maxsize=None)
def _block_mode_candidates(bw: int, bh: int, nvals: int):
    """All single-plane block modes usable for this block size and value count."""
    candidates = []
    for mode in range(2048):
        if (mode & 0x1FF) == 0x1FC:
            continue
        try:
            nx, ny, dual, wr = decode_block_mode(mode)
        except ValueError:
            continue
        if dual or nx > bw or ny > bh or nx * ny > 64:
            continue
        wb = ise_bits(nx * ny, wr)
        if wb < 24 or wb > 96:
            continue
        available = 128 - 17 - wb
        cr = next(
            (r for r in reversed(COLOR_RANGES) if ise_bits(nvals, r) <= available),
            None,
        )
        if cr is None or cr < 16:
            continue
        candidates.append((mode, nx, ny, wr, cr))
    return tuple(candidates)

def choose_config(bw: int, bh: int, nvals: int, span: float):
    """Pick the block mode with the lowest estimated error for ``span``."""
    best = None
    for mode, nx, ny, wr, cr in _block_mode_candidates(bw, bh, nvals):
        err = (
            span**2 / (12 * (wr - 1) ** 2)
            + 255**2 / (12 * (cr - 1) ** 2) * 0.5
            + span**2 * 0.08 * (1 - nx * ny / (bw * bh))
        )
        if best is None or err < best[0] - 1e-9:
            best = (err, mode, nx, ny, wr, cr)
    return best

def encode_level(img, bw: int, bh: int, pool=None) -> bytes:
    """Encode an (H, W, 4) uint8 image as ASTC blocks of size bw x bh."""
    height, width = img.shape[:2]
    nbx = (width + bw - 1) // bw
    nby = (height + bh - 1) // bh
    padded = np.pad(img, ((0, nby * bh - height), (0, nbx * bw - width), (0, 0)), mode="edge")

    has_alpha = bool((img[..., 3] < 255).any())
    channels = 4 if has_alpha else 3
    cem = 12 if has_alpha else 8
    nvals = 2 * channels

    blocks = (
        padded.reshape(nby, bh, nbx, bw, 4)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, bh * bw, 4)
        .astype(np.float64)
    )
    nb = blocks.shape[0]
    flat = (blocks.max(1) == blocks.min(1)).all(1)
    pixels = blocks[:, :, :channels]

    mean = pixels.mean(1)
    centred = pixels - mean[:, None, :]
    cov = np.einsum("bnc,bnd->bcd", centred, centred)
    _, vecs = np.linalg.eigh(cov)
    axis = vecs[:, :, -1]
    proj = np.einsum("bnc,bc->bn", centred, axis)
    proj_min = proj.min(1)
    proj_max = proj.max(1)
    proj_range = proj_max - proj_min

    non_flat = ~flat
    span = float(proj_range[non_flat].mean()) if non_flat.any() else 1.0
    config = choose_config(bw, bh, nvals, max(span, 1.0))
    if config is None:
        raise ValueError("no suitable block mode found")
    _, block_mode, nx, ny, weight_range, color_range = config

    e0 = mean + axis * proj_min[:, None]
    e1 = mean + axis * proj_max[:, None]
    w0 = np.where(
        proj_range[:, None] > 1e-9,
        (proj - proj_min[:, None]) / np.maximum(proj_range, 1e-9)[:, None],
        0.0,
    )
    a = 1 - w0
    b = w0
    saa = (a * a).sum(1)
    sab = (a * b).sum(1)
    sbb = (b * b).sum(1)
    det = saa * sbb - sab * sab
    pa = np.einsum("bn,bnc->bc", a, pixels)
    pb = np.einsum("bn,bnc->bc", b, pixels)
    well_posed = det > 1e-6
    safe_det = np.where(well_posed, det, 1.0)[:, None]
    n0 = (sbb[:, None] * pa - sab[:, None] * pb) / safe_det
    n1 = (saa[:, None] * pb - sab[:, None] * pa) / safe_det
    e0 = np.where(well_posed[:, None], n0, e0)
    e1 = np.where(well_posed[:, None], n1, e1)
    e0 = np.clip(np.rint(e0), 0, 255).astype(int)
    e1 = np.clip(np.rint(e1), 0, 255).astype(int)

    color_index, color_dequant = quant_luts("c", color_range)
    i0 = color_index[e0]
    i1 = color_index[e1]
    d0 = color_dequant[i0]
    d1 = color_dequant[i1]
    swap = d1[:, :3].sum(1) < d0[:, :3].sum(1)
    i0, i1 = np.where(swap[:, None], i1, i0), np.where(swap[:, None], i0, i1)
    d0, d1 = np.where(swap[:, None], d1, d0), np.where(swap[:, None], d0, d1)

    direction = (d1 - d0).astype(np.float64)
    denom = (direction * direction).sum(1)
    numer = np.einsum("bnc,bc->bn", pixels - d0[:, None, :], direction)
    texel_w = np.where(
        denom[:, None] > 1e-9,
        numer / np.maximum(denom, 1e-9)[:, None],
        0.0,
    ).clip(0, 1)
    infill = infill_matrix(bw, bh, nx, ny).astype(np.float64) / 16.0
    pinv = np.linalg.pinv(infill)
    grid = np.clip(np.rint((texel_w @ pinv.T) * 64), 0, 64).astype(int)
    weight_index, _ = quant_luts("w", weight_range)
    weight_idx = weight_index[grid]
    weight_bits = ise_bits(nx * ny, weight_range)


    params = (block_mode, cem, channels, color_range, weight_range, weight_bits)
    flat_l = flat.tolist()
    colours = blocks[:, 0, :].astype(np.int64).tolist()
    i0_l = i0.tolist()
    i1_l = i1.tolist()
    widx_l = weight_idx.tolist()

    if pool is not None and pool.jobs > 1 and nb >= PAR_MIN_ASTC_PACK_BLOCKS:
        tasks = [
            (params, flat_l[a:b], colours[a:b], i0_l[a:b], i1_l[a:b], widx_l[a:b])
            for a, b in bands(nb, pool.jobs * 2)
        ]
        return b"".join(pool.map(_astc_pack, tasks))
    return _astc_pack(params, flat_l, colours, i0_l, i1_l, widx_l)

def _astc_pack(params, flat, colours, i0, i1, widx) -> bytes:
    """Serialise already-quantised ASTC blocks (16 bytes each)."""
    block_mode, cem, channels, color_range, weight_range, weight_bits = params
    flat_word = 0x1FC | (3 << 10) | (((1 << 52) - 1) << 12)
    flat_cache: Dict[Tuple[int, ...], bytes] = {}
    color_cache: Dict[Tuple[int, ...], int] = {}
    weight_cache: Dict[Tuple[int, ...], int] = {}
    out = bytearray()

    for k in range(len(flat)):
        if flat[k]:
            colour = tuple(colours[k])
            data = flat_cache.get(colour)
            if data is None:
                word = flat_word
                for i in range(4):
                    word |= (colour[i] * 257) << (64 + 16 * i)
                data = word.to_bytes(16, "little")
                flat_cache[colour] = data
            out += data
            continue

        a0, a1 = i0[k], i1[k]
        values: List[int] = []
        for c in range(channels):
            values += [a0[c], a1[c]]
        ckey = tuple(values)
        color_word = color_cache.get(ckey)
        if color_word is None:
            color_word = ise_encode(values, color_range)
            if len(color_cache) < MEMO_LIMIT:
                color_cache[ckey] = color_word

        wkey = tuple(widx[k])
        weight_rev = weight_cache.get(wkey)
        if weight_rev is None:
            weight_word = ise_encode(list(wkey), weight_range)
            # Reverse all 128 bits; equals reversing weight_bits bits and
            # shifting into the top of the block.
            weight_rev = int.from_bytes(
                weight_word.to_bytes(16, "little")[::-1].translate(_REV8), "little"
            )
            if len(weight_cache) < MEMO_LIMIT:
                weight_cache[wkey] = weight_rev

        word = block_mode | (cem << 13) | (color_word << 17) | weight_rev
        out += word.to_bytes(16, "little")
    return bytes(out)
