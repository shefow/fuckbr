# fuckbr

![tests](tests.svg) ![coverage](coverage.svg)

Tools for the game files of Black Russia: textures, archives, collision files and
animations. Plain python with numpy and pillow, no compiled codecs, so it runs in
termux on the phone itself.

The idea is from [psychobye/fuckbr](https://github.com/psychobye/fuckbr). This is not a
fork, only the idea is taken, the code is mine ([shefow](https://github.com/shefow)).

Unofficial. Not affiliated with the game's developers, no game files in here. Use it on
your own copy for modding and poking around, not for cheating or getting around any
protection.

| tool | what it does |
|------|--------------|
| `btx.py` | KTX 1.1 `.btx` textures (ASTC / ETC2 / RGBA8) to PNG and back |
| `bpc.py` | `.bpc` archives (a zip XORed with a fixed key): pack, unpack, encrypt, decrypt |
| `bpcmeta.py` | builds the `.bpcmeta` index for a `.bpc` |
| `cls.py` | COL1 collision files `.col` to `.cls` and back (names and bounds only) |
| `ani.py` | ANP3 animations `.ani` to `.ifp` and back |

`common.py`, `arr.py`, `astc.py` and `etc2.py` are the shared parts, not tools.
`badge.py` is for the repo only, it draws the two badges above (see tests).

A usual round trip: `bpc.py pack.bpc`, edit what you need (`btx.py` for textures,
`cls.py` for collisions, `ani.py` for animations), `bpc.py pack/ -m`, then
`bpcmeta.py pack.bpc`.

## install

Python 3.8+, numpy (or rsnumpy), pillow. Only `btx.py` needs pillow.

    pip install numpy pillow

On termux: `pkg install python-numpy`, then `pip install pillow`.

## how they behave

Every tool takes one or more inputs, expands wildcards itself (windows and some
android shells won't), puts results next to the sources unless you give `-o`, and
exits with 1 if anything failed.

- `-o PATH` is the output file for one input, or the output folder for several.
- `-j N` is the number of workers, default all cores but at most 8.
- Writes are atomic, a failed conversion never leaves half a file behind.
- `btx.py` and `cls.py` also take a folder (searched at any depth) or a zip / `.bpc`.
  Matching members are converted, nested zips too. A zip is rewritten in place and the
  old one is kept as `FILE.bak`. Members that fail are copied over untouched and
  reported. A `.bpc` is read as a plain zip here, if it is encrypted run `bpc.py` first.

## btx.py

    btx.py tex.btx                     BTX -> PNG
    btx.py tex.png --astc              PNG -> BTX, ASTC 6x6
    btx.py tex.png --astc -b 4x4       another block size
    btx.py tex.png --etc2              ETC2, RGB8 or RGBA8 if there is alpha
    btx.py tex.png --rgba8 -m 4        uncompressed, 4 mip levels
    btx.py textures/ --etc2 -o out/    folder, zip or .bpc works as a root too

- `-a` / `--astc`, `-e` / `--etc2`, `-r` / `--rgba8`: PNG -> BTX in that format.
  Mandatory for PNG input. Without one it is BTX -> PNG.
- `-b WxH`: ASTC block size, one of the 14 standard LDR ones, default 6x6.
- `-m N`: mip levels, box filtered.
- `--fast`: skip the decode self check after encoding.

BTX -> PNG keeps the original btx (zlib) in a private `btXt` chunk and a hash of the
decoded pixels in `btXh`. Convert an unedited PNG back to the same format and you get
the original file byte for byte ("original restored 1:1"). Edited PNG: it re-encodes
and reuses the original header (block size, sRGB flag, mip count, key/value data,
trailing bytes).

KTX supported: 2D, little endian, ASTC LDR (UNORM and sRGB, 4x4 to 12x12), ETC2 RGB8 /
RGBA8 and RGBA8. Up to 16 stray bytes before the KTX signature are kept.

## bpc.py

    bpc.py pack.bpc          unpack to pack/
    bpc.py folder/           pack to folder.bpc
    bpc.py pack.bpc -z       decrypt to pack.zip
    bpc.py pack.zip          encrypt to pack.bpc
    bpc.py folder/ -m        stored zip, no dir entries, sorted (what bpcmeta.py wants)

What happens depends on the input: a `.bpc` is unpacked, a folder is packed, a `.zip`
is encrypted, `-z` decrypts a `.bpc` to a zip instead. Zip contents are CRC checked
while converting, `--fast` skips that. The XOR is obfuscation, not security.

## bpcmeta.py

    bpcmeta.py pack.bpc              -> pack.bpcmeta
    bpcmeta.py pack.zip              -> pack.bpcmeta
    bpcmeta.py folder/               -> folder.bpcmeta, for a folder packed with bpc.py -m
    bpcmeta.py a.bpc b.bpc -o metas/

The archive has to be stored, not compressed: `bpc.py -m` makes one. Layout, little
endian: `u32 count`, then per entry `u32 offset`, `u32 size`, `u8 tag` (`.wav` 0,
`.mp3` 1), `u16 name length`, utf-8 name. The index is checked (safe relative paths, no
duplicates, sorted, contiguous offsets) and parsed back before it is written.

## cls.py

    cls.py model.col                 COL -> CLS
    cls.py model.cls                 CLS -> COL
    cls.py models/ --cls -o out/     folder, zip or .bpc: say the way with --cls or --col
    cls.py models/ --cls -b          drop spheres, boxes and meshes instead of failing

Only COL1 (`COLL`) is read and only names and bounds are kept. A model with geometry is
refused unless you pass `-b`, which drops the geometry.

## ani.py

    ani.py walk.ani                  -> walk.ifp
    ani.py walk.ifp                  -> walk.ani
    ani.py anims/ --ifp -o out/      folder: say the way with --ifp or --ani

Only the 36 byte ANP3 header differs. The last 4 bytes of the 24 byte name don't
survive ani -> ifp, same as always.

## environment

- `BTX_ARRAY_BACKEND`: `auto` (default), `rsnumpy` or `numpy`. `auto` takes rsnumpy if
  it is installed.
- `BTX_ARRAY_THREADS`: threads for rsnumpy. `btx.py` sets it to 1 in its worker
  processes.

## what doesn't work

- ASTC is LDR only. HDR endpoint modes and HDR void-extent blocks raise
  `NotImplementedError`. The encoder is simple on purpose: one partition, one weight
  plane, direct RGB/RGBA endpoints fitted along the principal axis.
- The ETC2 encoder writes individual mode blocks (plus EAC alpha). The decoder does
  every mode, but the T, H and planar bit layouts are kept as in the original
  implementation and have not been checked against a reference decoder. The tests only
  check them against each other (scalar and vectorised decoders agree, each mode has
  the properties it should), not against the spec.
- CLS drops geometry, see above.
- The codecs are pure python, big textures are slow. `btx.py` uses worker processes.

## tests

343 tests, one file, only `unittest`. Every input (zip, `.bpc`, png, ktx, col/cls,
ani/ifp) is generated on the fly, nothing to check in next to it.

    python -m unittest discover -s tests -p "test_all.py" -v
    python -m pytest tests/test_all.py

Put `test_all.py` next to the scripts, or in a `tests/` folder one level below.

Line coverage, 2271 of 2272 lines (99.96%):

| module | lines |
|--------|------:|
| common.py | 252 / 252 |
| arr.py | 25 / 25 |
| ani.py | 55 / 55 |
| bpc.py | 215 / 215 |
| bpcmeta.py | 177 / 177 |
| cls.py | 165 / 165 |
| btx.py | 295 / 295 |
| astc.py | 688 / 689 |
| etc2.py | 329 / 329 |
| badge.py | 70 / 70 |

The one line left is `raise ValueError("bad quant mode")` in `astc.decode_block_mode`.
It can't fire: for every 11 bit block mode `quant_mode` is always 0..11, a test tries
all 2048 modes to prove it. Exclude it and coverage is 100%:

    # .coveragerc
    [report]
    exclude_lines =
        pragma: no cover
        raise ValueError\("bad quant mode"\)

Lines only, branches are not tracked. The table above is static, rerun to refresh it:

    pip install coverage
    coverage run -m unittest discover -s tests -p "test_all.py"
    coverage report -m

The badges at the top are not static, `badge.py` makes them from a saved test run:

    python -m unittest test_all -v > test_result.txt 2>&1
    python badge.py

That writes `tests.svg` and `coverage.svg` next to `test_result.txt`, commit all
three. unittest doesn't print coverage, so the coverage badge says n/a until the file
also has a coverage.py report in it:

    coverage run -m unittest test_all -v > test_result.txt 2>&1
    coverage report >> test_result.txt
    python badge.py

The tests badge turns red with the number of failed tests, grey if the file can't be
read as unittest output. `badge.py other.txt -o docs/` takes another input and folder.

CI, `.github/workflows/tests.yml`:

    name: tests
    on: [push, pull_request]
    jobs:
      test:
        runs-on: ubuntu-latest
        strategy:
          matrix:
            python-version: ["3.9", "3.10", "3.11", "3.12", "3.13"]
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with:
              python-version: ${{ matrix.python-version }}
          - run: pip install numpy pillow coverage
          - run: coverage run -m unittest discover -s tests -p "test_all.py"
          - run: coverage report -m --fail-under=100

## made and tested on

- Poco X7 Pro (RU version), HyperOS 3.0302.0, Android 16
- Termux 0.118.3
- numpy 2.4.4, Pillow 12.3.0

## credits

Code: [shefow and claude sonnet 5.5]
Idea: [psychobye]

## license

MIT, see [LICENSE](LICENSE).
