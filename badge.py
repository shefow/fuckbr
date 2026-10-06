#!/usr/bin/env python3
"""badge.py - tests.svg and coverage.svg from the output of a test run

    python -m unittest test_all -v > test_result.txt 2>&1
    badge.py                       reads test_result.txt, writes the badges next to it
    badge.py out.txt -o docs/      other input, other folder

Coverage is only there when the file also has a coverage.py report in it:

    coverage run -m unittest test_all -v > test_result.txt 2>&1
    coverage report >> test_result.txt

Without one the coverage badge says n/a.
"""

import argparse
import os
import re
import sys
from xml.sax.saxutils import escape

from common import write

RAN = re.compile(r"^Ran (\d+) tests? in ", re.M)
END = re.compile(r"^(OK|FAILED)(?: \((.*)\))?\s*$", re.M)
TOTAL = re.compile(r"^TOTAL\b.*?(\d+(?:\.\d+)?)%\s*$", re.M)

GREEN, LIME, YELLOW, ORANGE, RED, GREY = "#4c1", "#97ca00", "#dfb317", "#fe7d37", "#e05d44", "#9f9f9f"


def read(path):
    """The file as text. powershell writes utf-16 when redirecting, so cope with that."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", "replace")


def tests_badge(text):
    """(value, colour) for the unittest part of the output."""
    ran, end = RAN.findall(text), END.findall(text)
    if not ran or not end:
        return "unknown", GREY
    count = int(ran[-1])
    status, details = end[-1]
    found = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", details)}
    if status == "FAILED":
        return f"{found.get('failures', 0) + found.get('errors', 0)} failed", RED
    skipped = found.get("skipped", 0)
    if count == skipped:
        return "no tests", GREY
    return f"{count - skipped} passed" + (f", {skipped} skipped" if skipped else ""), GREEN


def coverage_badge(text):
    """(value, colour) from the TOTAL line of a coverage.py report, if there is one."""
    found = TOTAL.findall(text)
    if not found:
        return "n/a", GREY
    pct = float(found[-1])
    for limit, colour in ((100, GREEN), (90, LIME), (75, YELLOW), (50, ORANGE)):
        if pct >= limit:
            return found[-1] + "%", colour
    return found[-1] + "%", RED


def svg(label, value, colour):
    """A flat two part badge. Text is about 7 px per character, good enough."""
    left, right = 10 + 7 * len(label), 10 + 7 * len(value)
    label, value = escape(label), escape(value)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + right}" height="20" role="img" '
        f'aria-label="{label}: {value}">\n'
        f'<title>{label}: {value}</title>\n'
        f'<rect width="{left}" height="20" fill="#555"/>\n'
        f'<rect x="{left}" width="{right}" height="20" fill="{colour}"/>\n'
        f'<g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">\n'
        f'<text x="{left / 2}" y="14">{label}</text>\n'
        f'<text x="{left + right / 2}" y="14">{value}</text>\n'
        f'</g>\n</svg>\n'
    )


def main():
    ap = argparse.ArgumentParser(
        usage="badge.py [RESULT] [-o FOLDER]",
        description="tests.svg and coverage.svg from the output of a test run.")
    ap.add_argument("result", nargs="?", default="test_result.txt", metavar="RESULT",
                    help="saved test output, default: test_result.txt")
    ap.add_argument("-o", "--out", metavar="FOLDER", help="where the badges go, default: next to RESULT")
    a = ap.parse_args()

    try:
        text = read(a.result)
    except OSError as e:
        print(f"{a.result}: {e}", file=sys.stderr)
        sys.exit(1)
    folder = a.out or os.path.dirname(os.path.abspath(a.result))
    for name, (value, colour) in (("tests", tests_badge(text)), ("coverage", coverage_badge(text))):
        write(os.path.join(folder, name + ".svg"), svg(name, value, colour).encode("utf-8"))
        print(f"{name}.svg: {value}")


if __name__ == "__main__":
    main()
