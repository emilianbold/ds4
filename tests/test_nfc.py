#!/usr/bin/env python3
"""Check the Qwen3.8 path's NFC (tests/test_tokenizer --nfc) against the
Hugging Face tokenizers NFC normalizer.

    python3 tests/test_nfc.py

Cases: every codepoint alone, decomposed, after a base letter and before
combining marks of several classes; every composition pair and Hangul jamo
sequence; and random runs of bases, marks and jamo.  Requires the tokenizers
package.
"""
import os
import random
import subprocess
import sys
import tempfile
import unicodedata as ud

from tokenizers import normalizers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NFC = normalizers.NFC()


def cases():
    marks = ["́", "̴", "̣", "̸", "ְ", "่", "゙"]
    for cp in range(0x110000):
        if 0xD800 <= cp < 0xE000 or cp in (0x0A, 0x0D, 0x85, 0x2028, 0x2029):
            continue
        c = chr(cp)
        yield c
        d = ud.normalize("NFD", c)
        if d != c:
            yield d
            yield d[::-1]
        if ud.combining(c) or cp < 0x3400 or 0xF900 <= cp < 0xFB50:
            yield "a" + c
            for m in marks:
                yield c + m
                yield m + c
    rng = random.Random(1)
    pool = ([chr(c) for c in range(0x300, 0x370)] + list("aeiouAEIOUnsc") +
            [chr(c) for c in range(0x1100, 0x1113)] + [chr(c) for c in range(0x1161, 0x1176)] +
            [chr(c) for c in range(0x11A8, 0x11C3)] + ["가", "각", "େ", "ା",
            "ெ", "ா", "゙", "゚", "か", "ཱ", "ི", "ྀ"])
    for _ in range(100000):
        yield "".join(rng.choice(pool) for _ in range(rng.randint(1, 8)))


def main():
    exe = f"{ROOT}/tests/test_tokenizer"
    lines = [c for c in cases() if "\n" not in c and "\r" not in c]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        path = f.name
    try:
        out = subprocess.run([exe, "--nfc", path], check=True, capture_output=True).stdout
    finally:
        os.unlink(path)
    got = out.decode("utf-8").split("\n")[:-1]
    assert len(got) == len(lines), (len(got), len(lines))
    bad = [(s, g) for s, g in zip(lines, got) if g != NFC.normalize_str(s)]
    for s, g in bad[:20]:
        want = NFC.normalize_str(s)
        print("mismatch:", [hex(ord(x)) for x in s], "got", [hex(ord(x)) for x in g],
              "want", [hex(ord(x)) for x in want])
    print(f"NFC: {len(lines)} cases, {len(bad)} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
