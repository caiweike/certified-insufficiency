#!/usr/bin/env python3
"""Decompress every *.jsonl.gz under this repository next to the archive (keeps the .gz)."""
import gzip, pathlib
root = pathlib.Path(__file__).resolve().parent
for gz in sorted(root.rglob("*.jsonl.gz")):
    out = gz.with_suffix("")
    if out.exists():
        continue
    with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
        while chunk := fi.read(1 << 20):
            fo.write(chunk)
    print("unpacked", out.relative_to(root))
