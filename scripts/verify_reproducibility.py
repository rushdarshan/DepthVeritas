#!/usr/bin/env python3
"""Compare fresh benchmark scoring output to a checked-in baseline JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True)
    parser.add_argument("--actual", required=True)
    args = parser.parse_args()
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
    actual = json.loads(Path(args.actual).read_text(encoding="utf-8"))
    if expected != actual:
        print("Benchmark scores differ from baseline", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
