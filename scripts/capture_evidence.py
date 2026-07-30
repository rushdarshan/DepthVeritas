"""Create a traceable evidence/capability record for a DepthLab experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depthlab.evidence import capability_matrix, write_evidence_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture DepthLab experiment provenance")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evidence.json")
    parser.add_argument("--command", required=True, help="Exact command or operation being evidenced")
    parser.add_argument("--tier", choices=("measured", "synthetic", "blocked", "unavailable"), default="measured")
    parser.add_argument("--input", action="append", default=[], help="Existing file to hash; repeatable")
    parser.add_argument("--result", action="append", default=[], help="Output path; repeatable")
    args = parser.parse_args()
    path = write_evidence_bundle(args.output, command=args.command, tier=args.tier, inputs=args.input, outputs=args.result, capabilities=capability_matrix(ROOT))
    print(path)


if __name__ == "__main__":
    main()
