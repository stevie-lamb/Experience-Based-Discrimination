#!/usr/bin/env python3
"""Re-render LaTeX market outcomes from an existing JSON file."""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.market_outcomes import write_market_outcomes_tex


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate market_outcomes.tex from market_outcomes.json",
    )
    parser.add_argument("json_path", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output .tex path",
    )
    args = parser.parse_args()

    with args.json_path.open(encoding="utf-8") as f:
        out = json.load(f)
    write_market_outcomes_tex(out, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
