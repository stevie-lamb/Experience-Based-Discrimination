#!/usr/bin/env python3
"""Deprecated: use market_sim.py (unified three-scenario pipeline)."""

import subprocess
import sys

if __name__ == "__main__":
    print(
        "downward_biased_signal_sim.py is deprecated; running market_sim.py instead.",
        flush=True,
    )
    raise SystemExit(subprocess.call([sys.executable, "market_sim.py"]))
