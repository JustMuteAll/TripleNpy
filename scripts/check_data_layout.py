from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triplen_repro.config import load_config
from triplen_repro.data_layout import run_preflight


def main() -> int:
    config = load_config(ROOT / "configs" / "default_config.yaml")
    _, lines, failures = run_preflight(config)
    print(json.dumps({"lines": lines, "failures": failures}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
