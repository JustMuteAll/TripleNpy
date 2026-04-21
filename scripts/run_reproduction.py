from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triplen_repro.config import load_config
from triplen_repro.pipeline import run_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Triple-N Python reproduction stages.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default_config.yaml"))
    parser.add_argument("--stage", default="preflight", choices=["preflight", "basic", "fig1", "fig3", "fig5", "all"])
    parser.add_argument("--fig5-max-areas", type=int, default=None)
    parser.add_argument("--fig5-max-models", type=int, default=None)
    parser.add_argument("--fig5-max-voxels", type=int, default=None)
    parser.add_argument("--fig1-debug-compare", action="store_true")
    parser.add_argument("--matlab-compare", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.fig5_max_areas is not None:
        config.analysis["fig5_max_areas"] = args.fig5_max_areas
    if args.fig5_max_models is not None:
        config.analysis["fig5_max_models"] = args.fig5_max_models
    if args.fig5_max_voxels is not None:
        config.analysis["fig5_max_voxels"] = args.fig5_max_voxels
    summary = run_stage(config, args.stage, fig1_debug_compare=args.fig1_debug_compare, matlab_compare=args.matlab_compare)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
