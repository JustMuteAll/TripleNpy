from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triplen_repro.config import load_config
from triplen_repro.io.dataset import load_h5_session, load_h5_session_info, load_processed_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one Triple-N session into a Python-native package.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default_config.yaml"))
    parser.add_argument("--session", required=True, help="Session id like ses01 or integer 1")
    parser.add_argument("--output", default=None, help="Optional output .npz path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    processed = load_processed_session(config, args.session)
    info = load_h5_session_info(config, args.session)
    h5 = load_h5_session(config, args.session)
    output_path = Path(args.output) if args.output else config.paths.output_root / "session_exports" / f"{str(args.session).lower()}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        response_best=np.asarray(processed["response_best"]),
        mean_psth=np.asarray(processed["mean_psth"]),
        reliability_best=np.asarray(processed["reliability_best"]),
        unit_type=np.asarray(processed["UnitType"]),
        pos=np.asarray(processed["pos"]),
        img_idx=np.asarray(info["img_idx"]).reshape(-1),
        raster_matrix_img=h5["raster_matrix_img"],
        response_matrix_img=h5["response_matrix_img"],
        lfp_data=h5["LFP_Data"],
    )
    print(f"Saved session package to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
