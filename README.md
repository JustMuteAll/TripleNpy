# TripleNpy

A Python reproduction project for the Triple-N dataset and parts of the original MATLAB analysis pipeline.

This repository currently includes:
- a small config-driven pipeline
- dataset loading utilities for MAT and H5 files
- partial reproductions for Figure 1, Figure 3, and Figure 5
- a few helper scripts for preflight checks and session export

## Quick start

```powershell
pip install -r requirements.txt
python scripts/run_reproduction.py --stage preflight
python scripts/run_reproduction.py --stage fig1
python scripts/run_reproduction.py --stage fig3
```

The code is still under active alignment with the MATLAB version, so outputs should be treated as work-in-progress reproductions.
