#!/usr/bin/env python3
"""Standalone runner for the FGD configuration audit.

Each experiment downloads the public GENEA-2022 data, clones the reference FGD extractor,
trains the pose autoencoder, scores every system, and writes a JSON result. No generator is
trained. One GPU recommended (see README for the cu118 torch note).

Usage:
    python run.py grid        # Axis A: config-knob grid (upper-body)
    python run.py corpus      # Axis B: training-corpus axis (upper-body)
    python run.py grid_fb     # Axis A replication (full-body)
    python run.py corpus_fb   # Axis B replication (full-body)
    python run.py robust      # robustness controls (matched window count, epoch ablation, per-clip FGD CIs)
"""
import sys, json, importlib

EXPERIMENTS = {
    "grid": "scripts.grid_solve",
    "corpus": "scripts.corpus_solve",
    "grid_fb": "scripts.grid_fb_solve",
    "corpus_fb": "scripts.corpus_fb_solve",
    "robust": "scripts.robust_solve",
}

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in EXPERIMENTS:
        print(__doc__); sys.exit(1)
    name = sys.argv[1]
    mod = importlib.import_module(EXPERIMENTS[name])
    result = mod.solve({})
    out = f"{name}_result.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nwrote {out}")
    print(json.dumps(result.get("analysis", {}), indent=2, default=str)[:2000])

if __name__ == "__main__":
    main()
