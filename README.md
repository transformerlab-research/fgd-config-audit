# FGD configuration audit

Reproducibility package for *"Same Motions, Different Winners: The Gesture-Generation
Leaderboard and Its Human Agreement Are Configuration-Tunable."*

The audit measures how much a co-speech-gesture leaderboard scored by **Fréchet Gesture
Distance (FGD)** moves under undocumented configuration choices, and how far FGD agrees with
human judgment. It trains only the reference FGD pose autoencoder (seconds to minutes on one
GPU); **no gesture generator is trained.**

## Quick start

```bash
pip install numpy scipy
pip install torch --index-url https://download.pytorch.org/whl/cu118   # see note below
python run.py grid        # Axis A config-knob grid (upper-body)
python run.py corpus      # Axis B training-corpus axis (upper-body)
python run.py grid_fb     # Axis A replication (full-body)
python run.py corpus_fb   # Axis B replication (full-body)
python run.py robust      # robustness controls (MF1/MF2/MF3)
```

Each command downloads the public GENEA-2022 data, clones the reference extractor, runs, and
writes `<name>_result.json`. Reference outputs from our runs are in `reference_outputs/` so you
can check numbers without rerunning.

> **torch / GPU note (the one non-obvious requirement).** Install `torch` from the **cu118**
> index. The default build is too new for older CUDA drivers (for example the Lambda A10 on
> CUDA 12.0) and silently falls back to CPU; cu118 is backward-compatible and uses the GPU.

## Data (public, content-addressed by Zenodo record)

| role | Zenodo record | file | notes |
|------|---------------|------|-------|
| motion (upper) | 6973297 | `upper-body_npy.zip` | 11 systems, natural = UNA, pose 162-d |
| motion (full) | 6973297 | `full-body_npy.zip` | 10 systems, natural = FNA, pose 174-d |
| human (upper) | 6940057 | `upper-body_human-likeness.json` | per-participant ratings |
| human (full) | 6940057 | `genea_2022_subjective_evaluation.zip` → `full-body_human-likeness.json` | inside the zip |
| extractor | [genea-workshop/genea_numerical_evaluations](https://github.com/genea-workshop/genea_numerical_evaluations) | `FGD/embedding_net.py` | `EmbeddingNet`, 32-d latent |

Zenodo records are DOI-backed and immutable, so the record + filename is the data identity. The
scripts fetch files directly at run time. All data is public and used unmodified; we
redistribute none of it.

## Seeds and determinism

- Autoencoder seeds **0 and 1** (`torch.manual_seed`, `np.random.seed`). Axis A reports both;
  Axis B pools both. Subsampling in the robustness control uses `np.random.RandomState(0)`.
- FGD (`scipy.linalg.sqrtm`) is deterministic given the trained autoencoder.
- Autoencoder: 10 epochs, Adam, lr 5e-4 (reference default; also run at 50 epochs as a control).

## Experiments

| script | paper section | produces |
|--------|---------------|----------|
| `scripts/grid_solve.py` | Ranking stability + validity (Axis A, upper) | 12-config grid: FGD, pairwise τ, top-1, config-vs-human τ |
| `scripts/corpus_solve.py` | Training corpus (Axis B, upper) | 5-corpus FGD + cross-corpus τ + per-corpus human τ |
| `scripts/grid_fb_solve.py`, `scripts/corpus_fb_solve.py` | Replication | same, full-body tier |
| `scripts/robust_solve.py` | Robustness controls | MF1 matched-window τ, MF2 10/50-epoch seed flip, MF3 per-clip FGD 95% CIs |

## Headline numbers to reproduce (upper-body unless noted)

| quantity | value | script |
|----------|-------|--------|
| config-knob synthetic-only pairwise τ (median / min) | 0.867 / 0.733 | grid |
| feat-only synthetic-only τ (median / min) | 0.911 / 0.822 | grid |
| top-1 synthetic winner | USQ or USM (flips) | grid |
| USQ vs USM per-clip FGD 95% CIs | [2.21, 4.56] vs [2.51, 5.91], gap [-1.35, 3.11] | robust (MF3) |
| seed flip at 10 and 50 epochs | both flip | robust (MF2) |
| matched-window-count τ vs human (30/60/90) | 0.289 / 0.467 / 0.333 | robust (MF1) |
| config-vs-human synthetic-only τ (median, SD) | 0.422, SD 0.057 | grid |
| cross-corpus synthetic-only τ (median / min) | 0.800 / 0.644 | corpus |
| per-corpus human τ (SD) | SD 0.069 (0.378 to 0.556) | corpus |
| full-body config-knob τ (median / min) | 0.889 / 0.722 | grid_fb |
| full-body cross-corpus τ (median / min) | 0.806 / 0.667 | corpus_fb |

## Method notes

- Human-likeness score is the per-system median of the raw 0-100 ratings; the system code is
  the stimulus prefix (`<SYS>_stimuli_muted_NNN_cut`), excluding the `training` attention trial.
- Raw-space FGD mean-pools frames to the pose dimension (not the flattened window) to avoid an
  O(d³) `sqrtm` blow-up and to keep the dimension independent of window length.
- The GENEA `trn`/`val` corpora (Zenodo 6998231) are BVH joint-angle data, format-mismatched
  against the 3D-coordinate submissions, which is why the training-corpus axis uses in-format
  subsets of the submission pool.

## License

Code is MIT (see `LICENSE`). The GENEA-2022 data and human ratings are the property of their
original authors and are used under their public terms; this repository redistributes none of it.
