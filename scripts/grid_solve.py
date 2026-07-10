"""MUTABLE — fgdaudit s5.2/s5.3: compute per-system FGD across config axes and measure
ranking (in)stability vs the human ranking.

Faithful to the reference (genea_numerical_evaluations): reuses its EmbeddingNet architecture
and the standard Fréchet distance. This first run varies the reference tool's own two built-in
knobs — chunk_len ∈ {30,60,90} and feat-space vs raw-space — plus AE random seed, on the GENEA
2022 upper-body tier. AE is trained on the pooled submitted motion (one training-corpus cell;
disjoint GENEA-trn corpus is added in a later run). Reports per-config system rankings, the
pairwise-config Kendall τ (rank stability), the top-1 flip rate, and each config's τ vs the
human-likeness ranking.
"""
import os
import sys
import glob
import json
import time
import socket
import zipfile
import subprocess
import urllib.request
from itertools import combinations

import numpy as np

# hung Zenodo/arxiv fetches were the CPU-run stall; hard-cap every socket op.
socket.setdefaulttimeout(180)

MOTION_REC, RATINGS_REC = "6973297", "6940057"
TIER = "upper-body"
FGD_REPO = "https://github.com/genea-workshop/genea_numerical_evaluations"
CHUNKS = [30, 60, 90]
SEEDS = [0, 1]


def _log(msg):
    # lab.log surfaces live in task-logs (stdout is buffered into machine-logs until exit).
    line = f"[grid {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        from lab import lab
        lab.log(line)
    except Exception:
        pass


def _dl(url, dest):
    _log(f"download start: {url}")
    urllib.request.urlretrieve(url, dest)
    sz = os.path.getsize(dest)
    _log(f"download done: {dest} ({sz/1e6:.1f} MB)")
    return sz


def _zfile(rec, fname, dest):
    return _dl(f"https://zenodo.org/api/records/{rec}/files/{fname}/content", dest)


def _load_system_clips(sysdir):
    """each system dir -> list of [T, pose_dim] arrays (flatten [T,J,3] -> [T,J*3])."""
    clips = []
    for f in sorted(glob.glob(os.path.join(sysdir, "*.npy"))):
        a = np.load(f)
        if a.ndim == 3:
            a = a.reshape(a.shape[0], -1)
        clips.append(a.astype(np.float32))
    return clips


def _windows(clips, n_frames, mean, std):
    out = []
    for a in clips:
        a = (a - mean) / std
        n = a.shape[0] // n_frames
        for i in range(n):
            out.append(a[i * n_frames:(i + 1) * n_frames])
    return np.asarray(out, dtype=np.float32) if out else np.zeros((0, n_frames, mean.shape[0]), np.float32)


def _frechet(A, B):
    from scipy import linalg
    mu1, mu2 = A.mean(0), B.mean(0)
    s1, s2 = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1 + s2 - 2 * covmean))


def solve(config):
    import torch
    work = os.path.abspath("work"); os.makedirs(work, exist_ok=True)
    report = {"errors": [], "systems": [], "configs": {}, "human": {}, "analysis": {}}

    # --- data ---
    mzip = os.path.join(work, "m.zip")
    _zfile(MOTION_REC, f"{TIER}_npy.zip", mzip)
    with zipfile.ZipFile(mzip) as z:
        z.extractall(work)
    root = os.path.join(work, f"{TIER}_npy")
    systems = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    report["systems"] = systems
    _log(f"extracted {len(systems)} systems: {systems}")
    gt_name = next((s for s in systems if s.endswith("NA")), systems[0])   # UNA = natural
    clips = {s: _load_system_clips(os.path.join(root, s)) for s in systems}
    _log(f"loaded clips; gt(natural)={gt_name}; "
         + ", ".join(f"{s}:{len(clips[s])}" for s in systems))
    pose_dim = clips[gt_name][0].shape[1]
    report["pose_dim"] = pose_dim; report["gt_system"] = gt_name

    # normalization from pooled data
    allframes = np.concatenate([c for s in systems for c in clips[s]], 0)
    mean, std = allframes.mean(0), allframes.std(0) + 1e-6

    # --- reference EmbeddingNet ---
    _log("cloning reference FGD repo")
    subprocess.run(["git", "clone", "--depth", "1", FGD_REPO, os.path.join(work, "fgd")],
                   check=True, capture_output=True, timeout=300)
    sys.path.insert(0, os.path.join(work, "fgd", "FGD"))
    from embedding_net import EmbeddingNet
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _log(f"device={dev}; pose_dim={pose_dim}")

    def train_ae(n_frames, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        pooled = _windows([c for s in systems for c in clips[s]], n_frames, mean, std)
        net = EmbeddingNet(pose_dim, n_frames).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.5, 0.999))
        X = torch.tensor(pooled, device=dev)
        _log(f"train_ae n_frames={n_frames} seed={seed}: {X.shape[0]} windows dim={pose_dim}")
        for ep in range(10):
            perm = torch.randperm(X.shape[0])
            for i in range(0, X.shape[0], 64):
                b = X[perm[i:i + 64]]
                if b.shape[0] < 2:
                    continue
                out = net(b)
                recon = out[1] if isinstance(out, (tuple, list)) else out
                loss = torch.nn.functional.mse_loss(recon, b)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        return net

    def feats(net, W):
        with torch.no_grad():
            out = net(torch.tensor(W, device=dev))
            f = out[0] if isinstance(out, (tuple, list)) else out
            return f.cpu().numpy()

    # --- config grid: chunk_len x seed x {feat,raw} ---
    for n_frames in CHUNKS:
        for seed in SEEDS:
            try:
                net = train_ae(n_frames, seed)
                gtW = _windows(clips[gt_name], n_frames, mean, std)
                gt_feat = feats(net, gtW)
                for space in ("feat", "raw"):
                    fgd = {}
                    for s in systems:
                        W = _windows(clips[s], n_frames, mean, std)
                        if W.shape[0] < 2:
                            continue
                        if space == "feat":
                            fgd[s] = _frechet(feats(net, W), gt_feat)
                        else:
                            # raw-space contrast: mean-pool frames -> pose_dim per window.
                            # (flattening to pose_dim*n_frames both explodes the sqrtm cost —
                            # O(d^3) on up to 14580-d — and confounds the chunk-length axis
                            # since raw dim would scale with n_frames.)
                            fgd[s] = _frechet(W.mean(axis=1), gtW.mean(axis=1))
                    cfg = f"n{n_frames}_s{seed}_{space}"
                    report["configs"][cfg] = fgd
                    best = min(fgd, key=fgd.get) if fgd else None
                    _log(f"config {cfg}: {len(fgd)} systems scored, best(lowest FGD)={best}")
            except Exception as e:
                import traceback
                report["errors"].append(f"cfg n{n_frames} s{seed}: {e!r}\n{traceback.format_exc()[-400:]}")

    # --- human ranking (lower FGD = better; higher human-likeness = better) ---
    hj = os.path.join(work, "hl.json")
    try:
        _zfile(RATINGS_REC, f"{TIER}_human-likeness.json", hj)
        hl = json.load(open(hj))
        # the json holds per-condition human-likeness; derive a per-system median score
        report["human"] = _human_scores(hl, systems)
    except Exception as e:
        report["errors"].append(f"human json: {e!r}")

    # --- analysis: rank stability + human agreement ---
    report["analysis"] = _analyze(report["configs"], report.get("human", {}))
    return report


def _human_scores(hl, systems):
    """GENEA human-likeness = per-system median rating.

    The json is a list of ~150 participant records; each has `trials`, each trial has
    `responses` = [{"stimulus": "<SYS>_stimuli_muted_NNN_cut", "score": "65"}, ...]. The
    system code is the stimulus prefix. The `id=="training"` trial is an attention check —
    exclude it. (Earlier naive key-walk found nothing because scores live under compound
    stimulus ids, not bare system keys.)
    """
    scores = {}
    if isinstance(hl, list):
        for rec in hl:
            if not isinstance(rec, dict):
                continue
            for tr in rec.get("trials", []):
                if tr.get("id") == "training":
                    continue
                for resp in tr.get("responses", []):
                    s = str(resp.get("stimulus", "")).split("_")[0]
                    v = resp.get("score")
                    if s in systems and v not in (None, ""):
                        try:
                            scores.setdefault(s, []).append(float(v))
                        except (TypeError, ValueError):
                            pass
    return {s: float(np.median(v)) for s, v in scores.items() if v}


def _rank(d):
    return [k for k, _ in sorted(d.items(), key=lambda kv: kv[1])]  # ascending FGD (best first)


def _kendall(r1, r2):
    common = [x for x in r1 if x in r2]
    if len(common) < 2:
        return None
    p1 = {x: i for i, x in enumerate(r1)}
    p2 = {x: i for i, x in enumerate(r2)}
    c = d = 0
    for a, b in combinations(common, 2):
        s1 = np.sign(p1[a] - p1[b]); s2 = np.sign(p2[a] - p2[b])
        if s1 * s2 > 0:
            c += 1
        else:
            d += 1
    return (c - d) / (c + d) if (c + d) else None


def _analyze(configs, human):
    cfgs = list(configs.keys())
    ranks = {c: _rank(configs[c]) for c in cfgs}
    taus = []
    for a, b in combinations(cfgs, 2):
        t = _kendall(ranks[a], ranks[b])
        if t is not None:
            taus.append(t)
    top1 = {c: (ranks[c][0] if ranks[c] else None) for c in cfgs}
    flip = len(set(v for v in top1.values() if v)) > 1
    out = {"n_configs": len(cfgs), "pairwise_tau_median": float(np.median(taus)) if taus else None,
           "pairwise_tau_min": float(np.min(taus)) if taus else None,
           "top1_by_config": top1, "top1_flips": bool(flip),
           "top1_distinct_winners": len(set(v for v in top1.values() if v))}
    if human:
        hr = [k for k, _ in sorted(human.items(), key=lambda kv: -kv[1])]  # best human first
        out["human_ranking"] = hr
        out["config_vs_human_tau"] = {c: _kendall(ranks[c], hr) for c in cfgs}
    return out
