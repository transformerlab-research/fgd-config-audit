"""MUTABLE — fgdaudit s5.8: robustness controls for reviewer must-fixes.

Addresses three confounds raised in review, all on GENEA-2022 upper-body (feat space):

  MF1 (window vs sample-size). The window->human-agreement trend co-varies with the number
      of windows (shorter windows => more windows), and Frechet distance is sample-size
      biased. We recompute the trend with the per-system window COUNT held fixed across
      window lengths (subsample each system to the count available at window 90), so sample
      size is controlled. If the shorter-window advantage survives, the claim stands.

  MF2 (undertraining). A cross-seed top-1 flip at 10 epochs could be optimization noise. We
      retrain at {10, 50} epochs (n=30, feat, seeds {0,1}) and report whether the seed flip
      and the cross-seed FGD gap persist with more training.

  MF3 (no CI on FGD values). We bootstrap over the 40 clips (resample clips with replacement,
      recompute per-system FGD) to put a 95% CI on each system's FGD and on the USQ-USM gap,
      so we can say whether the top-1 flip is a tie-flip or a real inversion.
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

import numpy as np

socket.setdefaulttimeout(180)

MOTION_REC, RATINGS_REC = "6973297", "6940057"
TIER = "upper-body"
FGD_REPO = "https://github.com/genea-workshop/genea_numerical_evaluations"


def _log(msg):
    line = f"[robust {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        from lab import lab
        lab.log(line)
    except Exception:
        pass


def _dl(url, dest):
    _log(f"download start: {url}")
    urllib.request.urlretrieve(url, dest)
    _log(f"download done: {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")


def _zfile(rec, fname, dest):
    _dl(f"https://zenodo.org/api/records/{rec}/files/{fname}/content", dest)


def _load_clips(sysdir):
    clips = []
    for f in sorted(glob.glob(os.path.join(sysdir, "*.npy"))):
        a = np.load(f)
        if a.ndim == 3:
            a = a.reshape(a.shape[0], -1)
        clips.append(a.astype(np.float32))
    return clips


def _clip_windows(a, n_frames, mean, std):
    """windows for a single clip [T, D] -> [k, n_frames, D] normalized."""
    a = (a - mean) / std
    n = a.shape[0] // n_frames
    return np.asarray([a[i * n_frames:(i + 1) * n_frames] for i in range(n)], np.float32) \
        if n else np.zeros((0, n_frames, mean.shape[0]), np.float32)


def _frechet(A, B):
    from scipy import linalg
    mu1, mu2 = A.mean(0), B.mean(0)
    s1, s2 = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1 + s2 - 2 * covmean))


def _kendall(r1, r2):
    from itertools import combinations
    common = [x for x in r1 if x in r2]
    if len(common) < 2:
        return None
    p1 = {x: i for i, x in enumerate(r1)}
    p2 = {x: i for i, x in enumerate(r2)}
    c = d = 0
    for a, b in combinations(common, 2):
        s = np.sign(p1[a] - p1[b]) * np.sign(p2[a] - p2[b])
        c += s > 0
        d += s < 0
    return (c - d) / (c + d) if (c + d) else None


def _human_scores(hl, systems):
    scores = {}
    for rec in hl if isinstance(hl, list) else []:
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


def solve(config):
    import torch
    from itertools import combinations
    work = os.path.abspath("work"); os.makedirs(work, exist_ok=True)
    rep = {"errors": [], "mf1_window_matched": {}, "mf2_epochs": {}, "mf3_fgd_ci": {}, "analysis": {}}

    _zfile(MOTION_REC, f"{TIER}_npy.zip", os.path.join(work, "m.zip"))
    with zipfile.ZipFile(os.path.join(work, "m.zip")) as z:
        z.extractall(work)
    root = os.path.join(work, f"{TIER}_npy")
    systems = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    gt = next((s for s in systems if s.endswith("NA")), systems[0])
    clips = {s: _load_clips(os.path.join(root, s)) for s in systems}
    pose_dim = clips[gt][0].shape[1]
    allframes = np.concatenate([c for s in systems for c in clips[s]], 0)
    mean, std = allframes.mean(0), allframes.std(0) + 1e-6
    _log(f"{len(systems)} systems, gt={gt}, pose_dim={pose_dim}")

    subprocess.run(["git", "clone", "--depth", "1", FGD_REPO, os.path.join(work, "fgd")],
                   check=True, capture_output=True, timeout=300)
    sys.path.insert(0, os.path.join(work, "fgd", "FGD"))
    from embedding_net import EmbeddingNet
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _log(f"device={dev}")

    # human ranking
    try:
        _zfile(RATINGS_REC, f"{TIER}_human-likeness.json", os.path.join(work, "hl.json"))
        human = _human_scores(json.load(open(os.path.join(work, "hl.json"))), systems)
        hr = [k for k, _ in sorted(human.items(), key=lambda kv: -kv[1])]
        hrs = [k for k in hr if k != gt]
        rep["human_ranking"] = hr
    except Exception as e:
        rep["errors"].append(f"human: {e!r}"); hrs = []

    def sys_windows(s, n_frames):
        return [w for c in clips[s] for w in [_clip_windows(c, n_frames, mean, std)] if w.shape[0]]

    def all_windows(s, n_frames):
        ws = [_clip_windows(c, n_frames, mean, std) for c in clips[s]]
        ws = [w for w in ws if w.shape[0]]
        return np.concatenate(ws, 0) if ws else np.zeros((0, n_frames, pose_dim), np.float32)

    def train_ae(pool, n_frames, seed, epochs):
        torch.manual_seed(seed); np.random.seed(seed)
        net = EmbeddingNet(pose_dim, n_frames).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.5, 0.999))
        X = torch.tensor(pool, device=dev)
        for _ in range(epochs):
            perm = torch.randperm(X.shape[0])
            for i in range(0, X.shape[0], 64):
                b = X[perm[i:i + 64]]
                if b.shape[0] < 2:
                    continue
                out = net(b); recon = out[1] if isinstance(out, (tuple, list)) else out
                loss = torch.nn.functional.mse_loss(recon, b)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); return net

    def feats(net, W):
        if W.shape[0] == 0:
            return np.zeros((0, 32), np.float32)
        with torch.no_grad():
            out = net(torch.tensor(W, device=dev))
            return (out[0] if isinstance(out, (tuple, list)) else out).cpu().numpy()

    def rank_and_tau(fgd):
        r = [k for k, _ in sorted(fgd.items(), key=lambda kv: kv[1]) if k != gt]
        return r, (_kendall(r, hrs) if hrs else None)

    # ---- MF1: window trend with matched per-system window COUNT ----
    CHUNKS = [30, 60, 90]
    per_sys_counts = {n: min(all_windows(s, n).shape[0] for s in systems) for n in CHUNKS}
    match_k = min(per_sys_counts.values())   # windows per system, equal across lengths
    _log(f"MF1 matched per-system window count K={match_k} (raw per-sys min by len: {per_sys_counts})")
    for n in CHUNKS:
        rng = np.random.RandomState(0)
        # subsample each system to match_k windows; pool for training + per-system for scoring
        sysW = {}
        for s in systems:
            W = all_windows(s, n)
            idx = rng.choice(W.shape[0], size=match_k, replace=False) if W.shape[0] >= match_k else np.arange(W.shape[0])
            sysW[s] = W[idx]
        pool = np.concatenate([sysW[s] for s in systems], 0)
        net = train_ae(pool, n, 0, 10)
        gtf = feats(net, sysW[gt])
        fgd = {s: _frechet(feats(net, sysW[s]), gtf) for s in systems if sysW[s].shape[0] >= 2}
        _, tau = rank_and_tau(fgd)
        rep["mf1_window_matched"][f"n{n}"] = {"tau_vs_human": tau, "k_per_system": int(match_k),
                                              "total_windows": int(pool.shape[0])}
        _log(f"MF1 n{n}: matched-K tau_vs_human={tau}")

    # ---- MF2: epoch ablation on the seed flip (n30, feat, seeds {0,1}) ----
    n = 30
    pools = {s: all_windows(s, n) for s in systems}
    trainpool = np.concatenate([pools[s] for s in systems], 0)
    for epochs in [10, 50]:
        winners = {}
        gaps = {}
        for seed in [0, 1]:
            net = train_ae(trainpool, n, seed, epochs)
            gtf = feats(net, pools[gt])
            fgd = {s: _frechet(feats(net, pools[s]), gtf) for s in systems if pools[s].shape[0] >= 2}
            r, _ = rank_and_tau(fgd)
            winners[seed] = r[0] if r else None
            # record USQ/USM style top-2 gap generically: best two synthetic
            gaps[seed] = {r[0]: fgd.get(r[0]), r[1]: fgd.get(r[1])} if len(r) >= 2 else {}
        flip = winners.get(0) != winners.get(1)
        rep["mf2_epochs"][f"e{epochs}"] = {"winner_seed0": winners.get(0), "winner_seed1": winners.get(1),
                                           "seed_flip": bool(flip), "top2_fgd": gaps}
        _log(f"MF2 epochs={epochs}: winners s0={winners.get(0)} s1={winners.get(1)} flip={flip}")

    # ---- MF3: per-clip bootstrap CI on FGD values (n30, feat, seed0, 10 epochs) ----
    net = train_ae(trainpool, n, 0, 10)
    gtf_all = feats(net, pools[gt])
    # per-clip feature banks (list of per-clip window features) so we can resample clips
    clip_feats = {s: [feats(net, _clip_windows(c, n, mean, std)) for c in clips[s]] for s in systems}
    clip_feats = {s: [cf for cf in v if cf.shape[0] > 0] for s, v in clip_feats.items()}
    B = 500
    boot = {s: [] for s in systems}
    n_clips = {s: len(clip_feats[s]) for s in systems}
    rng = np.random.RandomState(0)
    for _ in range(B):
        for s in systems:
            cf = clip_feats[s]
            if len(cf) < 2:
                continue
            idx = rng.randint(0, len(cf), size=len(cf))
            F = np.concatenate([cf[i] for i in idx], 0)
            boot[s].append(_frechet(F, gtf_all))
    def ci(v):
        v = sorted(v); return [float(np.median(v)), float(v[int(.025 * len(v))]), float(v[int(.975 * len(v))])] if v else None
    rep["mf3_fgd_ci"] = {s: ci(boot[s]) for s in systems if boot[s]}
    # USQ vs USM gap CI (best two synthetic by median)
    synth = [s for s in systems if s != gt and boot[s]]
    synth.sort(key=lambda s: np.median(boot[s]))
    if len(synth) >= 2:
        a, b = synth[0], synth[1]
        # paired gap across bootstrap (independent resamples -> approximate)
        m = min(len(boot[a]), len(boot[b]))
        gap = sorted(boot[b][i] - boot[a][i] for i in range(m))
        rep["mf3_gap"] = {"best": a, "second": b,
                          "gap_median": float(np.median(gap)),
                          "gap_ci": [float(gap[int(.025 * m)]), float(gap[int(.975 * m)])],
                          "p_gap_gt_0": float(sum(g > 0 for g in gap) / m),
                          "ci_overlap": bool(rep["mf3_fgd_ci"][a][2] >= rep["mf3_fgd_ci"][b][1])}
        _log(f"MF3 best={a} second={b} gap_median={np.median(gap):.3f} p(gap>0)={sum(g>0 for g in gap)/m:.3f}")

    rep["analysis"] = {"mf1": rep["mf1_window_matched"], "mf2": rep["mf2_epochs"],
                       "mf3_gap": rep.get("mf3_gap")}
    return rep
