"""MUTABLE — fgdaudit s5.5b: FEATURE-EXTRACTOR TRAINING-CORPUS axis.

s5.3 varied the reference tool's built-in knobs (chunk_len, seed, feat/raw) and found the
FGD leaderboard already reorders (Kendall τ 0.73-0.87, top-1 flips). This run isolates a
knob the tool treats as fixed-and-neutral: **what corpus the EmbeddingNet feature extractor
is trained on.** We hold chunk_len=30 (best human agreement in s5.5a) and feat-space (the
canonical FGD) fixed, and train the AE on different corpora drawn from the GENEA-2022
upper-body pool, then re-rank all 11 systems by FGD vs natural (UNA):

  all       - pooled all 11 systems (the s5.3 baseline)
  natural   - UNA only (the "gold reference" corpus)
  synthetic - all submissions except UNA
  halfA     - even-indexed systems (disjoint split)
  halfB     - odd-indexed systems (disjoint split)

Each corpus is trained with seeds {0,1} and the two are pooled into one ranking per corpus
(seed is not the axis here). report["configs"] is keyed by corpus so the fixed score.py
reads pairwise-CORPUS Kendall τ (does the training corpus change the leaderboard?),
top-1 flips, and per-corpus τ vs the human ranking.
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

socket.setdefaulttimeout(180)

MOTION_REC, RATINGS_REC = "6973297", "6940057"
TIER = "upper-body"
FGD_REPO = "https://github.com/genea-workshop/genea_numerical_evaluations"
CHUNK = 30            # fixed (best human agreement, s5.5a)
SEEDS = [0, 1]        # pooled per corpus, not an axis


def _log(msg):
    line = f"[corpus {time.strftime('%H:%M:%S')}] {msg}"
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


def _corpora(systems):
    ss = sorted(systems)
    nat = [s for s in ss if s.endswith("NA")] or [ss[0]]
    return {
        "all": ss,
        "natural": nat,
        "synthetic": [s for s in ss if not s.endswith("NA")],
        "halfA": ss[::2],
        "halfB": ss[1::2],
    }


def solve(config):
    import torch
    work = os.path.abspath("work"); os.makedirs(work, exist_ok=True)
    report = {"errors": [], "systems": [], "configs": {}, "human": {}, "analysis": {}}

    mzip = os.path.join(work, "m.zip")
    _zfile(MOTION_REC, f"{TIER}_npy.zip", mzip)
    with zipfile.ZipFile(mzip) as z:
        z.extractall(work)
    root = os.path.join(work, f"{TIER}_npy")
    systems = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    report["systems"] = systems
    gt_name = next((s for s in systems if s.endswith("NA")), systems[0])
    clips = {s: _load_system_clips(os.path.join(root, s)) for s in systems}
    pose_dim = clips[gt_name][0].shape[1]
    report["pose_dim"] = pose_dim; report["gt_system"] = gt_name
    _log(f"{len(systems)} systems, gt={gt_name}, pose_dim={pose_dim}")

    # normalization from the FULL pool (fixed across corpora so only the AE differs)
    allframes = np.concatenate([c for s in systems for c in clips[s]], 0)
    mean, std = allframes.mean(0), allframes.std(0) + 1e-6

    subprocess.run(["git", "clone", "--depth", "1", FGD_REPO, os.path.join(work, "fgd")],
                   check=True, capture_output=True, timeout=300)
    sys.path.insert(0, os.path.join(work, "fgd", "FGD"))
    from embedding_net import EmbeddingNet
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _log(f"device={dev}")

    def train_ae(train_systems, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        pooled = _windows([c for s in train_systems for c in clips[s]], CHUNK, mean, std)
        net = EmbeddingNet(pose_dim, CHUNK).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.5, 0.999))
        X = torch.tensor(pooled, device=dev)
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

    corpora = _corpora(systems)
    gtW = _windows(clips[gt_name], CHUNK, mean, std)
    for cname, train_systems in corpora.items():
        try:
            # pool feature vectors across seeds for a stable per-corpus ranking
            per_sys_feats = {s: [] for s in systems}
            gt_feats = []
            for seed in SEEDS:
                net = train_ae(train_systems, seed)
                gt_feats.append(feats(net, gtW))
                for s in systems:
                    W = _windows(clips[s], CHUNK, mean, std)
                    if W.shape[0] >= 2:
                        per_sys_feats[s].append(feats(net, W))
            gt_feat = np.concatenate(gt_feats, 0)
            fgd = {}
            for s in systems:
                if per_sys_feats[s]:
                    fgd[s] = _frechet(np.concatenate(per_sys_feats[s], 0), gt_feat)
            report["configs"][cname] = fgd
            best = min(fgd, key=fgd.get) if fgd else None
            _log(f"corpus {cname} (trained on {len(train_systems)} systems): best(low FGD)={best}")
        except Exception as e:
            import traceback
            report["errors"].append(f"corpus {cname}: {e!r}\n{traceback.format_exc()[-400:]}")

    hj = os.path.join(work, "hl.json")
    try:
        _zfile(RATINGS_REC, f"{TIER}_human-likeness.json", hj)
        report["human"] = _human_scores(json.load(open(hj)), systems)
        _log(f"human ranking parsed: {len(report['human'])} systems")
    except Exception as e:
        report["errors"].append(f"human json: {e!r}")

    report["analysis"] = _analyze(report["configs"], report.get("human", {}))
    return report


def _human_scores(hl, systems):
    """GENEA human-likeness = per-system median rating; system code = stimulus prefix;
    exclude the id=='training' attention trial."""
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
    return [k for k, _ in sorted(d.items(), key=lambda kv: kv[1])]


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
    # synthetic-only (exclude the natural reference) — the decision-relevant ranking
    gt = next((s for s in (ranks[cfgs[0]] if cfgs else []) if s.endswith("NA")), None)
    if gt:
        sranks = {c: [s for s in ranks[c] if s != gt] for c in cfgs}
        st = [t for a, b in combinations(cfgs, 2)
              if (t := _kendall(sranks[a], sranks[b])) is not None]
        out["synth_pairwise_tau_median"] = float(np.median(st)) if st else None
        out["synth_pairwise_tau_min"] = float(np.min(st)) if st else None
        out["synth_top1_by_config"] = {c: (sranks[c][0] if sranks[c] else None) for c in cfgs}
        out["synth_top1_distinct"] = len(set(v for v in out["synth_top1_by_config"].values() if v))
    if human:
        hr = [k for k, _ in sorted(human.items(), key=lambda kv: -kv[1])]
        out["human_ranking"] = hr
        out["config_vs_human_tau"] = {c: _kendall(ranks[c], hr) for c in cfgs}
        if gt:
            hrs = [k for k in hr if k != gt]
            out["config_vs_human_tau_synth"] = {c: _kendall([s for s in ranks[c] if s != gt], hrs) for c in cfgs}
    return out
