"""
random_partition_null.py

Critical null control for the route-stratified Ising result. The previous
experiment found that one of K=4 route clusters (cluster 1, n=637) gives
within-cluster ARI = 0.191 vs marginal baseline 0.035. The question this
null answers: is the recovery specific to *route-based* stratification,
or does *any* K=4 stratification of the same data produce comparable
within-cluster ARI from sheer sample-size effects?

Protocol:
- Load saved attn tensor for OLMoE natural text (cached in
  results/olmoe_route_conditioned/attn_at_query.npy).
- Build template-free signal and per-head median split → spins.
- For each random seed (0..9):
    - Randomly partition 2000 examples into K=4 groups (uniform sizes
      ~500 each).
    - For each group with n >= min_cluster_size, fit Ising on head spins,
      spectral cluster, score vs supervised classification.
    - Record max + mean within-group ARI for this seed.
- Report null distribution: P(max_within_group_ARI >= 0.191 | random
  partition).
- Compare to route-stratified result.

Output: results/olmoe_route_conditioned/random_partition_null.json
"""

import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def template_free_signal(attn):
    return attn.max(axis=-1).reshape(attn.shape[0], -1)


def per_head_median_split(signal):
    medians = np.median(signal, axis=0, keepdims=True)
    return np.where(signal > medians, 1.0, -1.0)


def fit_ising(spins, l2=1e-3):
    spins = np.asarray(spins, dtype=np.float64)
    n, F = spins.shape
    J = np.zeros((F, F))
    C = 1.0 / max(l2, 1e-12)
    for i in range(F):
        y = (spins[:, i] > 0).astype(np.int32)
        if y.min() == y.max():
            continue
        mask = np.ones(F, dtype=bool); mask[i] = False
        X = spins[:, mask]
        clf = LogisticRegression(C=C, fit_intercept=True, solver="lbfgs",
                                  max_iter=500, n_jobs=1)
        clf.fit(X, y)
        idxs = np.where(mask)[0]
        j_row = np.zeros(F)
        j_row[idxs] = clf.coef_[0] / 2.0
        J[i, :] = j_row
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    return J


def spectral_cluster_safe(J, ks):
    A = np.abs(J) + 1e-12
    np.fill_diagonal(A, 0)
    out = {}
    for k in ks:
        try:
            sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                     assign_labels="kmeans", random_state=0,
                                     n_init=1)
            out[k] = sc.fit_predict(A).tolist()
        except Exception:
            out[k] = None
    return out


def score_against_gt(labels, gt_map, n_layer, n_head):
    F = n_layer * n_head
    gt, pred = [], []
    for f in range(F):
        L, H = divmod(f, n_head)
        if (L, H) in gt_map:
            gt.append(gt_map[(L, H)]); pred.append(labels[f])
    if not gt: return None
    classes = sorted(set(gt))
    cls2int = {c: i for i, c in enumerate(classes)}
    gt_int = np.array([cls2int[g] for g in gt])
    pred_arr = np.array(pred)
    correct = 0
    for c in np.unique(pred_arr):
        idx = pred_arr == c
        if idx.sum() == 0: continue
        members = np.array(gt)[idx]
        unique, counts = np.unique(members, return_counts=True)
        correct += counts.max()
    return {
        "purity": float(correct / len(gt)),
        "nmi": float(normalized_mutual_info_score(gt_int, pred_arr)),
        "ari": float(adjusted_rand_score(gt_int, pred_arr)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/olmoe_route_conditioned")
    ap.add_argument("--mechinterp-json", default="/Volumes/Brandy/mini_gpt/.claude/worktrees/nostalgic-lederberg-80a58d/olmoe_mechinterp_naturaltext.json")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--ks-spectral", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--min-group-size", type=int, default=80)
    args = ap.parse_args()

    rd = Path(args.results_dir)
    attn = np.load(rd / "attn_at_query.npy")
    print(f"loaded attn shape {attn.shape}")

    N, n_layer, n_head, T = attn.shape
    spins = per_head_median_split(template_free_signal(attn))
    F = spins.shape[1]
    print(f"  spins shape: {spins.shape}, F={F}")

    mech = json.load(open(args.mechinterp_json))
    gt_map = {(c["layer"], c["head"]): c["classification"]
              for c in mech.get("classifications", [])}

    print(f"\nRunning {args.n_seeds} random K={args.K} partitions...")
    seed_summary = []
    for seed in range(args.n_seeds):
        rng = np.random.RandomState(seed)
        # Random partition: assign each example uniformly to one of K groups
        labels_partition = rng.randint(0, args.K, size=N)
        per_group_aris = []
        per_group = []
        t0 = time.time()
        for g in range(args.K):
            mask = labels_partition == g
            n_in = int(mask.sum())
            if n_in < args.min_group_size:
                per_group.append({"group": g, "n": n_in, "skipped": True})
                continue
            sub_spins = spins[mask]
            J = fit_ising(sub_spins, l2=args.l2)
            sub_clusters = spectral_cluster_safe(J, args.ks_spectral)
            best_ari = -1
            best_record = None
            for ks, labels in sub_clusters.items():
                if labels is None: continue
                m = score_against_gt(labels, gt_map, n_layer, n_head)
                if m and m["ari"] > best_ari:
                    best_ari = m["ari"]
                    best_record = {"k_spectral": ks, **m}
            per_group.append({"group": g, "n": n_in, "best": best_record})
            per_group_aris.append(best_ari)
        if not per_group_aris:
            print(f"  seed {seed}: no valid groups, skip")
            continue
        max_ari = float(np.max(per_group_aris))
        mean_ari = float(np.mean(per_group_aris))
        seed_summary.append({
            "seed": seed,
            "max_within_group_ari": max_ari,
            "mean_within_group_ari": mean_ari,
            "per_group": per_group,
        })
        print(f"  seed {seed}: {time.time()-t0:.0f}s | max ARI {max_ari:.3f}, "
              f"mean {mean_ari:.3f}")

    max_aris = np.array([s["max_within_group_ari"] for s in seed_summary])
    mean_aris = np.array([s["mean_within_group_ari"] for s in seed_summary])

    print("\n=== NULL DISTRIBUTION (random K=4 partitions) ===")
    print(f"  max within-group ARI:  mean {max_aris.mean():.3f} ± {max_aris.std():.3f}"
          f"  (min {max_aris.min():.3f}, max {max_aris.max():.3f})")
    print(f"  mean within-group ARI: mean {mean_aris.mean():.3f} ± {mean_aris.std():.3f}"
          f"  (min {mean_aris.min():.3f}, max {mean_aris.max():.3f})")

    OBSERVED_MAX = 0.191   # route-K=4 cluster 1
    OBSERVED_MEAN = 0.075  # route-K=4 mean across 4 clusters
    z_max = (OBSERVED_MAX - max_aris.mean()) / max(max_aris.std(), 1e-9)
    z_mean = (OBSERVED_MEAN - mean_aris.mean()) / max(mean_aris.std(), 1e-9)
    p_exceed_max = float(np.mean(max_aris >= OBSERVED_MAX))
    p_exceed_mean = float(np.mean(mean_aris >= OBSERVED_MEAN))

    print()
    print(f"  OBSERVED route-K=4 max  ARI: {OBSERVED_MAX:.3f}  →  z={z_max:+.2f}σ "
          f"(P[null ≥ observed] = {p_exceed_max:.2f})")
    print(f"  OBSERVED route-K=4 mean ARI: {OBSERVED_MEAN:.3f}  →  z={z_mean:+.2f}σ "
          f"(P[null ≥ observed] = {p_exceed_mean:.2f})")

    out = {
        "n_seeds": args.n_seeds,
        "K_partition": args.K,
        "spectral_ks_tested": args.ks_spectral,
        "per_seed": seed_summary,
        "null_max_ari_mean": float(max_aris.mean()),
        "null_max_ari_std": float(max_aris.std()),
        "null_max_ari_min": float(max_aris.min()),
        "null_max_ari_max": float(max_aris.max()),
        "null_mean_ari_mean": float(mean_aris.mean()),
        "null_mean_ari_std": float(mean_aris.std()),
        "observed_route_max_ari": OBSERVED_MAX,
        "observed_route_mean_ari": OBSERVED_MEAN,
        "z_score_max": float(z_max),
        "z_score_mean": float(z_mean),
        "p_value_max_exceed": p_exceed_max,
        "p_value_mean_exceed": p_exceed_mean,
    }
    json.dump(out, open(rd / "random_partition_null.json", "w"), indent=2)
    print(f"\nsaved {rd / 'random_partition_null.json'}")


if __name__ == "__main__":
    main()
