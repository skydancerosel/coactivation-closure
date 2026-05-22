"""
route_conditioned_analysis_only.py

Pick up where route_conditioned_ising_olmoe.py crashed (sklearn KMeans
multiprocessing segfaults after MPS forward pass). Load saved attn +
routes tensors from disk; run all clustering + Ising fits in pure CPU
with n_init=1 to avoid the multiprocessing issue.

Strategies:
- BASELINE marginal Ising (re-fit for clean baseline)
- Strategy A: route-stratified Ising per route-cluster
- Strategy B: pooled Ising with route-cluster one-hot covariates

Output: results/olmoe_route_conditioned/results.json (and per-strategy
J matrices).
"""

import os
# avoid sklearn / joblib multiprocessing entirely
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def template_free_signal(attn):
    """attn: (N, L, H, T) numpy. Returns (N, L*H) max-attn signal."""
    return attn.max(axis=-1).reshape(attn.shape[0], -1)


def per_head_median_split(signal):
    """signal: (N, F) numpy. Returns spins in {-1, +1}."""
    medians = np.median(signal, axis=0, keepdims=True)
    return np.where(signal > medians, 1.0, -1.0)


def fit_ising_pseudolikelihood(spins, extra_X=None, l2=1e-3, verbose=False):
    spins = np.asarray(spins, dtype=np.float64)
    n, F = spins.shape
    J = np.zeros((F, F))
    h = np.zeros(F)
    C = 1.0 / max(l2, 1e-12)
    if extra_X is not None:
        extra_X = np.asarray(extra_X, dtype=np.float64)
    t0 = time.time()
    for i in range(F):
        y = (spins[:, i] > 0).astype(np.int32)
        if y.min() == y.max():
            continue
        mask = np.ones(F, dtype=bool); mask[i] = False
        X_spins = spins[:, mask]
        if extra_X is not None:
            X = np.hstack([X_spins, extra_X])
        else:
            X = X_spins
        clf = LogisticRegression(C=C, fit_intercept=True, solver="lbfgs",
                                  max_iter=500, n_jobs=1)
        clf.fit(X, y)
        idxs = np.where(mask)[0]
        j_row = np.zeros(F)
        j_row[idxs] = clf.coef_[0][:F - 1] / 2.0
        J[i, :] = j_row
        h[i] = clf.intercept_[0] / 2.0
        if verbose and (i + 1) % 32 == 0:
            print(f"    fit {i+1}/{F} spins ({(time.time()-t0)/(i+1)*1000:.0f} ms/spin)",
                  flush=True)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    return J, h


def spectral_cluster_no_mp(J, ks):
    """SpectralClustering with single-threaded settings."""
    A = np.abs(J) + 1e-12
    np.fill_diagonal(A, 0)
    out = {}
    for k in ks:
        try:
            sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                     assign_labels="kmeans",
                                     random_state=0, n_init=1)
            out[k] = sc.fit_predict(A).tolist()
        except Exception as e:
            print(f"  spectral k={k} failed: {e}")
            out[k] = None
    return out


def kmeans_no_mp(features, K, random_state=0):
    """KMeans with single-thread settings to dodge macOS multiprocessing segfault."""
    km = KMeans(n_clusters=K, random_state=random_state, n_init=1,
                init="random", max_iter=300)
    return km.fit_predict(features)


def score_against_gt(labels, mech_json_path, n_layer, n_head):
    mech = json.load(open(mech_json_path))
    gt_map = {(c["layer"], c["head"]): c["classification"]
              for c in mech.get("classifications", [])}
    F = n_layer * n_head
    gt, pred = [], []
    for f in range(F):
        L, H = divmod(f, n_head)
        if (L, H) in gt_map:
            gt.append(gt_map[(L, H)])
            pred.append(labels[f])
    if not gt:
        return None
    classes = sorted(set(gt))
    cls2int = {c: i for i, c in enumerate(classes)}
    gt_int = np.array([cls2int[g] for g in gt])
    pred_arr = np.array(pred)
    correct = 0
    for c in np.unique(pred_arr):
        idx = pred_arr == c
        if idx.sum() == 0:
            continue
        members = np.array(gt)[idx]
        unique, counts = np.unique(members, return_counts=True)
        correct += counts.max()
    return {
        "n_evaluated": len(gt),
        "purity": float(correct / len(gt)),
        "nmi": float(normalized_mutual_info_score(gt_int, pred_arr)),
        "ari": float(adjusted_rand_score(gt_int, pred_arr)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/olmoe_route_conditioned")
    ap.add_argument("--mechinterp-json", default="/Volumes/Brandy/mini_gpt/.claude/worktrees/nostalgic-lederberg-80a58d/olmoe_mechinterp_naturaltext.json")
    ap.add_argument("--ks-cluster", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--ks-spectral", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--min-cluster-size", type=int, default=80)
    args = ap.parse_args()

    rd = Path(args.results_dir)
    attn = np.load(rd / "attn_at_query.npy")
    routes = np.load(rd / "routes_at_query.npy")
    print(f"loaded attn={attn.shape} routes={routes.shape}")

    N, n_layer, n_head, T = attn.shape
    n_expert = routes.shape[2]
    F = n_layer * n_head

    # signals + spins
    sig = template_free_signal(attn)
    spins = per_head_median_split(sig)
    print(f"  spins shape: {spins.shape}")

    # routing entropy
    p = routes + 1e-12
    ent_per_layer = (-(p * np.log(p)).sum(axis=2)).mean(axis=0)
    overall_ent = float(ent_per_layer.mean())
    max_ent = float(np.log(n_expert))
    print(f"  mean per-layer entropy: {overall_ent:.3f} / max {max_ent:.3f} "
          f"({100*overall_ent/max_ent:.1f}% of max)")

    # --- BASELINE: marginal Ising (clean re-fit) ---
    print("\n[BASELINE] marginal Ising...")
    t0 = time.time()
    J_marg, _ = fit_ising_pseudolikelihood(spins, extra_X=None, l2=args.l2)
    np.save(rd / "J_marginal_v2.npy", J_marg)
    print(f"  done in {time.time()-t0:.0f}s, ||J||_F={np.linalg.norm(J_marg):.3f}")
    marg_clusters = spectral_cluster_no_mp(J_marg, args.ks_spectral)
    marg_metrics = {}
    for k, labels in marg_clusters.items():
        if labels is None: continue
        m = score_against_gt(labels, args.mechinterp_json, n_layer, n_head)
        if m:
            marg_metrics[str(k)] = m
            print(f"  marginal k={k}: purity={m['purity']:.3f} nmi={m['nmi']:.3f} "
                  f"ari={m['ari']:.3f}")

    route_features = routes.reshape(N, -1).astype(np.float32)

    # --- STRATEGY A: route-stratified ---
    print("\n[STRATEGY A] route-stratified Ising...")
    strategy_A = {}
    cluster_assignments = {}
    for K in args.ks_cluster:
        print(f"\n  K_route_clusters={K}:")
        cluster_labels = kmeans_no_mp(route_features, K)
        cluster_assignments[str(K)] = cluster_labels.tolist()

        per_cluster = []
        valid_aris = []
        for c in range(K):
            mask = cluster_labels == c
            n_in = int(mask.sum())
            if n_in < args.min_cluster_size:
                print(f"    cluster {c}: n={n_in} (skip, below min {args.min_cluster_size})")
                per_cluster.append({"cluster": c, "n_examples": n_in, "skipped": True})
                continue
            sub_spins = spins[mask]
            J_sub, _ = fit_ising_pseudolikelihood(sub_spins, extra_X=None,
                                                   l2=args.l2)
            sub_clusters = spectral_cluster_no_mp(J_sub, args.ks_spectral)
            best = {"k_spectral": None, "ari": -1, "nmi": 0, "purity": 0}
            for ks, labels in sub_clusters.items():
                if labels is None: continue
                m = score_against_gt(labels, args.mechinterp_json,
                                       n_layer, n_head)
                if m and m["ari"] > best["ari"]:
                    best = {"k_spectral": ks, **m}
            per_cluster.append({"cluster": c, "n_examples": n_in, "best": best})
            valid_aris.append(best["ari"])
            print(f"    cluster {c}: n={n_in} | "
                  f"best k_spectral={best['k_spectral']} "
                  f"ari={best['ari']:.3f} nmi={best['nmi']:.3f}")

        if valid_aris:
            mean_ari = float(np.mean(valid_aris))
            max_ari = float(np.max(valid_aris))
            print(f"  K={K}: mean within-cluster ARI = {mean_ari:.3f}, max = {max_ari:.3f}")
        else:
            mean_ari = max_ari = None
        strategy_A[str(K)] = {
            "per_cluster": per_cluster,
            "mean_within_cluster_ari": mean_ari,
            "max_within_cluster_ari": max_ari,
        }

    # --- STRATEGY B: pooled with route-cluster one-hot covariates ---
    print("\n[STRATEGY B] pooled Ising with route-cluster covariates...")
    strategy_B = {}
    for K in args.ks_cluster:
        print(f"\n  K_route_clusters={K}:")
        cl = kmeans_no_mp(route_features, K)
        onehot = np.zeros((N, K - 1), dtype=np.float64)
        for k in range(1, K):
            onehot[cl == k, k - 1] = 1.0
        J_cov, _ = fit_ising_pseudolikelihood(spins, extra_X=onehot,
                                                l2=args.l2)
        np.save(rd / f"J_cov_K{K}.npy", J_cov)
        cov_clusters = spectral_cluster_no_mp(J_cov, args.ks_spectral)
        cov_metrics = {}
        for ks, labels in cov_clusters.items():
            if labels is None: continue
            m = score_against_gt(labels, args.mechinterp_json, n_layer, n_head)
            if m:
                cov_metrics[str(ks)] = m
                print(f"    k_spectral={ks}: purity={m['purity']:.3f} "
                      f"nmi={m['nmi']:.3f} ari={m['ari']:.3f}")
        strategy_B[str(K)] = cov_metrics

    # --- save ---
    json.dump(cluster_assignments,
              open(rd / "route_cluster_assignments.json", "w"))
    summary = {
        "model": "allenai/OLMoE-1B-7B-0924",
        "n_layer": n_layer, "n_head": n_head, "n_expert": n_expert,
        "n_examples": int(N),
        "marginal_baseline_ari": marg_metrics,
        "strategy_A_stratified": strategy_A,
        "strategy_B_route_covariate": strategy_B,
        "overall_mean_routing_entropy": overall_ent,
        "max_possible_entropy": max_ent,
    }
    json.dump(summary, open(rd / "results.json", "w"), indent=2)
    print(f"\nsaved {rd / 'results.json'}")

    print("\n=== HEADLINE COMPARISON ===")
    marg_best = max((m["ari"] for m in marg_metrics.values()), default=-1)
    print(f"  marginal best ARI: {marg_best:.3f}")
    for K in args.ks_cluster:
        sA = strategy_A.get(str(K), {})
        if sA.get("max_within_cluster_ari") is not None:
            print(f"  Strategy A K={K}: max within-cluster ARI = "
                  f"{sA['max_within_cluster_ari']:.3f}, "
                  f"mean = {sA['mean_within_cluster_ari']:.3f}")
        sB = strategy_B.get(str(K), {})
        if sB:
            best_b = max((m["ari"] for m in sB.values()), default=-1)
            print(f"  Strategy B K={K}: best route-covariate ARI = {best_b:.3f}")


if __name__ == "__main__":
    main()
