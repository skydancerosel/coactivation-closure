"""
select_closure_candidate.py

Step 1 of the OLMoE route-conditional closure test. Refit Ising on
cluster 1's 637 natural-text inputs, spectral-cluster at k=4, and
report each sub-cluster's:
  - size
  - capability-class composition (per the natural-text mechinterp)
  - isolation ratio (within-sub-cluster mean |J| / outside mean |J|)
  - layer span

Pick the candidate: the sub-cluster with the highest combined
purity + isolation, of size 5-30 heads (to ensure minimality so the
closure test is interpretable like OLMo cluster 2).

Output: results/olmoe_route_conditioned/closure_candidate.json
"""

import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/olmoe_route_conditioned")
    ap.add_argument("--mechinterp-json", default="/Volumes/Brandy/mini_gpt/.claude/worktrees/nostalgic-lederberg-80a58d/olmoe_mechinterp_naturaltext.json")
    ap.add_argument("--k-spectral", type=int, default=4)
    ap.add_argument("--min-size", type=int, default=5)
    ap.add_argument("--max-size", type=int, default=30)
    args = ap.parse_args()

    rd = Path(args.results_dir)
    attn = np.load(rd / "attn_at_query.npy")
    cluster_assigns = json.load(open(rd / "route_cluster_assignments.json"))
    labels_K4 = np.array(cluster_assigns["4"])

    cluster1_mask = labels_K4 == 1
    n_in = int(cluster1_mask.sum())
    print(f"cluster 1 (route stratum): n={n_in} examples")

    sub_spins = per_head_median_split(template_free_signal(attn[cluster1_mask]))
    n_layer = attn.shape[1]
    n_head = attn.shape[2]
    F = n_layer * n_head
    print(f"  spins shape: {sub_spins.shape}, F={F}")

    print("\nrefitting Ising on cluster 1...")
    J = fit_ising(sub_spins)
    np.save(rd / "J_cluster1.npy", J)
    print(f"  ||J||_F={np.linalg.norm(J):.3f}")

    print(f"\nspectral-clustering J at k={args.k_spectral}...")
    A = np.abs(J) + 1e-12
    np.fill_diagonal(A, 0)
    sc = SpectralClustering(n_clusters=args.k_spectral, affinity="precomputed",
                             assign_labels="kmeans", random_state=0, n_init=1)
    sub_labels = sc.fit_predict(A)

    mech = json.load(open(args.mechinterp_json))
    gt_map = {(c["layer"], c["head"]): c["classification"]
              for c in mech.get("classifications", [])}

    print(f"\nsub-cluster analysis (k_spectral={args.k_spectral}):")
    sub_clusters = []
    for c in sorted(set(sub_labels)):
        members = np.where(sub_labels == c)[0]
        if len(members) == 0:
            continue
        head_list = [(int(m // n_head), int(m % n_head)) for m in members]
        layers = sorted(set(L for L, H in head_list))

        # class composition
        composition = {}
        n_classified = 0
        for f in members:
            L, H = divmod(f, n_head)
            cls = gt_map.get((L, H))
            if cls is not None:
                composition[cls] = composition.get(cls, 0) + 1
                n_classified += 1

        # isolation
        member_set = set(members)
        within = []
        outside = []
        for i in members:
            for j in members:
                if i >= j: continue
                within.append(abs(J[i, j]))
            for j in range(F):
                if j in member_set: continue
                outside.append(abs(J[i, j]))
        within_mean = float(np.mean(within)) if within else 0.0
        outside_mean = float(np.mean(outside)) if outside else 1e-12
        iso_ratio = within_mean / max(outside_mean, 1e-12)

        # purity (largest class fraction, among classified)
        purity = max(composition.values()) / n_classified if n_classified else 0.0

        rec = {
            "sub_cluster": int(c),
            "size": len(members),
            "layers": layers,
            "heads": head_list,
            "composition_classified": composition,
            "n_classified": n_classified,
            "purity_classified": float(purity),
            "isolation_ratio": iso_ratio,
            "within_mean_J": within_mean,
            "outside_mean_J": outside_mean,
        }
        sub_clusters.append(rec)
        comp_str = ", ".join(f"{k}={v}" for k, v in
                              sorted(composition.items(), key=lambda x: -x[1]))
        print(f"  c{c}: size={len(members)}  layers={layers}")
        print(f"       composition (top-K classified n={n_classified}): {comp_str}")
        print(f"       purity={purity:.2f}, isolation ratio={iso_ratio:.2f}x "
              f"(within {within_mean:.3f} vs outside {outside_mean:.3f})")

    # pick the candidate
    candidates = [s for s in sub_clusters
                  if args.min_size <= s["size"] <= args.max_size]
    if not candidates:
        print(f"\nNo sub-cluster in size range [{args.min_size}, {args.max_size}]")
        print("Falling back to highest isolation × purity, any size...")
        candidates = sub_clusters
    # combined score: purity × isolation × log(size) clamped (so we don't pick tiny noisy clusters)
    for c in candidates:
        size_factor = min(c["size"], args.max_size) / args.max_size
        c["combined_score"] = c["purity_classified"] * c["isolation_ratio"] * size_factor
    best = max(candidates, key=lambda x: x["combined_score"])

    print(f"\n=== CHOSEN CANDIDATE for closure test ===")
    print(f"  sub-cluster {best['sub_cluster']}, size={best['size']} heads")
    print(f"  layers: {best['layers']}")
    print(f"  heads: {best['heads']}")
    print(f"  composition: {best['composition_classified']}")
    print(f"  purity={best['purity_classified']:.2f}, "
          f"isolation={best['isolation_ratio']:.2f}x")

    out = {
        "n_cluster1": n_in,
        "k_spectral": args.k_spectral,
        "sub_clusters": sub_clusters,
        "candidate": best,
    }
    out_path = rd / "closure_candidate.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
