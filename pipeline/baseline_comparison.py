"""
baseline_comparison.py

Bhalla et al. compare their Ising-coupling community detection against
simpler baselines (decoder cosine similarity, Pearson correlation, PMI).
Run the same comparison for attention-head Ising couplings.

For each completed model, compare:
  1. Ising J (our main result)
  2. Pearson correlation of binary spins
  3. Pearson correlation of raw max-attn signal
  4. Mutual information of binary spins (PMI proxy)

Same spectral clustering + metrics protocol.

Usage:
  python baseline_comparison.py --tag pythia_1b ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def per_head_median_split(signal):
    medians = np.median(signal, axis=0, keepdims=True)
    return np.where(signal > medians, 1.0, -1.0)


def template_free_signal(attn):
    return attn.max(axis=-1)  # (n, L, H)


def cluster_and_score(affinity, n_clusters, gt_map, n_layer, n_head):
    """Spectral cluster affinity matrix, score against gt_map."""
    A = np.abs(affinity) + 1e-12
    np.fill_diagonal(A, 0)
    sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed",
                             assign_labels="kmeans", random_state=0)
    labels = sc.fit_predict(A)

    gt, pred = [], []
    F = n_layer * n_head
    for f in range(F):
        L, H = divmod(f, n_head)
        if (L, H) in gt_map:
            gt.append(gt_map[(L, H)])
            pred.append(labels[f])
    if len(gt) == 0:
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
    purity = correct / len(gt)
    nmi = normalized_mutual_info_score(gt_int, pred_arr)
    ari = adjusted_rand_score(gt_int, pred_arr)
    return {"purity": float(purity), "nmi": float(nmi), "ari": float(ari),
            "labels": labels.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--mechinterp-json", required=True)
    ap.add_argument("--n-layer", type=int, required=True)
    ap.add_argument("--n-head", type=int, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    attn = np.load(results_dir / "attn_at_query.npy")
    J = np.load(results_dir / "J.npy")
    mech = json.load(open(args.mechinterp_json))
    gt_map = {(c["layer"], c["head"]): c["classification"]
              for c in mech.get("classifications", [])}

    # build alternative affinities
    print(f"\n=== {args.tag} baseline comparison ===")
    n = attn.shape[0]
    F = args.n_layer * args.n_head
    signal_raw = template_free_signal(attn).reshape(n, F)
    spins = per_head_median_split(signal_raw)

    # Pearson on raw signal
    corr_raw = np.corrcoef(signal_raw.T)  # (F, F)
    # Pearson on binary spins
    corr_spin = np.corrcoef(spins.T)
    # Pairwise mutual information via 2x2 contingency
    # quick MI: for each pair, build 2x2 joint table on +1/-1
    def pairwise_mi(s):
        # s: (n, F) in {-1, +1}
        n_, F_ = s.shape
        S = (s + 1) // 2  # {0, 1}
        S = S.astype(np.int32)
        marg = S.mean(axis=0)  # P(s_i = 1) per spin
        MI = np.zeros((F_, F_))
        for i in range(F_):
            for j in range(i + 1, F_):
                p11 = float(((S[:, i] == 1) & (S[:, j] == 1)).mean())
                p10 = float(((S[:, i] == 1) & (S[:, j] == 0)).mean())
                p01 = float(((S[:, i] == 0) & (S[:, j] == 1)).mean())
                p00 = float(((S[:, i] == 0) & (S[:, j] == 0)).mean())
                pi1, pj1 = marg[i], marg[j]
                pi0, pj0 = 1 - pi1, 1 - pj1
                mi = 0.0
                for p, pi, pj in [(p11, pi1, pj1), (p10, pi1, pj0),
                                    (p01, pi0, pj1), (p00, pi0, pj0)]:
                    if p > 0 and pi > 0 and pj > 0:
                        mi += p * np.log(p / (pi * pj))
                MI[i, j] = MI[j, i] = mi
        return MI
    print("  computing pairwise MI (this may take a minute for large F)...")
    MI = pairwise_mi(spins.astype(np.int32))

    methods = {
        "ising_J": J,
        "pearson_raw": corr_raw,
        "pearson_spin": corr_spin,
        "mutual_info": MI,
    }

    summary = {}
    for name, A in methods.items():
        summary[name] = {}
        for k in args.ks:
            res = cluster_and_score(A, k, gt_map, args.n_layer, args.n_head)
            if res is None:
                continue
            summary[name][str(k)] = {
                "purity": res["purity"],
                "nmi": res["nmi"],
                "ari": res["ari"],
            }
            print(f"  {name:14s} k={k}: purity={res['purity']:.3f} "
                  f"nmi={res['nmi']:.3f} ari={res['ari']:.3f}")
        print()

    out = results_dir / "baseline_comparison.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
