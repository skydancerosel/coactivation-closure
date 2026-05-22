"""
layer_residualized_clustering.py

Test the hypothesis from the writeup: communities are layer-dominated; if we
factor out layer co-activation, function-defined communities should emerge.

Two residualization strategies:
  (C) Block-mean residual J: subtract the per-(layer_i, layer_j) block mean
      from |J|. Removes "heads in same layer X couple at level Y on average"
      structure.
  (D) Within-layer Ising: cluster heads within each layer separately, see
      whether layer-local communities track function.

For each model + each strategy:
  - Re-cluster on the residualized affinity
  - Score against ground truth (top-K classifications)
  - Compare to baseline (full J) ARI / NMI

Usage:
  python layer_residualized_clustering.py --tag pythia_1b ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def block_mean_residual(absJ, n_layer, n_head):
    """For each (layer_i, layer_j) block of |J|, subtract the block's mean.
    Returns residual affinity with shape (F, F)."""
    F = n_layer * n_head
    out = absJ.copy()
    for li in range(n_layer):
        for lj in range(n_layer):
            i0, i1 = li * n_head, (li + 1) * n_head
            j0, j1 = lj * n_head, (lj + 1) * n_head
            block = absJ[i0:i1, j0:j1]
            # don't include diagonal in mean if li == lj
            if li == lj:
                mask = ~np.eye(n_head, dtype=bool)
                mean_val = block[mask].mean() if mask.sum() > 0 else 0.0
            else:
                mean_val = block.mean()
            out[i0:i1, j0:j1] = block - mean_val
    out = np.maximum(out, 0.0)  # spectral clustering needs non-negative
    np.fill_diagonal(out, 0.0)
    return out


def cluster_and_score(A, n_clusters, gt_map, n_layer, n_head):
    A_use = A + 1e-12
    np.fill_diagonal(A_use, 0)
    sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed",
                             assign_labels="kmeans", random_state=0)
    labels = sc.fit_predict(A_use)

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
    composition = {}
    for c in np.unique(pred_arr):
        idx = pred_arr == c
        members = np.array(gt)[idx]
        if len(members) == 0:
            continue
        unique, counts = np.unique(members, return_counts=True)
        correct += counts.max()
        composition[int(c)] = {str(u): int(v) for u, v in zip(unique, counts)}
    purity = correct / len(gt)
    nmi = normalized_mutual_info_score(gt_int, pred_arr)
    ari = adjusted_rand_score(gt_int, pred_arr)
    return {"purity": float(purity), "nmi": float(nmi), "ari": float(ari),
            "labels": labels.tolist(), "composition": composition}


def within_layer_cluster(J, n_layer, n_head, k_per_layer=2):
    """Cluster heads within each layer separately. Returns global labels
    where label = layer * k_per_layer + within-layer cluster."""
    labels = np.zeros(n_layer * n_head, dtype=int)
    for L in range(n_layer):
        i0, i1 = L * n_head, (L + 1) * n_head
        if k_per_layer >= n_head:
            # too few heads — give each its own label
            labels[i0:i1] = L * k_per_layer + np.arange(n_head) % k_per_layer
            continue
        block = np.abs(J[i0:i1, i0:i1])
        np.fill_diagonal(block, 0)
        sc = SpectralClustering(n_clusters=k_per_layer, affinity="precomputed",
                                 assign_labels="kmeans", random_state=0)
        try:
            local_labels = sc.fit_predict(block + 1e-12)
        except Exception:
            local_labels = np.zeros(n_head, dtype=int)
        labels[i0:i1] = L * k_per_layer + local_labels
    return labels


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
    J = np.load(results_dir / "J.npy")
    mech = json.load(open(args.mechinterp_json))
    gt_map = {(c["layer"], c["head"]): c["classification"]
              for c in mech.get("classifications", [])}

    print(f"\n=== {args.tag} layer-residualized clustering ===\n")

    absJ = np.abs(J)
    np.fill_diagonal(absJ, 0)

    # Baseline: vanilla |J|
    print("BASELINE (|J|):")
    baseline = {}
    for k in args.ks:
        r = cluster_and_score(absJ, k, gt_map, args.n_layer, args.n_head)
        if r is None:
            continue
        baseline[str(k)] = {"purity": r["purity"], "nmi": r["nmi"], "ari": r["ari"]}
        print(f"  k={k}: purity={r['purity']:.3f} nmi={r['nmi']:.3f} ari={r['ari']:.3f}")

    # Strategy C: block-mean residual
    print("\nSTRATEGY C: block-mean residual J:")
    resid = block_mean_residual(absJ, args.n_layer, args.n_head)
    strategy_c = {}
    for k in args.ks:
        r = cluster_and_score(resid, k, gt_map, args.n_layer, args.n_head)
        if r is None:
            continue
        strategy_c[str(k)] = {"purity": r["purity"], "nmi": r["nmi"], "ari": r["ari"],
                               "composition": r["composition"]}
        print(f"  k={k}: purity={r['purity']:.3f} nmi={r['nmi']:.3f} ari={r['ari']:.3f}")

    # Strategy D: within-layer clustering
    print("\nSTRATEGY D: within-layer Ising clusters:")
    strategy_d = {}
    for kpl in (2, 3, 4):
        labels = within_layer_cluster(J, args.n_layer, args.n_head, k_per_layer=kpl)
        # Score these labels directly via gt_map (no spectral clustering call)
        gt, pred = [], []
        for f in range(args.n_layer * args.n_head):
            L, H = divmod(f, args.n_head)
            if (L, H) in gt_map:
                gt.append(gt_map[(L, H)])
                pred.append(labels[f])
        if not gt:
            continue
        classes = sorted(set(gt))
        cls2int = {c: i for i, c in enumerate(classes)}
        gt_int = np.array([cls2int[g] for g in gt])
        pred_arr = np.array(pred)
        purity = 0
        for c in np.unique(pred_arr):
            idx = pred_arr == c
            members = np.array(gt)[idx]
            if len(members) == 0:
                continue
            unique, counts = np.unique(members, return_counts=True)
            purity += counts.max()
        purity /= len(gt)
        nmi = normalized_mutual_info_score(gt_int, pred_arr)
        ari = adjusted_rand_score(gt_int, pred_arr)
        strategy_d[f"kpl_{kpl}"] = {
            "k_per_layer": kpl,
            "total_clusters": int(np.unique(pred_arr).size),
            "purity": float(purity), "nmi": float(nmi), "ari": float(ari),
        }
        print(f"  k_per_layer={kpl}: total={int(np.unique(pred_arr).size)} "
              f"purity={purity:.3f} nmi={nmi:.3f} ari={ari:.3f}")

    out = {
        "tag": args.tag,
        "baseline_absJ": baseline,
        "strategy_c_block_residual": strategy_c,
        "strategy_d_within_layer": strategy_d,
    }
    out_path = results_dir / "layer_residualized.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
