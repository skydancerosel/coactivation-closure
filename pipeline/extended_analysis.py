"""
extended_analysis.py

Run per-model extended analyses:
  - All-heads classification (not just top-K) using all_head_selectivity
  - Layer-locality of J coupling matrix
  - Per-cluster layer distribution
  - Null permutation z-scores at multiple k values

Usage:
  python extended_analysis.py --tag pythia_1b \
      --results-dir results/pythia_1b_ising \
      --mechinterp-json .../pythia_mechinterp.json \
      --n-layer 16 --n-head 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def classify_all_heads(all_head_selectivity, threshold=30.0):
    """Build per-(L, H) class assignment using all_head_selectivity dict.

    For each head, pick the class with maximum selectivity ≥ threshold.
    If all classes are below threshold, label "unspecialized".

    all_head_selectivity is keyed by 'L{l}_H{h}' with per-class dict.
    """
    out = {}
    for k, sel in all_head_selectivity.items():
        L = int(k.split("_")[0][1:])
        H = int(k.split("_")[1][1:])
        # find the class with highest selectivity, only among canonical
        # classes (skip 'baseline')
        cls_sels = {c: v for c, v in sel.items() if c != "baseline"}
        if not cls_sels:
            out[(L, H)] = "unspecialized"
            continue
        best = max(cls_sels.items(), key=lambda x: x[1])
        if best[1] >= threshold:
            out[(L, H)] = best[0]
        else:
            out[(L, H)] = "unspecialized"
    return out


def layer_locality(J, n_layer, n_head):
    """Measure how much J's mass concentrates within-layer vs across-layer.

    Returns: within_frac (sum |J| within same layer / total sum |J|),
    plus per-layer-distance distribution of |J|.
    """
    F = n_layer * n_head
    absJ = np.abs(J)
    np.fill_diagonal(absJ, 0)
    total = absJ.sum()

    within = 0.0
    by_dist = np.zeros(n_layer)
    for i in range(F):
        Li = i // n_head
        for j in range(i + 1, F):
            Lj = j // n_head
            d = abs(Li - Lj)
            by_dist[d] += 2 * absJ[i, j]  # symmetrize
            if d == 0:
                within += 2 * absJ[i, j]

    return {
        "within_layer_frac": float(within / total),
        "by_layer_distance": by_dist.tolist(),
        "total_abs_J": float(total),
    }


def cluster_layer_distribution(labels, n_layer, n_head):
    """For each cluster, what fraction of its members are in which layer?
    Returns dict cluster_id → per-layer count.
    """
    labels = np.array(labels)
    F = n_layer * n_head
    out = {}
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        per_layer = np.zeros(n_layer, dtype=int)
        for f in members:
            per_layer[f // n_head] += 1
        out[int(c)] = per_layer.tolist()
    return out


def all_heads_metrics(labels, all_class_map, n_layer, n_head, exclude="unspecialized"):
    """Compute purity/NMI/ARI against all-heads classification.

    Excludes 'unspecialized' heads from comparison.
    """
    F = n_layer * n_head
    gt, pred = [], []
    for f in range(F):
        L, H = divmod(f, n_head)
        g = all_class_map.get((L, H))
        if g is None or g == exclude:
            continue
        gt.append(g)
        pred.append(labels[f])
    if len(gt) == 0:
        return {"n_evaluated": 0}
    gt_arr = np.array(gt)
    pred_arr = np.array(pred)
    classes = sorted(set(gt))
    cls2int = {c: i for i, c in enumerate(classes)}
    gt_int = np.array([cls2int[g] for g in gt])

    # purity
    correct = 0
    composition = {}
    for c in np.unique(pred_arr):
        idx = pred_arr == c
        if idx.sum() == 0:
            continue
        members = gt_arr[idx]
        unique, counts = np.unique(members, return_counts=True)
        correct += counts.max()
        composition[int(c)] = {str(u): int(v) for u, v in zip(unique, counts)}

    nmi = float(normalized_mutual_info_score(gt_int, pred_arr))
    ari = float(adjusted_rand_score(gt_int, pred_arr))

    # null
    rng = np.random.RandomState(0)
    null_ari = []
    null_nmi = []
    for _ in range(200):
        perm = rng.permutation(pred_arr)
        null_ari.append(adjusted_rand_score(gt_int, perm))
        null_nmi.append(normalized_mutual_info_score(gt_int, perm))
    null_ari = np.array(null_ari)
    null_nmi = np.array(null_nmi)

    return {
        "n_evaluated": len(gt),
        "purity": float(correct / len(gt)),
        "nmi": nmi,
        "ari": ari,
        "null_ari_mean": float(null_ari.mean()),
        "null_ari_std": float(null_ari.std()),
        "null_nmi_mean": float(null_nmi.mean()),
        "null_nmi_std": float(null_nmi.std()),
        "ari_z": float((ari - null_ari.mean()) / max(null_ari.std(), 1e-9)),
        "nmi_z": float((nmi - null_nmi.mean()) / max(null_nmi.std(), 1e-9)),
        "cluster_composition": composition,
        "class_counts": {c: int((gt_int == i).sum()) for c, i in cls2int.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--mechinterp-json", required=True)
    ap.add_argument("--n-layer", type=int, required=True)
    ap.add_argument("--n-head", type=int, required=True)
    ap.add_argument("--threshold", type=float, default=30.0)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    J = np.load(results_dir / "J.npy")
    res = json.load(open(results_dir / "results.json"))
    mech = json.load(open(args.mechinterp_json))

    all_sel = mech.get("all_head_selectivity", {})
    all_class_map = classify_all_heads(all_sel, threshold=args.threshold)

    # report class counts
    print(f"\n=== {args.tag} ===")
    print("All-heads classification (≥{}× selectivity):".format(args.threshold))
    from collections import Counter
    counts = Counter(all_class_map.values())
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {n}")

    # for each k, compute extended metrics
    print("\nAll-heads metrics across k:")
    extended = {}
    for k in res["cluster_results"]:
        labels = res["cluster_results"][k]
        if labels is None:
            continue
        m = all_heads_metrics(labels, all_class_map, args.n_layer, args.n_head)
        extended[k] = m
        print(f"  k={k}: n={m.get('n_evaluated', 0)} "
              f"purity={m.get('purity', 0):.3f} "
              f"NMI={m.get('nmi', 0):.3f}(z={m.get('nmi_z', 0):.1f}) "
              f"ARI={m.get('ari', 0):.3f}(z={m.get('ari_z', 0):.1f})")

    # layer locality
    print("\nLayer locality of J:")
    loc = layer_locality(J, args.n_layer, args.n_head)
    print(f"  within-layer fraction of |J|: {loc['within_layer_frac']:.3f}")
    print(f"  random expectation (8/256 same-layer pairs for 16x16): "
          f"{(args.n_head - 1) / (args.n_layer * args.n_head - 1):.3f}")
    # show first few layer-distance bins
    by_d = loc["by_layer_distance"]
    print(f"  |J| by layer distance (d=0..5): "
          f"{[f'{x:.2f}' for x in by_d[:6]]}")

    # save
    out = {
        "tag": args.tag,
        "all_heads_class_counts": dict(counts),
        "extended_metrics_by_k": extended,
        "layer_locality": loc,
    }
    out_path = results_dir / "extended_analysis.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
