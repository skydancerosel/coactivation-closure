"""
supervised_vs_unsupervised.py

Head-to-head comparison: does the unsupervised co-activation pipeline
recover the same heads that the supervised 3-step methodology (PR-integral
+ ≥30× selectivity + ablation) identifies?

For each model:
  1. Take the SUPERVISED head-sets per capability class (from
     all_head_selectivity ≥ 30× threshold).
  2. For each unsupervised cluster, score precision/recall against each
     class. Identify the best-matched cluster for each class.
  3. Compute the per-class "recovery" metric: fraction of supervised
     class members captured by their best-matched cluster.
  4. Special focus: L0-L1 BOS heads (the universal finding across 5 models
     from the methodology paper) — how completely does the unsupervised
     method find them?
  5. Confusion-matrix visualization.

Usage:
  python supervised_vs_unsupervised.py --tag pythia_1b ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def classify_all_heads(all_head_selectivity, threshold=30.0):
    out = {}
    for k, sel in all_head_selectivity.items():
        L = int(k.split("_")[0][1:])
        H = int(k.split("_")[1][1:])
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


def confusion_matrix(class_map, labels, n_layer, n_head):
    """Returns (n_classes x n_clusters) confusion matrix.

    class_map: dict (L, H) -> class name
    labels: array of length F = n_layer * n_head, cluster assignment
    """
    F = n_layer * n_head
    classes = sorted(set(class_map.values()) - {"unspecialized"})
    clusters = sorted(set(int(l) for l in labels))
    cm = np.zeros((len(classes), len(clusters)), dtype=int)
    cls2i = {c: i for i, c in enumerate(classes)}
    clu2j = {c: j for j, c in enumerate(clusters)}
    for f in range(F):
        L, H = divmod(f, n_head)
        cls = class_map.get((L, H), "unspecialized")
        if cls == "unspecialized":
            continue
        cm[cls2i[cls], clu2j[int(labels[f])]] += 1
    return cm, classes, clusters


def per_class_recall_precision(cm, classes, clusters):
    """For each class, find the cluster with the most class members.
    Compute recall (frac of class in best cluster) and precision
    (frac of best cluster matching class)."""
    out = {}
    for i, c in enumerate(classes):
        row = cm[i, :]
        if row.sum() == 0:
            out[c] = {"recall": 0.0, "precision": 0.0,
                      "best_cluster": None, "class_total": 0}
            continue
        j_best = int(np.argmax(row))
        recall = float(row[j_best] / row.sum())
        col_sum = cm[:, j_best].sum()
        precision = float(row[j_best] / col_sum) if col_sum > 0 else 0.0
        out[c] = {
            "recall": recall,
            "precision": precision,
            "f1": 2 * recall * precision / max(recall + precision, 1e-9),
            "best_cluster": int(clusters[j_best]),
            "class_total": int(row.sum()),
            "cluster_total": int(col_sum),
        }
    return out


def bos_l01_focus(class_map, labels, n_layer, n_head):
    """Special focus: L0-L1 first-token heads (the universal finding).
    How distributed are they across clusters?"""
    bos_l01 = [(L, H) for (L, H), cls in class_map.items()
               if cls == "first-token" and L in (0, 1)]
    if not bos_l01:
        return None
    cluster_assignment = []
    for L, H in bos_l01:
        f = L * n_head + H
        cluster_assignment.append(int(labels[f]))
    unique, counts = np.unique(cluster_assignment, return_counts=True)
    top_cluster = int(unique[np.argmax(counts)])
    top_recall = float(counts.max() / len(bos_l01))
    return {
        "n_bos_l01": len(bos_l01),
        "heads": [(L, H) for L, H in bos_l01],
        "cluster_assignment": cluster_assignment,
        "distribution": {str(int(u)): int(c) for u, c in zip(unique, counts)},
        "top_cluster": top_cluster,
        "top_cluster_recall_of_bos_l01": top_recall,
        "n_clusters_holding_bos_l01": int(len(unique)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--mechinterp-json", required=True)
    ap.add_argument("--n-layer", type=int, required=True)
    ap.add_argument("--n-head", type=int, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[6, 8, 10, 12])
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    res = json.load(open(results_dir / "results.json"))
    mech = json.load(open(args.mechinterp_json))

    print(f"\n=== {args.tag} supervised vs unsupervised ===\n")
    class_map = classify_all_heads(mech.get("all_head_selectivity", {}),
                                     threshold=30.0)
    # report supervised class sizes
    from collections import Counter
    sup_counts = Counter(class_map.values())
    print("Supervised class sizes (≥30× selectivity):")
    for c, n in sorted(sup_counts.items(), key=lambda x: -x[1]):
        if c == "unspecialized":
            continue
        print(f"  {c}: {n}")

    summary = {"tag": args.tag, "supervised_counts": dict(sup_counts),
               "by_k": {}}
    for k in args.ks:
        labels = np.array(res["cluster_results"][str(k)])
        cm, classes, clusters = confusion_matrix(class_map, labels,
                                                   args.n_layer, args.n_head)
        per_class = per_class_recall_precision(cm, classes, clusters)
        bos_focus = bos_l01_focus(class_map, labels, args.n_layer, args.n_head)
        summary["by_k"][str(k)] = {
            "per_class": per_class,
            "bos_l01_focus": bos_focus,
        }

        print(f"\nk={k}:")
        for c, m in per_class.items():
            print(f"  {c:18s}: recall={m['recall']:.2f} prec={m['precision']:.2f} "
                  f"f1={m['f1']:.2f} (class n={m['class_total']}, "
                  f"best cluster {m['best_cluster']} n={m['cluster_total']})")
        if bos_focus:
            print(f"  L0-L1 BOS focus: {bos_focus['n_bos_l01']} heads, "
                  f"top cluster {bos_focus['top_cluster']} captures "
                  f"{bos_focus['top_cluster_recall_of_bos_l01']:.1%}, "
                  f"distributed over {bos_focus['n_clusters_holding_bos_l01']} clusters")

    # Visualize confusion matrix at the best k (most informative for L0-L1 BOS)
    best_k = max(args.ks)
    labels = np.array(res["cluster_results"][str(best_k)])
    cm, classes, clusters = confusion_matrix(class_map, labels,
                                              args.n_layer, args.n_head)

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(clusters)),
                                      max(3, 0.6 * len(classes))))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([f"c{c}" for c in clusters])
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(clusters)):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                        fontsize=9)
    ax.set_xlabel("unsupervised cluster")
    ax.set_ylabel("supervised class")
    ax.set_title(f"{args.tag}: confusion (supervised classes × unsupervised clusters), k={best_k}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_png = results_dir / f"{args.tag}_supervised_vs_unsupervised_k{best_k}.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out_png}")

    out_json = results_dir / "supervised_vs_unsupervised.json"
    json.dump(summary, open(out_json, "w"), indent=2)
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
