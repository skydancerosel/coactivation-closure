"""
visualize_pure_cluster.py

For OLMo 1B cluster 2 (100%-pure self-attention community of 5 L0 heads),
visualize the coupling structure: how strongly do these 5 heads couple to
each other vs. to the rest of the model?

Uses any of {ising_J, mutual_info, pearson_spin} from
baseline_comparison.json reproduction.

Usage:
  python visualize_pure_cluster.py --tag olmo_1b
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="olmo_1b")
    ap.add_argument("--results-dir", default="results/olmo_1b_ising")
    ap.add_argument("--n-layer", type=int, default=16)
    ap.add_argument("--n-head", type=int, default=16)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    rd = Path(args.results_dir)
    J = np.load(rd / "J.npy")
    res = json.load(open(rd / "results.json"))
    labels = np.array(res["cluster_results"][str(args.k)])

    # Find pure-self cluster: pick the cluster that contains most L0 heads
    F = args.n_layer * args.n_head
    n_head = args.n_head

    # Identify cluster 2 from the OLMo k=10 result (5 self heads)
    # Heuristic: pick cluster with smallest size > 3 that has all members
    # within a single layer.
    cluster_sizes = {}
    cluster_layer_span = {}
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        layers = set(m // n_head for m in members)
        cluster_sizes[int(c)] = len(members)
        cluster_layer_span[int(c)] = (len(layers), sorted(layers))

    print(f"Cluster sizes and layer span:")
    for c, sz in sorted(cluster_sizes.items()):
        span = cluster_layer_span[c]
        members = np.where(labels == c)[0]
        head_ids = [(m // n_head, m % n_head) for m in members]
        print(f"  c{c}: size={sz}, layers={span[1]}, members={head_ids}")

    # pick the cluster with the smallest layer span (most layer-local)
    # ties broken by smallest size > 1
    best_cluster = min(
        [(c, len(cluster_layer_span[c][1]), cluster_sizes[c]) for c in cluster_sizes],
        key=lambda x: (x[1], x[2])
    )[0]
    members = np.where(labels == best_cluster)[0]
    print(f"\nMost layer-local cluster: c{best_cluster} (n={len(members)})")
    print(f"  members: {[(m // n_head, m % n_head) for m in members]}")

    # Build a figure showing:
    #  (1) full J with the cluster members highlighted
    #  (2) zoomed J on cluster members (within-cluster couplings)
    #  (3) mean |J| from each cluster member to all heads (showing isolation)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (1)
    ax = axes[0]
    vmax = np.percentile(np.abs(J), 99)
    im = ax.imshow(J, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for m in members:
        ax.axhline(m, color="lime", lw=0.4, alpha=0.7)
        ax.axvline(m, color="lime", lw=0.4, alpha=0.7)
    ax.set_title(f"{args.tag} Ising J — cluster {best_cluster} members highlighted")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (2)
    ax = axes[1]
    J_sub = J[np.ix_(members, members)]
    im = ax.imshow(J_sub, cmap="RdBu_r",
                   vmin=-np.abs(J_sub).max(), vmax=np.abs(J_sub).max())
    head_labels = [f"L{m//n_head}H{m%n_head}" for m in members]
    ax.set_xticks(range(len(members)))
    ax.set_yticks(range(len(members)))
    ax.set_xticklabels(head_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(head_labels, fontsize=8)
    ax.set_title(f"Within-cluster coupling sub-matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (3) For each cluster member, mean |J| within cluster vs outside
    ax = axes[2]
    within_means = []
    outside_means = []
    for m in members:
        mask = np.zeros(F, dtype=bool)
        mask[members] = True
        mask[m] = False  # exclude self
        within = np.abs(J[m, mask & np.isin(np.arange(F), members)]).mean()
        outside_mask = ~np.isin(np.arange(F), members)
        outside = np.abs(J[m, outside_mask]).mean()
        within_means.append(within)
        outside_means.append(outside)

    x = np.arange(len(members))
    w = 0.4
    ax.bar(x - w/2, within_means, w, label="within cluster", color="tab:blue")
    ax.bar(x + w/2, outside_means, w, label="outside cluster", color="tab:gray")
    ax.set_xticks(x)
    ax.set_xticklabels(head_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("mean |J|")
    ax.set_title(f"Per-member: within vs outside coupling")
    ax.legend()
    avg_ratio = np.mean(within_means) / max(np.mean(outside_means), 1e-9)
    ax.text(0.05, 0.95, f"avg within/outside = {avg_ratio:.2f}×",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8))

    plt.tight_layout()
    out = rd / f"{args.tag}_pure_cluster.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
