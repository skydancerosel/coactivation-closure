"""
visualize_ising.py

Plot the J coupling matrix and community structure for one model.
Saves a 2x2 figure:
  - top-left: J reordered by layer (raw structure)
  - top-right: J reordered by spectral-cluster community (block view)
  - bottom-left: per-(layer, head) ground-truth class
  - bottom-right: per-(layer, head) cluster label

Usage:
  python visualize_ising.py --results-dir results/pythia_1b_ising \
      --mechinterp-json .../pythia_mechinterp.json \
      --n-layer 16 --n-head 8 --k 6 --tag pythia_1b
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_COLOR = {
    "first-token": "tab:blue",
    "previous-token": "tab:orange",
    "duplicate-token": "tab:green",
    "induction": "tab:red",
    "self": "tab:purple",
    "local": "tab:brown",
    "unclassified": "tab:gray",
    "other": "white",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--mechinterp-json", required=True)
    ap.add_argument("--n-layer", type=int, required=True)
    ap.add_argument("--n-head", type=int, required=True)
    ap.add_argument("--k", type=int, default=6, help="Cluster size for block view")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    J = np.load(results_dir / "J.npy")
    res = json.load(open(results_dir / "results.json"))
    mech = json.load(open(args.mechinterp_json))

    F = args.n_layer * args.n_head
    labels = np.array(res["cluster_results"][str(args.k)])

    # Ground truth per head
    gt_map = {(c["layer"], c["head"]): c["classification"]
              for c in mech.get("classifications", [])}
    gt_per_head = []
    for f in range(F):
        L, H = divmod(f, args.n_head)
        gt_per_head.append(gt_map.get((L, H), "other"))

    # Reorder by community for block view
    order = np.argsort(labels)
    J_blocked = J[np.ix_(order, order)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11),
                              gridspec_kw={"height_ratios": [3, 1.5]})

    vmax = np.percentile(np.abs(J), 99)

    # top-left: J in (layer, head) order
    ax = axes[0, 0]
    im = ax.imshow(J, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    # draw layer boundaries
    for l in range(1, args.n_layer):
        ax.axhline(l * args.n_head - 0.5, color="black", lw=0.3, alpha=0.3)
        ax.axvline(l * args.n_head - 0.5, color="black", lw=0.3, alpha=0.3)
    ax.set_title(f"Ising coupling J  (layer-major order)\n"
                 f"{args.tag}: F={F}, ||J||_F={np.linalg.norm(J):.2f}")
    ax.set_xlabel("head index (L * n_head + H)")
    ax.set_ylabel("head index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # top-right: J reordered by community
    ax = axes[0, 1]
    im = ax.imshow(J_blocked, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    # cluster boundaries
    boundaries = []
    last = labels[order[0]]
    for i, idx in enumerate(order):
        if labels[idx] != last:
            boundaries.append(i)
            last = labels[idx]
    for b in boundaries:
        ax.axhline(b - 0.5, color="black", lw=0.5)
        ax.axvline(b - 0.5, color="black", lw=0.5)
    ax.set_title(f"J reordered by community  (k={args.k})\n"
                 f"purity={res['metrics_by_k'][str(args.k)]['purity']:.3f} "
                 f"NMI={res['metrics_by_k'][str(args.k)]['nmi']:.3f} "
                 f"ARI={res['metrics_by_k'][str(args.k)]['ari']:.3f}")
    ax.set_xlabel("reordered head index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # bottom-left: ground truth per (L, H) — heatmap with class colors
    ax = axes[1, 0]
    gt_grid = np.array(gt_per_head).reshape(args.n_layer, args.n_head)
    classes_seen = sorted(set(gt_per_head))
    cls2color = {c: CLASS_COLOR.get(c, "white") for c in classes_seen}
    grid_rgb = np.zeros((args.n_layer, args.n_head, 4))
    for L in range(args.n_layer):
        for H in range(args.n_head):
            c = gt_grid[L, H]
            rgba = matplotlib.colors.to_rgba(cls2color[c])
            grid_rgb[L, H] = rgba
    ax.imshow(grid_rgb, aspect="auto")
    ax.set_xticks(range(args.n_head))
    ax.set_yticks(range(args.n_layer))
    ax.set_xlabel("head H")
    ax.set_ylabel("layer L")
    ax.set_title("Ground truth class (≥30× selectivity)")
    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=cls2color[c]) for c in classes_seen]
    ax.legend(handles, classes_seen, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8)

    # bottom-right: cluster label per (L, H)
    ax = axes[1, 1]
    cluster_grid = labels.reshape(args.n_layer, args.n_head)
    im = ax.imshow(cluster_grid, aspect="auto", cmap="tab20")
    ax.set_xticks(range(args.n_head))
    ax.set_yticks(range(args.n_layer))
    ax.set_xlabel("head H")
    ax.set_ylabel("layer L")
    ax.set_title(f"Spectral cluster label (k={args.k})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = results_dir / f"{args.tag}_ising_overview.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
