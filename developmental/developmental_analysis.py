"""
developmental_analysis.py

Aggregate per-checkpoint JSONs from a developmental scan into trajectory
plots. Produces:
  - {out_dir}/{model}_trajectory.{pdf,png}: best-ARI vs step, ||J||_F
    vs step, per-k ARI breakdown, top eigengap vs step.
  - {out_dir}/{model}_per_k_ari.{pdf,png}: ARI at each fixed k value
    across training (to test whether best-k jumping causes volatility).
  - {out_dir}/{model}_cross_ckpt_nmi.{pdf,png}: cross-ckpt cluster
    stability NMI heatmap at a fixed k.

Usage:
  python developmental_analysis.py \
      --results-dir results/developmental/pythia_1b \
      --model-tag pythia_1b \
      --step-rank step1,step4,...,step143000 \
      --out-dir figures
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def parse_step(revision):
    """Extract integer step from a revision name like 'step38000' or
    'step25000-tokens52B'."""
    m = re.search(r"step(\d+)", revision)
    if not m:
        return None
    return int(m.group(1))


def parse_tokens(revision):
    """Extract token count from a revision name like 'step38000-tokens52B'
    or return None for 'step38000'."""
    m = re.search(r"tokens(\d+)([KMBT])", revision)
    if not m:
        return None
    n = int(m.group(1))
    suffix = m.group(2)
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[suffix]
    return int(n * mult)


def load_trajectories(results_dir):
    """Load all per-ckpt JSON files, return sorted by step."""
    results_dir = Path(results_dir)
    records = []
    for path in results_dir.glob("*.json"):
        if path.name.endswith("_routes.json"): continue
        with open(path) as f:
            d = json.load(f)
        rev = d["revision"]
        step = parse_step(rev)
        if step is None:
            print(f"  skipping {rev} — no step in name")
            continue
        tokens = parse_tokens(rev)
        records.append({
            "revision": rev,
            "step": step,
            "tokens": tokens,
            "ising_norm": d["ising_norm"],
            "metrics_by_k": d.get("metrics_by_k", {}),
            "eig_info": d.get("eig_info", {}),
            "routes_summary": d.get("routes_summary", None),
            "n_layer": d["n_layer"],
            "n_head": d["n_head"],
        })
    records.sort(key=lambda r: r["step"])
    return records


def trajectory_plot(records, out_path, model_tag, has_routes=False):
    """Main trajectory plot: best-ARI, ||J||, per-k ARI, top eigengap."""
    steps = np.array([r["step"] for r in records])
    norms = np.array([r["ising_norm"] for r in records])

    # ARI at each k
    ks = sorted({int(k) for r in records for k in r["metrics_by_k"].keys()})
    ari_by_k = {}
    for k in ks:
        ari_by_k[k] = np.array([r["metrics_by_k"].get(str(k), {}).get("ari", np.nan)
                                for r in records])
    best_ari = np.array([max((m["ari"] for m in r["metrics_by_k"].values()),
                              default=np.nan)
                          for r in records])

    # eigengap (first 5)
    eigs = []
    for r in records:
        g = r["eig_info"].get("top_gaps", [])
        eigs.append(g[:5] if len(g) >= 5 else g + [np.nan] * (5 - len(g)))
    eigs = np.array(eigs)

    n_panels = 4 if not has_routes else 5
    fig, axes = plt.subplots(n_panels, 1, figsize=(7.5, 1.9 * n_panels),
                              sharex=True)

    # 1. best ARI + per-k
    ax = axes[0]
    for k in ks:
        ax.plot(steps, ari_by_k[k], "o-", markersize=3, linewidth=0.8,
                alpha=0.5, label=f"k={k}")
    ax.plot(steps, best_ari, "s-", color="black", markersize=4, linewidth=1.2,
            label="best k", zorder=10)
    ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":")
    ax.set_ylabel("ARI vs end-state labels")
    ax.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.9)
    ax.set_title(f"{model_tag}: developmental scan", fontsize=10)
    ax.set_xscale("log")

    # 2. ||J||_F
    ax = axes[1]
    ax.plot(steps, norms, "o-", color="C2", markersize=4, linewidth=1)
    ax.set_ylabel(r"$\|J\|_F$ (Ising norm)")
    ax.set_xscale("log")

    # 3. eigengaps (top 3-5)
    ax = axes[2]
    for i in range(min(eigs.shape[1], 5)):
        ax.plot(steps, eigs[:, i], "o-", markersize=3, linewidth=0.8,
                alpha=0.6, label=f"gap{i+1}")
    ax.set_ylabel("Top eigenvalue gaps")
    ax.legend(loc="upper left", fontsize=7, ncol=5)
    ax.set_xscale("log")

    # 4. best-k indicator (which k won at each ckpt)
    ax = axes[3]
    best_k_vals = []
    for r in records:
        best_k = None; best_a = -np.inf
        for k, m in r["metrics_by_k"].items():
            if m["ari"] > best_a:
                best_a = m["ari"]; best_k = int(k)
        best_k_vals.append(best_k or 0)
    ax.plot(steps, best_k_vals, "o-", color="C3", markersize=4, linewidth=1)
    ax.set_ylabel("argmax-k (best)")
    ax.set_xscale("log")

    # 5. routing entropy (OLMoE only)
    if has_routes:
        ax = axes[4]
        ents = [r["routes_summary"]["fraction_of_max"]
                 if r["routes_summary"] else np.nan
                 for r in records]
        ax.plot(steps, ents, "o-", color="C4", markersize=4, linewidth=1)
        ax.set_ylabel(r"$H_{\text{route}} / H_{\max}$")
        ax.axhline(1.0, color="0.5", linewidth=0.5, linestyle=":")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)

    axes[-1].set_xlabel("Training step (log scale)")

    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {out_path} (and .pdf)")


def cross_ckpt_nmi_heatmap(records, out_path, model_tag, k_fixed=6):
    """NMI heatmap of cluster assignments across all checkpoint pairs
    at a fixed k."""
    from sklearn.metrics import normalized_mutual_info_score

    valid = [r for r in records
             if str(k_fixed) in r["metrics_by_k"] and
                "labels" in r["metrics_by_k"][str(k_fixed)]]
    if len(valid) < 2:
        print(f"  not enough labels at k={k_fixed} for cross-ckpt NMI")
        return

    n = len(valid)
    nmi_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            li = valid[i]["metrics_by_k"][str(k_fixed)]["labels"]
            lj = valid[j]["metrics_by_k"][str(k_fixed)]["labels"]
            nmi_mat[i, j] = normalized_mutual_info_score(li, lj)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(nmi_mat, cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([v["revision"].replace("step", "s") for v in valid],
                        rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([v["revision"].replace("step", "s") for v in valid],
                        fontsize=7)
    ax.set_title(f"{model_tag}: cross-ckpt cluster NMI (k={k_fixed})",
                 fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7, label="NMI")
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {out_path}")


def summary_table(records, model_tag):
    print(f"\n=== {model_tag} trajectory summary ===")
    print(f"{'step':>8} {'tokens':>12} {'||J||':>8} {'best-k':>7} {'ARI':>8} {'NMI@k=6':>9}")
    for r in records:
        tk = r["tokens"]
        tk_str = f"{tk/1e9:.1f}B" if tk is not None else "—"
        best_k = None; best_a = -np.inf; best_nmi = None
        for k, m in r["metrics_by_k"].items():
            if m["ari"] > best_a:
                best_a = m["ari"]; best_k = int(k)
        nmi6 = r["metrics_by_k"].get("6", {}).get("nmi", None)
        nmi6_str = f"{nmi6:.3f}" if nmi6 is not None else "—"
        print(f"{r['step']:>8} {tk_str:>12} {r['ising_norm']:>8.2f} "
              f"{best_k or '—':>7} {best_a:>8.3f} {nmi6_str:>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--k-fixed-nmi", type=int, default=6,
                    help="k value to use for cross-ckpt NMI heatmap")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    records = load_trajectories(args.results_dir)
    print(f"loaded {len(records)} checkpoints")
    if not records:
        print("no records to analyze")
        return

    has_routes = any(r["routes_summary"] for r in records)
    summary_table(records, args.model_tag)
    trajectory_plot(records, out_dir / f"dev_{args.model_tag}_trajectory.png",
                     args.model_tag, has_routes=has_routes)
    cross_ckpt_nmi_heatmap(records,
                             out_dir / f"dev_{args.model_tag}_cross_nmi.png",
                             args.model_tag, k_fixed=args.k_fixed_nmi)


if __name__ == "__main__":
    main()
