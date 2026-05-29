"""
dev_analysis_checks.py

Three methodological checks on the Pythia developmental scan, before
committing to a developmental story:

  (1) Fixed-k survival of the step-512 peak: plot ARI at k=4 (the early
      best-k) alongside best-ARI and NMI@k=6. If peak survives at fixed
      k=4 it's not a "best-k jumping" artifact.

  (2) Real selection criterion for best-k drift: argmax of top-eigengap
      of |J| at each checkpoint (using saved eig_info.top_gaps).
      Compare to argmax-ARI best-k. If they agree on the drift 4→8,
      drift is robust; if they disagree, argmax-ARI was wandering on
      a flat curve.

  (3) Heatmap decay shape: cross-checkpoint NMI as a function of
      step-distance in log space. Exponential decay → continuous drift.
      Step-shaped decay → two-regime transition.

Output: figures/dev_pythia_1b_checks.{pdf,png}
"""

import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def parse_step(rev):
    m = re.search(r"step(\d+)", rev)
    return int(m.group(1)) if m else None


def load(results_dir):
    records = []
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name.endswith("_routes.json"): continue
        with open(path) as f:
            d = json.load(f)
        records.append(d)
    records.sort(key=lambda r: parse_step(r["revision"]))
    return records


def main():
    RES = Path("results/developmental/pythia_1b")
    OUT = Path("figures/dev_pythia_1b_checks.png")

    records = load(RES)
    steps = np.array([parse_step(r["revision"]) for r in records])
    print(f"loaded {len(records)} checkpoints")

    # ──── (1) Fixed-k=4 column survival ────────────────────────────────
    print("\n=== check 1: ARI at fixed k=4 vs best-ARI vs NMI@k=6 ===")
    ari_best = np.array([max((m["ari"] for m in r["metrics_by_k"].values()), default=np.nan)
                          for r in records])
    ari_k4 = np.array([r["metrics_by_k"].get("4", {}).get("ari", np.nan) for r in records])
    nmi_k6 = np.array([r["metrics_by_k"].get("6", {}).get("nmi", np.nan) for r in records])

    print(f"{'step':>8} {'ARI@k=4':>10} {'best-ARI':>10} {'NMI@k=6':>10}")
    for s, a4, ab, n6 in zip(steps, ari_k4, ari_best, nmi_k6):
        print(f"{s:>8} {a4:>10.3f} {ab:>10.3f} {n6:>10.3f}")
    peak_idx = int(np.argmax(ari_best))
    peak_k4_idx = int(np.argmax(ari_k4))
    print(f"\n  best-ARI peak: step {steps[peak_idx]} (ARI={ari_best[peak_idx]:.3f})")
    print(f"  ARI@k=4 peak:  step {steps[peak_k4_idx]} (ARI={ari_k4[peak_k4_idx]:.3f})")
    if peak_idx == peak_k4_idx:
        print("  → peaks AGREE: the step-512 peak survives at fixed k=4.")
    else:
        print("  → peaks DISAGREE: the headline peak is partly a best-k artifact.")

    # ──── (2) Eigengap selection vs argmax-ARI ─────────────────────────
    print("\n=== check 2: eigengap-argmax best-k vs argmax-ARI best-k ===")
    # eig_info.top_gaps[i] is gap between eigenvalue_(i+1) and eigenvalue_(i+2)
    # so the "k that maximizes the gap" is the index i where top_gaps is largest
    eig_argmax_k = []
    eig_gaps_at_argmax_ari = []
    ari_argmax_k = []
    for r in records:
        gaps = r["eig_info"].get("top_gaps", [])
        if gaps:
            argmax_i = int(np.argmax(gaps))  # 0-indexed; k = argmax + 1
            eig_argmax_k.append(argmax_i + 1)
        else:
            eig_argmax_k.append(None)
        # argmax over ARI
        best_k = None; best_a = -np.inf
        for k_str, m in r["metrics_by_k"].items():
            if m["ari"] > best_a:
                best_a = m["ari"]; best_k = int(k_str)
        ari_argmax_k.append(best_k)

    print(f"{'step':>8} {'best-k (ARI)':>14} {'best-k (eigengap)':>20}")
    for s, ak, ek in zip(steps, ari_argmax_k, eig_argmax_k):
        ek_str = str(ek) if ek is not None else "—"
        print(f"{s:>8} {ak:>14} {ek_str:>20}")

    # ──── (3) Heatmap decay shape ──────────────────────────────────────
    print("\n=== check 3: NMI decay as function of step-distance ===")
    # Build NMI matrix at fixed k=6 (where we have labels)
    has_k6 = [r for r in records if "labels" in r["metrics_by_k"].get("6", {})]
    n = len(has_k6)
    if n < 3:
        print("  not enough labels to compute NMI matrix")
        return
    nmi_mat = np.zeros((n, n))
    step_arr = np.array([parse_step(r["revision"]) for r in has_k6])
    for i in range(n):
        for j in range(n):
            li = has_k6[i]["metrics_by_k"]["6"]["labels"]
            lj = has_k6[j]["metrics_by_k"]["6"]["labels"]
            nmi_mat[i, j] = normalized_mutual_info_score(li, lj)

    # NMI vs |log step_i - log step_j|
    log_steps = np.log10(np.maximum(step_arr, 1))
    distances = []
    nmis = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(abs(log_steps[i] - log_steps[j]))
            nmis.append(nmi_mat[i, j])
    distances = np.array(distances); nmis = np.array(nmis)

    # Bin by distance and compute mean NMI
    bins = np.linspace(0, distances.max() + 0.01, 8)
    bin_means = np.full(len(bins) - 1, np.nan)
    bin_stds = np.full(len(bins) - 1, np.nan)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    for b in range(len(bins) - 1):
        mask = (distances >= bins[b]) & (distances < bins[b + 1])
        if mask.sum() > 0:
            bin_means[b] = nmis[mask].mean()
            bin_stds[b] = nmis[mask].std()

    print(f"{'log-step-dist bin':>20} {'mean NMI':>10} {'std':>8} {'n_pairs':>8}")
    for b, c, m, s in zip(range(len(bins) - 1), bin_centers, bin_means, bin_stds):
        if not np.isnan(m):
            n_pairs = int(((distances >= bins[b]) & (distances < bins[b + 1])).sum())
            print(f"  [{bins[b]:.2f}, {bins[b+1]:.2f})    {m:>8.3f} {s:>8.3f} {n_pairs:>8}")

    # Plot all three checks
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.0))

    # (1) ARI@k=4 vs best-ARI vs NMI@k=6
    ax = axes[0]
    ax.plot(steps, ari_k4, "o-", label="ARI @ k=4 (fixed)", color="C1", markersize=4)
    ax.plot(steps, ari_best, "s-", label="best ARI (argmax over k)", color="black", markersize=4)
    ax.plot(steps, nmi_k6, "^-", label="NMI @ k=6 (fixed)", color="C2", markersize=4, alpha=0.6)
    ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("Training step (log)")
    ax.set_ylabel("Alignment vs end-state labels")
    ax.set_title("Check 1: peak survival under fixed k=4", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)

    # (2) argmax-ARI best-k vs eigengap-argmax best-k
    ax = axes[1]
    ax.plot(steps, ari_argmax_k, "o-", label="argmax-ARI best-k", color="C3", markersize=5)
    eig_k_plot = [e if e is not None else np.nan for e in eig_argmax_k]
    ax.plot(steps, eig_k_plot, "s--", label="argmax-eigengap best-k",
            color="C0", markersize=5)
    ax.set_xscale("log")
    ax.set_xlabel("Training step (log)")
    ax.set_ylabel("best-k (selection criterion)")
    ax.set_title("Check 2: best-k drift under two selection criteria", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 16)

    # (3) NMI decay vs step-distance
    ax = axes[2]
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt="o-",
                 color="C4", markersize=5, capsize=3, label="binned mean ± std")
    ax.scatter(distances, nmis, alpha=0.3, s=15, color="C4", label="all pairs")
    ax.axhline(1.0, color="0.5", linewidth=0.5, linestyle=":")
    ax.set_xlabel(r"log10 step-distance $|\log_{10}\text{step}_i - \log_{10}\text{step}_j|$")
    ax.set_ylabel(r"NMI(cluster$_i$, cluster$_j$) at k=6")
    ax.set_title("Check 3: NMI decay shape (drift vs regime)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.05)

    fig.suptitle("Pythia 1B developmental scan: methodological checks",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"\nsaved {OUT} (and .pdf)")


if __name__ == "__main__":
    main()
