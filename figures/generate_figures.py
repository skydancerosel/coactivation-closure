"""
Generate the three figures for the tiny paper:
  fig1_verdict.pdf       — 5-test verdict, 3 metrics × 5 tests
  fig2_redundancy.pdf    — Pythia 1B Δloss vs Δlogit divergence
  fig3_moe_story.pdf     — MoE 3-panel story arc

All figures use matplotlib with PDF backend, no rasterization, sized for
single-column LaTeX inclusion.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,    # TrueType, no Type-3 warnings
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# Colors (consistent across figures)
COL_PASS = "#2a7a3a"        # green
COL_WEAK = "#c98a1a"        # orange
COL_FAIL = "#b03030"        # red
COL_CTRL = "#888888"        # gray for control distributions
COL_CTRL_LIGHT = "#cccccc"

# ──────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────────────────────────────────────


def load_synthetic(path, cand_key, ctrl_key):
    """Synthetic-batch closure_test.json (OLMo, Pythia synthetic).

    Returns: baseline metric dict, candidate metric dict, control metric list.
    """
    d = json.load(open(path))
    baseline = d["baseline"]
    candidate = d[cand_key]
    controls = d[ctrl_key]
    # synthetic uses "mean_logit_B"; natural uses "mean_logit_target"
    # normalize the key
    for x in [baseline, candidate] + controls:
        if "mean_logit_B" in x and "mean_logit_target" not in x:
            x["mean_logit_target"] = x["mean_logit_B"]
    return baseline, candidate, controls


def load_natural(path):
    """Natural-batch closure_test.json (OLMo nat, Pythia nat)."""
    d = json.load(open(path))
    return d["baseline"], d["candidate_ablation"], d["controls"]


def load_moe(path):
    """OLMoE route-conditioned closure. Returns split-by-cluster1 metrics."""
    d = json.load(open(path))
    return d


def compute_z(cand_delta, control_deltas):
    """Z-score of candidate Δ vs distribution of control Δ."""
    arr = np.array(control_deltas)
    if arr.std() < 1e-9:
        return 0.0
    return (cand_delta - arr.mean()) / arr.std()


def compute_test_metrics(baseline, candidate, controls):
    """Return:
        dloss_cand, dacc_cand, dlogit_cand,
        z_loss, z_acc, z_logit,
        ctrl_dlosses, ctrl_daccs, ctrl_dlogits.
    """
    dloss_c = candidate["loss"] - baseline["loss"]
    dacc_c = candidate["acc_top1"] - baseline["acc_top1"]
    dlogit_c = candidate["mean_logit_target"] - baseline["mean_logit_target"]
    ctrl_dl = [c["loss"] - baseline["loss"] for c in controls]
    ctrl_da = [c["acc_top1"] - baseline["acc_top1"] for c in controls]
    ctrl_dz = [c["mean_logit_target"] - baseline["mean_logit_target"] for c in controls]
    return {
        "dloss": dloss_c, "dacc": dacc_c, "dlogit": dlogit_c,
        "z_loss": compute_z(dloss_c, ctrl_dl),
        "z_acc": compute_z(dacc_c, ctrl_da),
        "z_logit": compute_z(dlogit_c, ctrl_dz),
        "ctrl_dl": ctrl_dl, "ctrl_da": ctrl_da, "ctrl_dz": ctrl_dz,
        "baseline_loss": baseline["loss"],
    }


def load_all_tests():
    """Returns a dict of test_name -> metric dict (cluster-1 view for MoE)."""
    out = {}

    # 1. OLMo synthetic
    b, c, ctrls = load_synthetic(
        "results/olmo_1b_ising/closure_test.json",
        "ablate_cluster2", "matched_random_controls")
    out["OLMo synth"] = compute_test_metrics(b, c, ctrls)

    # 2. Pythia synthetic
    b, c, ctrls = load_synthetic(
        "results/pythia_1b_ising/closure_test.json",
        "ablate_cluster4", "matched_size_controls")
    out["Pythia synth"] = compute_test_metrics(b, c, ctrls)

    # 3. OLMo natural
    b, c, ctrls = load_natural(
        "results/olmo_1b_nat_ising/closure_test.json")
    out["OLMo nat"] = compute_test_metrics(b, c, ctrls)

    # 4. Pythia natural
    b, c, ctrls = load_natural(
        "results/pythia_1b_nat_ising/closure_test.json")
    out["Pythia nat"] = compute_test_metrics(b, c, ctrls)

    # 5. OLMoE route-conditional (on cluster 1 subset)
    moe = load_moe("results/olmoe_route_conditioned/closure_test.json")
    b1 = moe["baseline"]["cluster1"]
    c1 = moe["candidate_ablation"]["cluster1"]
    ctrls1 = [c["cluster1"] for c in moe["controls"]]
    out["OLMoE nat (route, c1)"] = compute_test_metrics(b1, c1, ctrls1)
    # also keep the non-cluster-1 view for figure 3
    bn = moe["baseline"]["non_cluster1"]
    cn = moe["candidate_ablation"]["non_cluster1"]
    ctrlsn = [c["non_cluster1"] for c in moe["controls"]]
    out["__OLMoE_nc1"] = compute_test_metrics(bn, cn, ctrlsn)

    return out


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 1: Five-test verdict (3 metrics × 5 tests, with control overlay)
# ──────────────────────────────────────────────────────────────────────────


def figure_1_verdict(tests):
    """Three-panel bar chart of Δloss, Δacc, Δlogit across 5 tests.

    For each (test, metric), show the candidate's value as a colored bar,
    and the control distribution as a thin gray bar with ±1 std error bar."""
    test_names = ["OLMo synth", "Pythia synth", "OLMo nat",
                  "Pythia nat", "OLMoE nat (route, c1)"]
    short_labels = ["OLMo\nsynth\n(5h)", "Pythia\nsynth\n(25h)",
                    "OLMo\nnat\n(10h)", "Pythia\nnat\n(9h)",
                    "OLMoE\nnat\n(22h)"]
    verdicts = ["pass", "weak pass", "pass", "pass", "FAIL"]
    colors = [COL_PASS, COL_WEAK, COL_PASS, COL_PASS, COL_FAIL]

    metric_names = [
        ("dloss", "ctrl_dl", r"$\Delta$loss (nats)",
         "↑ damage", True),
        ("dacc", "ctrl_da", r"$\Delta$accuracy",
         "↓ damage", False),
        ("dlogit", "ctrl_dz", r"$\Delta$target logit",
         "↓ damage", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    x = np.arange(len(test_names))
    bar_w = 0.32

    for j, (mk, ck, ylabel, direction_text, damage_pos) in enumerate(metric_names):
        ax = axes[j]
        cand_vals = [tests[t][mk] for t in test_names]
        ctrl_means = [np.mean(tests[t][ck]) for t in test_names]
        ctrl_stds = [np.std(tests[t][ck]) for t in test_names]
        z_vals = [tests[t]["z_" + ("loss" if mk == "dloss" else
                                    "acc" if mk == "dacc" else "logit")]
                  for t in test_names]

        # Control bars (gray, narrower, with error bars showing ±1σ)
        ax.bar(x - bar_w / 2, ctrl_means, bar_w, color=COL_CTRL_LIGHT,
               edgecolor=COL_CTRL, linewidth=0.5, label="controls (5)",
               yerr=ctrl_stds, ecolor=COL_CTRL, capsize=2)

        # Candidate bars (colored by verdict)
        bars = ax.bar(x + bar_w / 2, cand_vals, bar_w, color=colors,
                       edgecolor="black", linewidth=0.5, label="candidate")

        # z-score annotation above each candidate bar
        for xi, cv, z in zip(x, cand_vals, z_vals):
            sign = "+" if z >= 0 else ""
            va = "bottom" if cv >= 0 else "top"
            offset = max(abs(cv), 0.05) * 0.15
            ax.text(xi + bar_w / 2, cv + (offset if cv >= 0 else -offset),
                    f"{sign}{z:.1f}σ", ha="center", va=va, fontsize=6.5,
                    color="black")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}  ({direction_text})", fontsize=8.5)

        # Add a small "→ pass" arrow indicating damage direction
        ymin, ymax = ax.get_ylim()
        if damage_pos:
            ax.text(0.99, 0.97, "pass: bar above 0", ha="right", va="top",
                    transform=ax.transAxes, fontsize=6.5, color=COL_PASS,
                    style="italic")
        else:
            ax.text(0.99, 0.03, "pass: bar below 0", ha="right", va="bottom",
                    transform=ax.transAxes, fontsize=6.5, color=COL_PASS,
                    style="italic")

    fig.suptitle("Five-test closure verdict across three metrics",
                 fontsize=10, y=1.005)
    fig.tight_layout()
    out = FIG_DIR / "fig1_verdict.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 2: Pythia 1B redundancy signature
# ──────────────────────────────────────────────────────────────────────────


def figure_2_redundancy(tests):
    """Two-panel: per-test Δloss (left) vs Δtarget-logit (right).

    Pythia natural reads as a near-zero bar on the left and a large
    negative bar on the right — the 25× metric divergence is visual."""
    test_names = ["OLMo synth", "Pythia synth", "OLMo nat",
                  "Pythia nat", "OLMoE nat (route, c1)"]
    short_labels = ["OLMo\nsynth", "Pythia\nsynth", "OLMo\nnat",
                    "Pythia\nnat", "OLMoE\nroute"]
    colors = [COL_PASS, COL_WEAK, COL_PASS, COL_PASS, COL_FAIL]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    x = np.arange(len(test_names))

    # Left panel: Δloss
    ax = axes[0]
    dlosses = [tests[t]["dloss"] for t in test_names]
    ax.bar(x, dlosses, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=7.5)
    ax.set_ylabel(r"$\Delta$loss (nats)")
    ax.set_title(r"$\Delta$loss: aggregate cross-entropy", fontsize=9)
    # Highlight Pythia nat with annotation
    pn_idx = 3
    ax.annotate(f"{dlosses[pn_idx]:+.2f}",
                xy=(pn_idx, dlosses[pn_idx]),
                xytext=(pn_idx, max(dlosses) * 0.5),
                fontsize=7, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="gray"))

    # Right panel: Δtarget-logit
    ax = axes[1]
    dlogits = [tests[t]["dlogit"] for t in test_names]
    ax.bar(x, dlogits, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=7.5)
    ax.set_ylabel(r"$\Delta$ target-token logit")
    ax.set_title(r"$\Delta$logit: per-target prediction confidence",
                 fontsize=9)
    ax.annotate(f"{dlogits[pn_idx]:+.2f}",
                xy=(pn_idx, dlogits[pn_idx]),
                xytext=(pn_idx + 0.4, min(dlogits) * 0.5),
                fontsize=7, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="gray"))

    fig.suptitle("Pythia 1B natural-text divergence: " r"$\Delta$loss"
                 " barely moves while " r"$\Delta$logit" " collapses",
                 fontsize=9.5, y=1.04)
    fig.tight_layout()
    out = FIG_DIR / "fig2_redundancy.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 3: MoE story arc (3 panels)
# ──────────────────────────────────────────────────────────────────────────


def figure_3_moe(tests):
    """Three panels:
        (a) Natural-text marginal Ising ARI across 3 models
        (b) OLMoE route-stratified per-cluster ARI vs null
        (c) OLMoE closure Δloss on cluster 1 vs non-cluster 1 (candidate vs ctrl)
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.3))

    # ── panel a: natural-text marginal ARI by model ──
    ax = axes[0]
    models = ["Pythia 1B", "OLMo 1B", "OLMoE\n1B-7B"]
    ari_vals = [0.350, 0.199, 0.006]
    colors_a = [COL_PASS, COL_PASS, COL_FAIL]
    bars = ax.bar(np.arange(3), ari_vals, color=colors_a, edgecolor="black",
                   linewidth=0.5)
    for i, v in enumerate(ari_vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(models, fontsize=7.5)
    ax.set_ylabel("Best ARI (natural text)")
    ax.set_title("(a) Marginal Ising: MoE collapses on natural text",
                 fontsize=8.5)
    ax.set_ylim(0, 0.42)

    # ── panel b: route-stratified per-cluster ARI vs null ──
    ax = axes[1]
    # Read route-stratified results from results.json
    moe_results = json.load(open("results/olmoe_route_conditioned/results.json"))
    # strategy_A_stratified["4"]["per_cluster"][c]["best"]["ari"]
    per_cluster = moe_results["strategy_A_stratified"]["4"]["per_cluster"]
    cluster_aris = []
    for c_record in per_cluster:
        if "best" in c_record and c_record["best"]:
            cluster_aris.append(c_record["best"]["ari"])
        else:
            cluster_aris.append(0.0)

    # Random-partition null statistics
    null_data = json.load(open(
        "results/olmoe_route_conditioned/random_partition_null.json"))
    null_mean = null_data["null_max_ari_mean"]
    null_std = null_data["null_max_ari_std"]

    cluster_x = np.arange(len(cluster_aris))
    bar_colors = [COL_PASS if v > null_mean + null_std else COL_CTRL
                   for v in cluster_aris]
    ax.bar(cluster_x, cluster_aris, color=bar_colors, edgecolor="black",
           linewidth=0.5)
    # Null band: mean ± 1 std
    ax.axhspan(null_mean - null_std, null_mean + null_std,
               color=COL_CTRL_LIGHT, alpha=0.5, zorder=0,
               label=f"random-partition null\nmean ± 1σ")
    ax.axhline(null_mean, color=COL_CTRL, linestyle="--", linewidth=0.7,
               zorder=0)
    ax.set_xticks(cluster_x)
    ax.set_xticklabels([f"c{c}\n(n={per_cluster[c]['n_examples']})"
                         for c in cluster_x], fontsize=7)
    ax.set_ylabel("Within-stratum best ARI")
    ax.set_title("(b) Route-stratified: cluster 1 above null (+3σ)",
                 fontsize=8.5)
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    # Mark cluster 1
    ax.text(1, cluster_aris[1] + 0.012, f"{cluster_aris[1]:.3f}",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # ── panel c: closure direction (wrong) ──
    ax = axes[2]
    moe_data = json.load(open(
        "results/olmoe_route_conditioned/closure_test.json"))

    # Candidate Δloss on c1 and non-c1
    base_c1 = moe_data["baseline"]["cluster1"]["loss"]
    base_nc1 = moe_data["baseline"]["non_cluster1"]["loss"]
    cand_c1 = moe_data["candidate_ablation"]["cluster1"]["loss"] - base_c1
    cand_nc1 = moe_data["candidate_ablation"]["non_cluster1"]["loss"] - base_nc1
    ctrl_c1 = [c["cluster1"]["loss"] - base_c1 for c in moe_data["controls"]]
    ctrl_nc1 = [c["non_cluster1"]["loss"] - base_nc1 for c in moe_data["controls"]]

    x = np.arange(2)
    bar_w = 0.32
    ax.bar(x - bar_w / 2, [np.mean(ctrl_c1), np.mean(ctrl_nc1)], bar_w,
           color=COL_CTRL_LIGHT, edgecolor=COL_CTRL, linewidth=0.5,
           yerr=[np.std(ctrl_c1), np.std(ctrl_nc1)], ecolor=COL_CTRL,
           capsize=2, label="controls (5)")
    ax.bar(x + bar_w / 2, [cand_c1, cand_nc1], bar_w,
           color=COL_FAIL, edgecolor="black", linewidth=0.5,
           label="candidate")
    ax.axhline(0, color="black", linewidth=0.5)

    # z-score annotations
    z_c1 = (cand_c1 - np.mean(ctrl_c1)) / np.std(ctrl_c1)
    z_nc1 = (cand_nc1 - np.mean(ctrl_nc1)) / np.std(ctrl_nc1)
    ax.text(0 + bar_w / 2, cand_c1 - 0.1,
            f"{z_c1:+.1f}σ\nwrong dir.", ha="center", va="top", fontsize=6.5,
            color=COL_FAIL)
    ax.text(1 + bar_w / 2, cand_nc1 - 0.1,
            f"{z_nc1:+.1f}σ\nwrong dir.", ha="center", va="top", fontsize=6.5,
            color=COL_FAIL)

    ax.set_xticks(x)
    ax.set_xticklabels(["cluster 1\n(n=637)", "non-cluster 1\n(n=1363)"],
                       fontsize=7.5)
    ax.set_ylabel(r"$\Delta$loss (nats)")
    ax.set_title("(c) Closure: ablation helps loss (wrong dir.)",
                 fontsize=8.5)
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    # Add "pass would be ABOVE 0" annotation
    ax.text(0.5, 0.95, r"pass: bars above 0", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.5, color=COL_PASS,
            style="italic")

    fig.suptitle("MoE story: marginal collapse, route-conditional statistical "
                 "recovery, closure failure",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    out = FIG_DIR / "fig3_moe_story.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ──────────────────────────────────────────────────────────────────────────


def main():
    print("loading closure data...")
    tests = load_all_tests()
    for k, v in tests.items():
        if k.startswith("__"):
            continue
        print(f"  {k}: Δloss={v['dloss']:+.3f} (z={v['z_loss']:+.2f}σ)  "
              f"Δacc={v['dacc']:+.3f} (z={v['z_acc']:+.2f}σ)  "
              f"Δlogit={v['dlogit']:+.3f} (z={v['z_logit']:+.2f}σ)")

    print("\ngenerating figures...")
    figure_1_verdict(tests)
    figure_2_redundancy(tests)
    figure_3_moe(tests)
    print("\ndone.")


if __name__ == "__main__":
    main()
