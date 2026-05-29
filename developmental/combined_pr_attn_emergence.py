"""
combined_pr_attn_emergence.py

Combine three developmental signals for each end-state-classified head:
  - PR of per-head attention output across examples (functional signal,
    from spectral-probe-circuits phase1_trajectory.json)
  - Attention-to-canonical-target selectivity (behavioral signal, from
    our dev_per_class JSONs)
  - End-state class assignment (from mechinterp_naturaltext.json)

For each (model, class), plot per-head trajectories of PR and attention
selectivity on the same x-axis (training tokens). Identifies whether
function-emergence (PR) leads, lags, or matches pattern-emergence
(attention-to-target).

Output: figures/pr_attn_combined_{model}_{class}.png

Also produces a cross-architecture summary: time-to-PR-threshold vs
time-to-attention-threshold per head.
"""

import json, re
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
})

PHASE_DIR = Path("/Volumes/Brandy/mini_gpt/.claude/worktrees/nostalgic-lederberg-80a58d")

MODELS = [
    ("pythia_1b",  "Pythia 1B",
     PHASE_DIR / "pythia_phase1_trajectory.json",
     PHASE_DIR / "pythia_mechinterp_naturaltext.json",
     "results/dev_per_class/pythia_1b",
     2_000_000),  # tokens/step (Pythia uses tokens count we compute)
    ("olmo_1b", "OLMo 1B",
     PHASE_DIR / "olmo_phase1_trajectory.json",
     PHASE_DIR / "olmo_mechinterp_naturaltext.json",
     "results/dev_per_class/olmo_1b",
     None),
    ("olmoe_1b_7b", "OLMoE 1B-7B",
     PHASE_DIR / "olmoe_phase1_trajectory.json",
     PHASE_DIR / "olmoe_mechinterp_naturaltext.json",
     "results/dev_per_class/olmoe_1b_7b",
     None),
]

CLASSES = ["first-token", "previous-token", "self"]
COLORS = {
    "first-token": "#1f77b4",
    "previous-token": "#ff7f0e",
    "self": "#9467bd",
}

PR_THRESHOLD = 5.0    # PR > 5 = head is doing substantial content-dependent work
ATTN_THRESHOLD = 30.0  # 30× selectivity, same as supervised


def parse_step(rev):
    m = re.search(r"step(\d+)", rev)
    return int(m.group(1)) if m else None


def parse_tokens(rev, default_tps=None):
    m = re.search(r"tokens(\d+)([KMBT])", rev)
    if m:
        return int(m.group(1)) * {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[m.group(2)]
    if default_tps is not None:
        step = parse_step(rev)
        return step * default_tps if step else None
    return None


def load_attn_data(per_class_dir, default_tps):
    """Return dict ckpt_step -> per-class selectivity arrays (n_layer, n_head)."""
    records = []
    for path in sorted(Path(per_class_dir).glob("*.json")):
        d = json.load(open(path))
        d["step"] = parse_step(d["revision"])
        d["tokens"] = parse_tokens(d["revision"], default_tps)
        records.append(d)
    records.sort(key=lambda r: r["step"])
    return records


def main():
    Path("figures").mkdir(exist_ok=True)
    summary_data = {}  # for cross-arch summary

    for tag, name, phase_path, mech_path, per_class_dir, default_tps in MODELS:
        print(f"\n=== {tag} ===")
        # Load PR
        phase = json.load(open(phase_path))
        pr_data = phase["pr"]  # dict L{L}_H{H} -> [PR per ckpt]
        pr_steps = phase["ckpt_step"]
        pr_tokens = np.array(phase["ckpt_tokens_B"]) * 1e9
        n_layer = phase["n_layer"]
        n_head = phase["num_heads"]
        print(f"  PR: {n_layer}L × {n_head}H, {len(pr_steps)} ckpts: {pr_steps[:5]}...")

        # Load attn-to-target per ckpt
        attn_records = load_attn_data(per_class_dir, default_tps)
        attn_steps = [r["step"] for r in attn_records]
        attn_tokens = np.array([r["tokens"] for r in attn_records])
        print(f"  attn-to-target: {len(attn_records)} ckpts: {attn_steps[:5]}...")

        # Load end-state classification
        mech = json.load(open(mech_path))
        cls_map = {(c["layer"], c["head"]): c["classification"]
                   for c in mech.get("classifications", [])}
        heads_per_class = {}
        for (L, H), cls in cls_map.items():
            heads_per_class.setdefault(cls, []).append((L, H))

        # For each end-state class with ≥3 heads, plot combined
        n_panels = sum(1 for c in CLASSES if len(heads_per_class.get(c, [])) >= 3)
        if n_panels == 0:
            print("  no class with ≥3 heads — skipping")
            continue
        fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3.0 * n_panels),
                                  sharex=True)
        if n_panels == 1:
            axes = [axes]

        panel = 0
        for cls in CLASSES:
            heads = heads_per_class.get(cls, [])
            if len(heads) < 3:
                continue
            ax = axes[panel]
            panel += 1

            # Aggregate PR over class heads
            pr_per_head = np.array([pr_data[f"L{L}_H{H}"] for L, H in heads])
            pr_mean = pr_per_head.mean(axis=0)
            pr_max = pr_per_head.max(axis=0)

            # Aggregate attn-to-target over class heads
            attn_per_head = np.array([
                [np.array(r["selectivity"][cls])[L, H] for L, H in heads]
                for r in attn_records
            ])  # (n_ckpts, n_heads)
            attn_mean = attn_per_head.mean(axis=1)
            attn_max = attn_per_head.max(axis=1)

            color = COLORS[cls]

            # Twin axis for two metrics
            ax2 = ax.twinx()
            line_pr_mean = ax.plot(pr_tokens, pr_mean, "o-", color=color,
                                     markersize=4, linewidth=1.5,
                                     label="PR (mean)")
            line_pr_max = ax.plot(pr_tokens, pr_max, "o--", color=color,
                                    markersize=3, linewidth=0.8, alpha=0.5,
                                    label="PR (max)")
            line_attn_mean = ax2.plot(attn_tokens, attn_mean, "s-",
                                        color=color, markersize=4,
                                        linewidth=1.5, alpha=0.7,
                                        label="attn-sel (mean)")
            ax.axhline(PR_THRESHOLD, color=color, linestyle=":", linewidth=0.5,
                        alpha=0.5)
            ax2.axhline(ATTN_THRESHOLD, color=color, linestyle="--",
                         linewidth=0.5, alpha=0.5)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax2.set_yscale("log")
            ax.set_ylabel(f"PR ({cls})", color=color)
            ax2.set_ylabel(f"attn-selectivity ({cls})", color=color)
            ax.tick_params(axis="y", labelcolor=color)
            ax2.tick_params(axis="y", labelcolor=color)
            ax.set_title(f"{cls}: {len(heads)} end-state heads", fontsize=9)
            ax.grid(True, alpha=0.3, which="both")

            # Collect summary
            # First ckpt where mean crosses threshold
            pr_cross = np.where(pr_mean >= PR_THRESHOLD)[0]
            pr_cross_token = pr_tokens[pr_cross[0]] if len(pr_cross) else None
            attn_cross = np.where(attn_mean >= ATTN_THRESHOLD)[0]
            attn_cross_token = attn_tokens[attn_cross[0]] if len(attn_cross) else None
            summary_data.setdefault(tag, {})[cls] = {
                "pr_cross_token": float(pr_cross_token) if pr_cross_token is not None else None,
                "attn_cross_token": float(attn_cross_token) if attn_cross_token is not None else None,
                "n_heads": len(heads),
                "pr_final": float(pr_mean[-1]),
                "attn_final": float(attn_mean[-1]),
            }

        axes[-1].set_xlabel("Training tokens")
        fig.suptitle(f"{name}: PR (left axis) vs attention-selectivity (right axis), "
                     "per end-state class",
                     fontsize=10, y=1.005)
        fig.tight_layout()
        out = Path("figures") / f"pr_attn_combined_{tag}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  saved {out}")

    # Cross-architecture summary
    print("\n\n=== CROSS-ARCHITECTURE PR-vs-ATTN EMERGENCE SUMMARY ===")
    print(f"{'arch':>12} {'class':>15} {'n':>4} "
          f"{'PR-cross':>14} {'attn-cross':>14} {'PR-final':>10} {'attn-final':>10}")
    for tag in summary_data:
        for cls, info in summary_data[tag].items():
            pr_t = f"{info['pr_cross_token']/1e9:.1f}B" if info['pr_cross_token'] else "—"
            at_t = f"{info['attn_cross_token']/1e9:.1f}B" if info['attn_cross_token'] else "—"
            print(f"{tag:>12} {cls:>15} {info['n_heads']:>4} {pr_t:>14} {at_t:>14}"
                  f" {info['pr_final']:>10.1f} {info['attn_final']:>10.1f}")

    json.dump(summary_data, open("figures/pr_attn_summary.json", "w"), indent=2)
    print(f"\nsaved figures/pr_attn_summary.json")


if __name__ == "__main__":
    main()
