"""
per_class_emergence_plot.py

Plot per-class emergence trajectory: for each end-state-classified head
of class C, track its per-checkpoint selectivity-to-C, then aggregate.

Two views:
  (a) Mean selectivity across end-state-classified heads of class C vs
      step. One curve per class.
  (b) Max selectivity (sharpest head) across all heads of class C vs
      step. One curve per class.

Also: a per-head version — restricted to a few specific end-state-
classified heads — to see whether they emerge synchronously or some are
"leaders."
"""

import argparse, json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["first-token", "previous-token", "induction",
           "duplicate-token", "self", "local"]
COLORS = {
    "first-token": "#1f77b4",
    "previous-token": "#ff7f0e",
    "induction": "#2ca02c",
    "duplicate-token": "#d62728",
    "self": "#9467bd",
    "local": "#8c564b",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def parse_step(rev):
    m = re.search(r"step(\d+)", rev)
    return int(m.group(1)) if m else None


def load_records(results_dir):
    records = []
    for path in sorted(Path(results_dir).glob("*.json")):
        with open(path) as f:
            d = json.load(f)
        d["step"] = parse_step(d["revision"])
        records.append(d)
    records.sort(key=lambda r: r["step"])
    return records


def get_endstate_class_map(mech_json):
    """Return dict mapping (L, H) -> class for top-K classified heads."""
    mech = json.load(open(mech_json))
    return {(c["layer"], c["head"]): c["classification"]
            for c in mech.get("classifications", [])}


def plot_per_class_emergence(results_dir, mech_json, model_tag, out_dir):
    records = load_records(results_dir)
    class_map = get_endstate_class_map(mech_json)
    print(f"loaded {len(records)} checkpoints, {len(class_map)} end-state classified heads")

    # Group end-state heads by class
    heads_per_class = {c: [] for c in CLASSES}
    for (L, H), cls in class_map.items():
        if cls in heads_per_class:
            heads_per_class[cls].append((L, H))
    for c, hs in heads_per_class.items():
        print(f"  {c}: {len(hs)} heads")

    # Per checkpoint, per class: selectivity-to-canonical-target of
    # end-state-classified-as-C heads.
    steps = np.array([r["step"] for r in records])
    n_layer = records[0]["n_layer"]
    n_head = records[0]["n_head"]

    # mean and max selectivity-to-C across end-state-C heads
    mean_sel = {c: np.full(len(records), np.nan) for c in CLASSES}
    max_sel = {c: np.full(len(records), np.nan) for c in CLASSES}
    # also: selectivity-to-C across ALL heads (no class restriction)
    mean_sel_all = {c: np.full(len(records), np.nan) for c in CLASSES}

    for i, r in enumerate(records):
        sel = r["selectivity"]  # dict class -> (n_layer, n_head) list
        for c in CLASSES:
            sel_arr = np.array(sel[c])  # (n_layer, n_head)
            heads_in_c = heads_per_class[c]
            if heads_in_c:
                vals = np.array([sel_arr[L, H] for L, H in heads_in_c])
                mean_sel[c][i] = vals.mean()
                max_sel[c][i] = vals.max()
            mean_sel_all[c][i] = sel_arr.mean()

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # 1. Mean selectivity-to-C across end-state-C heads
    ax = axes[0, 0]
    for c in CLASSES:
        if heads_per_class[c]:
            ax.plot(steps, mean_sel[c], "o-", color=COLORS[c],
                    label=f"{c} (n={len(heads_per_class[c])})",
                    linewidth=1.2, markersize=4)
    ax.axhline(30, color="0.5", linestyle="--", linewidth=0.7, label="≥30× threshold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean selectivity (end-state heads)")
    ax.set_title(f"{model_tag}: per-class emergence (mean over end-state heads)", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    # 2. Max selectivity
    ax = axes[0, 1]
    for c in CLASSES:
        if heads_per_class[c]:
            ax.plot(steps, max_sel[c], "o-", color=COLORS[c],
                    label=f"{c} (n={len(heads_per_class[c])})",
                    linewidth=1.2, markersize=4)
    ax.axhline(30, color="0.5", linestyle="--", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Max selectivity (sharpest head)")
    ax.set_title(f"{model_tag}: max selectivity per class", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    # 3. Threshold-crossing time per head (within each class)
    # Find first step at which each end-state head crosses 30× selectivity
    ax = axes[1, 0]
    threshold = 30.0
    for c in CLASSES:
        if not heads_per_class[c]:
            continue
        cross_times = []
        for L, H in heads_per_class[c]:
            sel_per_step = np.array([
                np.array(r["selectivity"][c])[L, H] for r in records
            ])
            crossed = np.where(sel_per_step >= threshold)[0]
            if len(crossed) > 0:
                cross_times.append(steps[crossed[0]])
            else:
                cross_times.append(np.nan)
        # Plot as a horizontal bar: each head a point at its crossing step
        y = list(CLASSES).index(c)
        x_vals = [t for t in cross_times if not np.isnan(t)]
        if x_vals:
            ax.scatter(x_vals, [y] * len(x_vals),
                        color=COLORS[c], s=30, alpha=0.7,
                        label=f"{c} (n={len(x_vals)}/{len(cross_times)} crossed)")
    ax.set_xscale("log")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Training step at threshold crossing (≥30×)")
    ax.set_title(f"{model_tag}: per-head emergence time", fontsize=10)
    ax.grid(True, alpha=0.3, which="major")

    # 4. Mean selectivity across ALL heads (control: should be near 1 for non-target classes)
    ax = axes[1, 1]
    for c in CLASSES:
        ax.plot(steps, mean_sel_all[c], "o-", color=COLORS[c],
                label=c, linewidth=1.0, markersize=3, alpha=0.7)
    ax.axhline(1.0, color="0.5", linestyle=":", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean selectivity (all heads)")
    ax.set_title(f"{model_tag}: control — mean across all 128 heads", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out_path = Path(out_dir) / f"per_class_emergence_{model_tag}.png"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--mech-json", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    plot_per_class_emergence(args.results_dir, args.mech_json,
                               args.model_tag, args.out_dir)


if __name__ == "__main__":
    main()
