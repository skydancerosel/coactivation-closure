"""
closure_olmoe_route_cluster1.py

Step 2 of the OLMoE route-conditional closure test. The candidate from
step 1 (sub-cluster 2 of cluster 1's k_spectral=4 sub-clustering) is a
22-head set spanning 8 layers with isolation ratio 2.74x.

Closure logic:
- Ablate the 22 candidate heads. Measure damage on:
  (a) cluster 1 inputs (n=637, where the community was discovered)
  (b) non-cluster-1 inputs (n=1363, where the route stratum doesn't apply)
  (c) 5 matched-random-head controls (22 random heads each, same protocol)
- Tests:
  T1: candidate Δloss on (a) > candidate Δloss on (b) → route-specific
  T2: candidate Δloss on (a) > random-control Δloss on (a) → head-specific

If both, the route-conditional community is causally specific AND the
specificity is route-conditional, not head-set-conditional.

Eval metrics computed on the natural-text batch:
- CE loss at query position (per-example, averaged within the relevant
  subset)
- top-1 accuracy
- mean logit at the target token

Output: results/olmoe_route_conditioned/closure_test.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def make_pre_hook(heads_in_layer, head_dim):
    lo_his = [(h * head_dim, (h + 1) * head_dim) for h in heads_in_layer]
    def pre_hook(_module, ainputs):
        x = ainputs[0].clone()
        for lo, hi in lo_his:
            x[..., lo:hi] = 0
        return (x,) + ainputs[1:]
    return pre_hook


def install_ablation(model, ablate_dict, head_dim):
    """ablate_dict: dict layer_idx -> list of head indices to zero."""
    handles = []
    for layer_idx, heads in ablate_dict.items():
        if not heads:
            continue
        h = model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
            make_pre_hook(heads, head_dim))
        handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def evaluate(model, tokens, query_pos, targets, device, batch_size=4):
    """Returns per-example loss, top-1 hit, target-token logit at query_pos.

    Note: model is fed tokens[:, :query_pos[i]+1] effectively, but since we
    just forward the full sequence and read logits at query_pos[i], that's
    what we get (causal LM).
    """
    n = tokens.shape[0]
    per_ex_loss = np.zeros(n, dtype=np.float32)
    per_ex_top1 = np.zeros(n, dtype=np.float32)
    per_ex_logit = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tok = tokens[start:end].to(device)
            logits = model(tok).logits.float()
            for j in range(end - start):
                qp = int(query_pos[start + j].item())
                tgt = int(targets[start + j].item())
                logit_at_qp = logits[j, qp, :]
                per_ex_loss[start + j] = F.cross_entropy(
                    logit_at_qp.unsqueeze(0),
                    torch.tensor([tgt], device=device)
                ).item()
                per_ex_top1[start + j] = float(int(logit_at_qp.argmax().item()) == tgt)
                per_ex_logit[start + j] = float(logit_at_qp[tgt].item())
            del logits
            if device == "mps":
                torch.mps.empty_cache()
            if start > 0 and start % (batch_size * 25) == 0:
                rate = end / (time.time() - t0)
                eta = (n - end) / rate
                print(f"    {end}/{n}  ({rate:.1f} ex/s, ETA {eta:.0f}s)",
                      flush=True)
    print(f"  eval done in {time.time()-t0:.0f}s")
    return per_ex_loss, per_ex_top1, per_ex_logit


def split_metrics(per_ex_loss, per_ex_top1, per_ex_logit, mask):
    sub_loss = per_ex_loss[mask]; sub_top1 = per_ex_top1[mask]; sub_logit = per_ex_logit[mask]
    return {
        "n": int(mask.sum()),
        "loss": float(sub_loss.mean()),
        "loss_std": float(sub_loss.std()),
        "acc_top1": float(sub_top1.mean()),
        "mean_logit_target": float(sub_logit.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--batch-file", default="/Volumes/Brandy/mini_gpt/.claude/worktrees/nostalgic-lederberg-80a58d/natural_induction_batch.pt")
    ap.add_argument("--results-dir", default="results/olmoe_route_conditioned")
    ap.add_argument("--candidate-json", default="results/olmoe_route_conditioned/closure_candidate.json")
    ap.add_argument("--cluster-assignments", default="results/olmoe_route_conditioned/route_cluster_assignments.json")
    ap.add_argument("--n-controls", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    rd = Path(args.results_dir)

    # Load candidate heads
    candidate = json.load(open(args.candidate_json))["candidate"]
    cand_heads = [tuple(h) for h in candidate["heads"]]
    print(f"\ncandidate: sub-cluster {candidate['sub_cluster']}, "
          f"size={candidate['size']} heads, layers {candidate['layers']}")
    # Group by layer
    by_layer = {}
    for L, H in cand_heads:
        by_layer.setdefault(int(L), []).append(int(H))

    # Load route-cluster-1 mask
    cluster_assigns = json.load(open(args.cluster_assignments))
    labels_K4 = np.array(cluster_assigns["4"])
    cluster1_mask = labels_K4 == 1
    n_cluster1 = int(cluster1_mask.sum())
    n_non_cluster1 = int((~cluster1_mask).sum())
    print(f"cluster 1: n={n_cluster1}, non-cluster-1: n={n_non_cluster1}")

    # Load natural-text batch
    print(f"\nloading natural-text batch from {args.batch_file}...")
    batch = torch.load(args.batch_file, weights_only=False)
    tokens = batch["tokens"]; query_pos = batch["query_pos"]; targets = batch["targets"]
    print(f"  tokens={tuple(tokens.shape)}")

    # Load model
    print(f"\nloading {args.model}...")
    t0 = time.time()
    from transformers import OlmoeForCausalLM
    model = OlmoeForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float16,
        attn_implementation="eager")
    model = model.to(device).eval()
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_head
    print(f"  loaded in {time.time()-t0:.0f}s "
          f"({n_layer}L × {n_head}H, head_dim={head_dim})")

    n_total_heads = n_layer * n_head

    results = {}

    # --- 1. baseline (no ablation) ---
    print("\n=== CONDITION: baseline (no ablation) ===")
    loss, top1, logit = evaluate(model, tokens, query_pos, targets, device,
                                   batch_size=args.batch_size)
    results["baseline"] = {
        "cluster1": split_metrics(loss, top1, logit, cluster1_mask),
        "non_cluster1": split_metrics(loss, top1, logit, ~cluster1_mask),
        "overall": split_metrics(loss, top1, logit, np.ones_like(cluster1_mask)),
    }
    print(f"  cluster1 loss: {results['baseline']['cluster1']['loss']:.4f}")
    print(f"  non_cluster1 loss: {results['baseline']['non_cluster1']['loss']:.4f}")

    # --- 2. ablate candidate heads ---
    print(f"\n=== CONDITION: ablate candidate (22 heads) ===")
    handles = install_ablation(model, by_layer, head_dim)
    loss, top1, logit = evaluate(model, tokens, query_pos, targets, device,
                                   batch_size=args.batch_size)
    remove_hooks(handles)
    results["candidate_ablation"] = {
        "cluster1": split_metrics(loss, top1, logit, cluster1_mask),
        "non_cluster1": split_metrics(loss, top1, logit, ~cluster1_mask),
        "overall": split_metrics(loss, top1, logit, np.ones_like(cluster1_mask)),
        "ablated_heads": cand_heads,
        "by_layer": {str(k): v for k, v in by_layer.items()},
    }
    print(f"  cluster1 loss: {results['candidate_ablation']['cluster1']['loss']:.4f} "
          f"(Δ={results['candidate_ablation']['cluster1']['loss'] - results['baseline']['cluster1']['loss']:+.4f})")
    print(f"  non_cluster1 loss: {results['candidate_ablation']['non_cluster1']['loss']:.4f} "
          f"(Δ={results['candidate_ablation']['non_cluster1']['loss'] - results['baseline']['non_cluster1']['loss']:+.4f})")

    # --- 3. matched random controls ---
    print(f"\n=== CONDITION: {args.n_controls} matched-random controls (22 random heads each) ===")
    rng = np.random.RandomState(0)
    control_results = []
    cand_set = set((L, H) for L, H in cand_heads)
    for seed_idx in range(args.n_controls):
        # sample 22 random heads from the 256, excluding overlap with candidate
        all_heads = [(L, H) for L in range(n_layer) for H in range(n_head)]
        available = [hd for hd in all_heads if hd not in cand_set]
        idxs = rng.choice(len(available), size=len(cand_heads), replace=False)
        ctrl_heads = [available[i] for i in idxs]
        ctrl_by_layer = {}
        for L, H in ctrl_heads:
            ctrl_by_layer.setdefault(L, []).append(H)
        handles = install_ablation(model, ctrl_by_layer, head_dim)
        loss, top1, logit = evaluate(model, tokens, query_pos, targets, device,
                                       batch_size=args.batch_size)
        remove_hooks(handles)
        rec = {
            "seed_idx": seed_idx,
            "heads": ctrl_heads,
            "cluster1": split_metrics(loss, top1, logit, cluster1_mask),
            "non_cluster1": split_metrics(loss, top1, logit, ~cluster1_mask),
            "overall": split_metrics(loss, top1, logit, np.ones_like(cluster1_mask)),
        }
        control_results.append(rec)
        d_c1 = rec['cluster1']['loss'] - results['baseline']['cluster1']['loss']
        d_nc1 = rec['non_cluster1']['loss'] - results['baseline']['non_cluster1']['loss']
        print(f"  seed {seed_idx}: cluster1 Δloss {d_c1:+.4f}, "
              f"non_cluster1 Δloss {d_nc1:+.4f}")
    results["controls"] = control_results

    # --- summary ---
    b_c1 = results['baseline']['cluster1']['loss']
    b_nc1 = results['baseline']['non_cluster1']['loss']
    cand_dl_c1 = results['candidate_ablation']['cluster1']['loss'] - b_c1
    cand_dl_nc1 = results['candidate_ablation']['non_cluster1']['loss'] - b_nc1
    ctrl_dl_c1 = np.array([c['cluster1']['loss'] - b_c1 for c in control_results])
    ctrl_dl_nc1 = np.array([c['non_cluster1']['loss'] - b_nc1 for c in control_results])

    z_cand_vs_ctrl_c1 = (cand_dl_c1 - ctrl_dl_c1.mean()) / max(ctrl_dl_c1.std(), 1e-9)
    z_cand_vs_ctrl_nc1 = (cand_dl_nc1 - ctrl_dl_nc1.mean()) / max(ctrl_dl_nc1.std(), 1e-9)

    print("\n=== closure summary ===")
    print(f"  baseline loss cluster1={b_c1:.4f} non_cluster1={b_nc1:.4f}")
    print(f"  candidate Δloss   cluster1={cand_dl_c1:+.4f}  non_cluster1={cand_dl_nc1:+.4f}")
    print(f"    cluster1 - non_cluster1 (route-specificity): {cand_dl_c1 - cand_dl_nc1:+.4f}")
    print(f"  control Δloss     cluster1={ctrl_dl_c1.mean():+.4f}±{ctrl_dl_c1.std():.4f}  "
          f"non_cluster1={ctrl_dl_nc1.mean():+.4f}±{ctrl_dl_nc1.std():.4f}")
    print(f"  z(candidate vs controls): cluster1={z_cand_vs_ctrl_c1:+.2f}σ  "
          f"non_cluster1={z_cand_vs_ctrl_nc1:+.2f}σ")

    results["summary"] = {
        "baseline_loss_cluster1": b_c1,
        "baseline_loss_non_cluster1": b_nc1,
        "candidate_dloss_cluster1": cand_dl_c1,
        "candidate_dloss_non_cluster1": cand_dl_nc1,
        "candidate_route_specificity": cand_dl_c1 - cand_dl_nc1,
        "control_dloss_cluster1_mean": float(ctrl_dl_c1.mean()),
        "control_dloss_cluster1_std": float(ctrl_dl_c1.std()),
        "control_dloss_non_cluster1_mean": float(ctrl_dl_nc1.mean()),
        "control_dloss_non_cluster1_std": float(ctrl_dl_nc1.std()),
        "z_candidate_vs_controls_cluster1": float(z_cand_vs_ctrl_c1),
        "z_candidate_vs_controls_non_cluster1": float(z_cand_vs_ctrl_nc1),
    }

    out_path = rd / "closure_test.json"
    json.dump(results, open(out_path, "w"), indent=2, default=float)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
