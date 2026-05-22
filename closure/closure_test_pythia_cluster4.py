"""
closure_test_pythia_cluster4.py

Replication of the OLMo cluster-2 closure test on a different model and
a different (less geometrically pure) cluster.

Pythia 1B Ising cluster 4 (k=6, n=25 heads):
  - top-K composition: 10 prev-token + 6 unclassified + 2 self
  - spans layers: 0, 1, 2, 3, 6, 8, 9 (per-layer count: 7/4/4/5/1/2/2)
  - essentially the "early-and-mid layers" community

This cluster is less geometrically isolated than OLMo cluster 2 — it
mixes layers and capability classes. Does the closure result generalize?

Conditions:
  1. baseline (no ablation)
  2. ablate cluster 4 (25 heads)
  3. ablate ALL heads in cluster 4's layers (upper bound, 56 heads)
  4. 5 matched-size random controls (25 random heads from the same layer
     set, sampled uniformly within those layers, may overlap cluster 4)

Pythia 1B hook target: model.gpt_neox.layers[L].attention.dense (the o_proj
analog in NeoX), zero per-head input slice.

Metrics: LM CE loss at last position, top-1/5 accuracy, mean logit_B.

Output: results/pythia_1b_ising/closure_test.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def build_induction_batch(n_examples=2000, seq_len=256, vocab_lo=100,
                          vocab_hi=10000, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    tokens = np.zeros((n_examples, seq_len), dtype=np.int64)
    targets = np.zeros(n_examples, dtype=np.int64)
    for i in range(n_examples):
        seq = rng.randint(vocab_lo, vocab_hi, size=seq_len).astype(np.int64)
        a, b = rng.choice(np.arange(vocab_lo, vocab_hi), size=2, replace=False)
        ab_idx = rng.randint(20, seq_len // 2)
        seq[ab_idx] = a
        seq[ab_idx + 1] = b
        for k in range(seq_len):
            if seq[k] == a and k != ab_idx:
                seq[k] = rng.randint(vocab_lo, vocab_hi)
        for k in range(seq_len):
            if seq[k] == b and k != ab_idx + 1:
                seq[k] = rng.randint(vocab_lo, vocab_hi)
        seq[-1] = a
        tokens[i] = seq
        targets[i] = int(b)
    return torch.from_numpy(tokens), torch.from_numpy(targets)


def make_pre_hook(heads_in_layer, head_dim):
    lo_his = [(h * head_dim, (h + 1) * head_dim) for h in heads_in_layer]
    def pre_hook(_module, ainputs):
        x = ainputs[0].clone()
        for lo, hi in lo_his:
            x[..., lo:hi] = 0
        return (x,) + ainputs[1:]
    return pre_hook


def evaluate(model, tokens, targets, device, batch_size=8):
    n = tokens.shape[0]
    last = tokens.shape[1] - 1
    losses, accs1, accs5 = [], [], []
    sum_logit_B = 0.0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tok = tokens[start:end].to(device)
            tgt = targets[start:end].to(device)
            logits = model(tok).logits[:, last, :]
            loss = F.cross_entropy(logits, tgt, reduction="none")
            losses.append(loss.cpu().float().numpy())
            top1 = logits.argmax(dim=-1)
            accs1.append((top1 == tgt).float().cpu().numpy())
            top5 = logits.topk(5, dim=-1).indices
            accs5.append((top5 == tgt.unsqueeze(-1)).any(dim=-1).float().cpu().numpy())
            sum_logit_B += logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
            if device == "mps":
                torch.mps.empty_cache()
    return {
        "loss": float(np.concatenate(losses).mean()),
        "loss_std": float(np.concatenate(losses).std()),
        "acc_top1": float(np.concatenate(accs1).mean()),
        "acc_top5": float(np.concatenate(accs5).mean()),
        "mean_logit_B": float(sum_logit_B / n),
    }


def run_condition(model, ablate_dict, tokens, targets, device, head_dim,
                  batch_size):
    handles = []
    for layer_idx, heads in ablate_dict.items():
        if not heads:
            continue
        h = model.gpt_neox.layers[layer_idx].attention.dense.register_forward_pre_hook(
            make_pre_hook(heads, head_dim))
        handles.append(h)
    try:
        return evaluate(model, tokens, targets, device, batch_size)
    finally:
        for h in handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1b")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-control-seeds", type=int, default=5)
    ap.add_argument("--out", default="results/pythia_1b_ising/closure_test.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    # Cluster 4 from Pythia 1B k=6 Ising: 25 heads
    cluster_4_heads = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,7),
                        (1,0),(1,2),(1,3),(1,6),
                        (2,0),(2,1),(2,2),(2,3),
                        (3,2),(3,3),(3,5),(3,6),(3,7),
                        (6,1),
                        (8,2),(8,5),
                        (9,0),(9,7)]
    cluster_layers = sorted(set(L for L, _ in cluster_4_heads))
    print(f"cluster 4: n={len(cluster_4_heads)} heads across layers {cluster_layers}")
    n_head = 8

    # build per-layer dict for cluster 4
    by_layer = {L: [] for L in cluster_layers}
    for L, H in cluster_4_heads:
        by_layer[L].append(H)

    # build full-cluster-layers dict (all heads in those layers)
    full_layers_dict = {L: list(range(n_head)) for L in cluster_layers}

    print(f"\nLoading {args.model}...")
    t0 = time.time()
    from transformers import GPTNeoXForCausalLM
    model = GPTNeoXForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                 dtype=torch.float16,
                                                 attn_implementation="eager")
    model = model.to(device).eval()
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(f"  loaded in {time.time()-t0:.0f}s "
          f"({cfg.num_hidden_layers}L × {cfg.num_attention_heads}H, head_dim={head_dim})")

    print("\nBuilding induction batch...")
    tokens, targets = build_induction_batch(args.n_examples, args.seq_len,
                                              rng=np.random.RandomState(42))
    print(f"  batch shape {tuple(tokens.shape)}")

    results = {}

    print("\n--- baseline ---")
    t0 = time.time()
    results["baseline"] = run_condition(model, {}, tokens, targets, device,
                                          head_dim, args.batch_size)
    b = results["baseline"]
    print(f"  {time.time()-t0:.0f}s | loss={b['loss']:.4f} acc1={b['acc_top1']:.4f} "
          f"acc5={b['acc_top5']:.4f} logit_B={b['mean_logit_B']:.3f}")

    print(f"\n--- ablate cluster 4 ({len(cluster_4_heads)} heads) ---")
    t0 = time.time()
    results["ablate_cluster4"] = run_condition(model, by_layer, tokens, targets,
                                                 device, head_dim, args.batch_size)
    r = results["ablate_cluster4"]
    print(f"  {time.time()-t0:.0f}s | loss={r['loss']:.4f} "
          f"Δloss={r['loss']-b['loss']:+.4f} acc1={r['acc_top1']:.4f} "
          f"logit_B={r['mean_logit_B']:.3f}")

    print(f"\n--- ablate ALL heads in cluster layers (upper bound, "
          f"{n_head*len(cluster_layers)} heads) ---")
    t0 = time.time()
    results["ablate_all_in_cluster_layers"] = run_condition(
        model, full_layers_dict, tokens, targets, device, head_dim,
        args.batch_size)
    r = results["ablate_all_in_cluster_layers"]
    print(f"  {time.time()-t0:.0f}s | loss={r['loss']:.4f} "
          f"Δloss={r['loss']-b['loss']:+.4f} acc1={r['acc_top1']:.4f} "
          f"logit_B={r['mean_logit_B']:.3f}")

    # Matched-size random controls: 25 random heads from cluster layers
    print(f"\n--- {args.n_control_seeds} matched-size random controls "
          f"(25 random heads from layers {cluster_layers}) ---")
    rng = np.random.RandomState(0)
    available_heads = [(L, h) for L in cluster_layers for h in range(n_head)]
    cluster_set = set(cluster_4_heads)
    available_non_cluster = [hd for hd in available_heads if hd not in cluster_set]
    print(f"  available heads in cluster layers: {len(available_heads)} "
          f"(non-cluster: {len(available_non_cluster)})")
    control_results = []
    for seed_idx in range(args.n_control_seeds):
        # sample 25 from available_heads (may overlap cluster_4 partially)
        idxs = rng.choice(len(available_heads), size=len(cluster_4_heads),
                          replace=False)
        ctrl = [available_heads[i] for i in idxs]
        # build per-layer dict
        ctrl_by_layer = {}
        for L, H in ctrl:
            ctrl_by_layer.setdefault(L, []).append(H)
        # overlap with cluster 4
        overlap = sum(1 for hd in ctrl if hd in cluster_set)
        t0 = time.time()
        r = run_condition(model, ctrl_by_layer, tokens, targets, device,
                          head_dim, args.batch_size)
        r["seed_idx"] = seed_idx
        r["heads"] = ctrl
        r["overlap_with_cluster4"] = overlap
        control_results.append(r)
        print(f"  seed {seed_idx} (overlap={overlap}/25) | "
              f"{time.time()-t0:.0f}s loss={r['loss']:.4f} "
              f"Δloss={r['loss']-b['loss']:+.4f} acc1={r['acc_top1']:.4f} "
              f"logit_B={r['mean_logit_B']:.3f}")
    results["matched_size_controls"] = control_results

    # Summary
    baseline_loss = b['loss']
    c4_dloss = results['ablate_cluster4']['loss'] - baseline_loss
    full_dloss = results['ablate_all_in_cluster_layers']['loss'] - baseline_loss
    ctrl_dlosses = np.array([c["loss"] - baseline_loss for c in control_results])
    ctrl_acc1s = np.array([c["acc_top1"] for c in control_results])
    ctrl_logitBs = np.array([c["mean_logit_B"] for c in control_results])
    print("\n=== summary ===")
    print(f"  baseline loss: {baseline_loss:.4f}")
    print(f"  cluster 4 Δloss: {c4_dloss:+.4f}")
    print(f"  controls Δloss: {ctrl_dlosses.mean():+.4f} ± {ctrl_dlosses.std():.4f} "
          f"(min {ctrl_dlosses.min():+.4f}, max {ctrl_dlosses.max():+.4f})")
    print(f"  all-cluster-layers Δloss: {full_dloss:+.4f}")
    print(f"  z(cluster 4 vs controls): "
          f"Δloss={(c4_dloss - ctrl_dlosses.mean())/max(ctrl_dlosses.std(), 1e-9):+.2f}σ "
          f"acc1={(results['ablate_cluster4']['acc_top1'] - ctrl_acc1s.mean())/max(ctrl_acc1s.std(), 1e-9):+.2f}σ "
          f"logit_B={(results['ablate_cluster4']['mean_logit_B'] - ctrl_logitBs.mean())/max(ctrl_logitBs.std(), 1e-9):+.2f}σ")
    print(f"  ratio cluster 4 / all-cluster-layers Δloss: "
          f"{c4_dloss / max(abs(full_dloss), 1e-9):.2f} "
          f"(cluster 4 = {len(cluster_4_heads)}/{n_head*len(cluster_layers)} "
          f"= {len(cluster_4_heads)/(n_head*len(cluster_layers)):.2f} of heads)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
