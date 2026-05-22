"""
closure_test_olmo_cluster2.py

Closure test: does ablating the 5-head OLMo 1B Ising cluster 2 (the
100%-pure layer-0 self-attention community discovered unsupervised by the
Bhalla pipeline) cause functionally specific damage?

Conditions:
  1. baseline (no ablation)
  2. ablate cluster 2: L0 H{0, 1, 2, 10, 13}
  3. ablate matched-random L0 controls (5 random heads from the other 11
     L0 heads, NOT in cluster 2) — 5 independent random seeds for error
     bars
  4. ablate ALL 16 L0 heads — upper bound

Metrics on the same 2000-example induction batch (synthetic):
  - LM cross-entropy loss at the last position (= induction target B)
  - top-1 / top-5 accuracy
  - mean logit of target B

Hook style ports olmo_ablation.py: pre-hook on self_attn.o_proj that
zeros the input slice for each ablated head.

Output: results/olmo_1b_ising/closure_test.json
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


def run_condition(model, ablate_layer_to_heads, tokens, targets, device,
                  head_dim, batch_size):
    """ablate_layer_to_heads: dict layer_idx -> list of head indices."""
    handles = []
    for layer_idx, heads in ablate_layer_to_heads.items():
        if not heads:
            continue
        h = model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
            make_pre_hook(heads, head_dim))
        handles.append(h)
    try:
        return evaluate(model, tokens, targets, device, batch_size)
    finally:
        for h in handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-1B-0724-hf")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-control-seeds", type=int, default=5)
    ap.add_argument("--out", default="results/olmo_1b_ising/closure_test.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    # OLMo cluster 2 from k=10 Ising clustering: L0 self-attention heads
    CLUSTER2_HEADS = [0, 1, 2, 10, 13]
    L = 0
    N_HEAD_L0 = 16  # OLMo 1B
    OTHER_L0_HEADS = [h for h in range(N_HEAD_L0) if h not in CLUSTER2_HEADS]
    print(f"cluster 2 (L=0): heads {CLUSTER2_HEADS}")
    print(f"other L0 heads available for controls: {OTHER_L0_HEADS}")

    print(f"\nLoading {args.model}...")
    t0 = time.time()
    from transformers import OlmoForCausalLM
    model = OlmoForCausalLM.from_pretrained(args.model, revision=args.revision,
                                              dtype=torch.float16,
                                              attn_implementation="eager")
    model = model.to(device).eval()
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(f"  loaded in {time.time()-t0:.0f}s "
          f"({cfg.num_hidden_layers}L × {cfg.num_attention_heads}H, "
          f"head_dim={head_dim})")

    print("\nBuilding induction batch...")
    tokens, targets = build_induction_batch(args.n_examples, args.seq_len,
                                              rng=np.random.RandomState(42))
    print(f"  batch shape {tuple(tokens.shape)}")

    results = {}

    # 1. baseline
    print("\n--- baseline ---")
    t0 = time.time()
    results["baseline"] = run_condition(model, {}, tokens, targets, device,
                                          head_dim, args.batch_size)
    print(f"  {time.time()-t0:.0f}s | loss={results['baseline']['loss']:.4f} "
          f"acc1={results['baseline']['acc_top1']:.4f} "
          f"acc5={results['baseline']['acc_top5']:.4f} "
          f"logit_B={results['baseline']['mean_logit_B']:.3f}")

    # 2. ablate cluster 2
    print("\n--- ablate cluster 2 (5 heads, L0 self-attention community) ---")
    t0 = time.time()
    results["ablate_cluster2"] = run_condition(
        model, {L: CLUSTER2_HEADS}, tokens, targets, device, head_dim,
        args.batch_size)
    print(f"  {time.time()-t0:.0f}s | loss={results['ablate_cluster2']['loss']:.4f} "
          f"Δloss={results['ablate_cluster2']['loss']-results['baseline']['loss']:+.4f} "
          f"acc1={results['ablate_cluster2']['acc_top1']:.4f} "
          f"logit_B={results['ablate_cluster2']['mean_logit_B']:.3f}")

    # 3. matched-random L0 controls
    print(f"\n--- {args.n_control_seeds} matched-random controls (5 random L0 heads) ---")
    control_results = []
    rng = np.random.RandomState(0)
    for seed_idx in range(args.n_control_seeds):
        ctrl_heads = sorted(rng.choice(OTHER_L0_HEADS, size=5, replace=False).tolist())
        t0 = time.time()
        r = run_condition(model, {L: ctrl_heads}, tokens, targets, device,
                          head_dim, args.batch_size)
        r["seed_idx"] = seed_idx
        r["heads"] = ctrl_heads
        control_results.append(r)
        print(f"  seed {seed_idx} heads={ctrl_heads} | "
              f"{time.time()-t0:.0f}s loss={r['loss']:.4f} "
              f"Δloss={r['loss']-results['baseline']['loss']:+.4f} "
              f"acc1={r['acc_top1']:.4f} "
              f"logit_B={r['mean_logit_B']:.3f}")
    results["matched_random_controls"] = control_results

    # 4. all L0 heads
    print("\n--- ablate ALL 16 L0 heads (upper bound) ---")
    t0 = time.time()
    results["ablate_all_L0"] = run_condition(
        model, {L: list(range(N_HEAD_L0))}, tokens, targets, device, head_dim,
        args.batch_size)
    print(f"  {time.time()-t0:.0f}s | loss={results['ablate_all_L0']['loss']:.4f} "
          f"Δloss={results['ablate_all_L0']['loss']-results['baseline']['loss']:+.4f} "
          f"acc1={results['ablate_all_L0']['acc_top1']:.4f} "
          f"logit_B={results['ablate_all_L0']['mean_logit_B']:.3f}")

    # summary
    baseline_loss = results['baseline']['loss']
    cluster2_dloss = results['ablate_cluster2']['loss'] - baseline_loss
    ctrl_dlosses = np.array([c["loss"] - baseline_loss for c in control_results])
    print("\n=== summary ===")
    print(f"  baseline loss: {baseline_loss:.4f}")
    print(f"  cluster 2 Δloss: {cluster2_dloss:+.4f}")
    print(f"  controls Δloss: {ctrl_dlosses.mean():+.4f} ± {ctrl_dlosses.std():.4f} "
          f"(min {ctrl_dlosses.min():+.4f}, max {ctrl_dlosses.max():+.4f})")
    z = (cluster2_dloss - ctrl_dlosses.mean()) / max(ctrl_dlosses.std(), 1e-6)
    print(f"  z-score of cluster 2 Δloss vs control distribution: {z:+.2f}σ")
    print(f"  ratio cluster 2 / mean control Δloss: "
          f"{cluster2_dloss / max(abs(ctrl_dlosses.mean()), 1e-9):+.2f}")

    # save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
