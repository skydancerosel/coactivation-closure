"""
closure_pythia_natural.py

Closure test on Pythia 1B natural-text Ising cluster 6 (9 heads, L0-L3,
isolation 2.47x).

Pythia uses GPT-NeoX; the per-head ablation hook target is
`gpt_neox.layers[L].attention.dense` (the o_proj equivalent), as in
nostalgic-lederberg's pythia_ablation.py.
"""

from __future__ import annotations

import argparse, json, time
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
    handles = []
    for layer_idx, heads in ablate_dict.items():
        if not heads:
            continue
        h = model.gpt_neox.layers[layer_idx].attention.dense.register_forward_pre_hook(
            make_pre_hook(heads, head_dim))
        handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def evaluate(model, tokens, query_pos, targets, device, batch_size=8):
    n = tokens.shape[0]
    per_loss = np.zeros(n, dtype=np.float32)
    per_top1 = np.zeros(n, dtype=np.float32)
    per_logit = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tok = tokens[start:end].to(device)
            logits = model(tok).logits.float()
            for j in range(end - start):
                qp = int(query_pos[start + j].item())
                tgt = int(targets[start + j].item())
                row = logits[j, qp, :]
                per_loss[start + j] = F.cross_entropy(
                    row.unsqueeze(0), torch.tensor([tgt], device=device)).item()
                per_top1[start + j] = float(int(row.argmax().item()) == tgt)
                per_logit[start + j] = float(row[tgt].item())
            del logits
            if device == "mps":
                torch.mps.empty_cache()
            if start > 0 and start % (batch_size * 25) == 0:
                rate = end / (time.time() - t0)
                eta = (n - end) / rate
                print(f"    {end}/{n} ({rate:.1f} ex/s, ETA {eta:.0f}s)",
                      flush=True)
    print(f"  eval done in {time.time()-t0:.0f}s")
    return per_loss, per_top1, per_logit


def summarize(per_loss, per_top1, per_logit):
    return {
        "n": int(per_loss.shape[0]),
        "loss": float(per_loss.mean()),
        "loss_std": float(per_loss.std()),
        "acc_top1": float(per_top1.mean()),
        "mean_logit_target": float(per_logit.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1b")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--batch-file", default="/Volumes/Brandy/mini_gpt/.claude/worktrees/nostalgic-lederberg-80a58d/natural_induction_batch.pt")
    ap.add_argument("--candidate-json", default="results/pythia_1b_nat_ising/closure_candidate.json")
    ap.add_argument("--out", default="results/pythia_1b_nat_ising/closure_test.json")
    ap.add_argument("--n-controls", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    candidate = json.load(open(args.candidate_json))["candidate"]
    cand_heads = [tuple(h) for h in candidate["heads"]]
    print(f"\ncandidate: {candidate['size']} heads, layers {candidate['layers']}")
    print(f"  heads: {cand_heads}")
    by_layer = {}
    for L, H in cand_heads:
        by_layer.setdefault(int(L), []).append(int(H))

    print(f"\nloading natural-text batch...")
    batch = torch.load(args.batch_file, weights_only=False)
    tokens = batch["tokens"]; query_pos = batch["query_pos"]; targets = batch["targets"]
    print(f"  tokens={tuple(tokens.shape)}")

    print(f"\nloading {args.model}...")
    t0 = time.time()
    from transformers import GPTNeoXForCausalLM
    model = GPTNeoXForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float16,
        attn_implementation="eager")
    model = model.to(device).eval()
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_head
    print(f"  loaded in {time.time()-t0:.0f}s ({n_layer}L × {n_head}H, head_dim={head_dim})")

    results = {}

    print("\n=== CONDITION: baseline (no ablation) ===")
    loss, top1, logit = evaluate(model, tokens, query_pos, targets, device, args.batch_size)
    results["baseline"] = summarize(loss, top1, logit)
    print(f"  baseline: loss={results['baseline']['loss']:.4f} "
          f"acc1={results['baseline']['acc_top1']:.4f} "
          f"logit={results['baseline']['mean_logit_target']:.3f}")

    print(f"\n=== CONDITION: ablate candidate ({candidate['size']} heads) ===")
    handles = install_ablation(model, by_layer, head_dim)
    loss, top1, logit = evaluate(model, tokens, query_pos, targets, device, args.batch_size)
    remove_hooks(handles)
    results["candidate_ablation"] = summarize(loss, top1, logit)
    results["candidate_ablation"]["ablated_heads"] = cand_heads
    results["candidate_ablation"]["by_layer"] = {str(k): v for k, v in by_layer.items()}
    dloss = results['candidate_ablation']['loss'] - results['baseline']['loss']
    print(f"  candidate: loss={results['candidate_ablation']['loss']:.4f} "
          f"(Δ={dloss:+.4f})")

    print(f"\n=== CONDITION: {args.n_controls} matched-random controls ===")
    rng = np.random.RandomState(0)
    cand_set = set((L, H) for L, H in cand_heads)
    all_heads = [(L, H) for L in range(n_layer) for H in range(n_head)]
    available = [hd for hd in all_heads if hd not in cand_set]
    control_results = []
    for seed_idx in range(args.n_controls):
        idxs = rng.choice(len(available), size=len(cand_heads), replace=False)
        ctrl_heads = [available[i] for i in idxs]
        ctrl_by_layer = {}
        for L, H in ctrl_heads:
            ctrl_by_layer.setdefault(L, []).append(H)
        handles = install_ablation(model, ctrl_by_layer, head_dim)
        loss, top1, logit = evaluate(model, tokens, query_pos, targets, device, args.batch_size)
        remove_hooks(handles)
        rec = summarize(loss, top1, logit)
        rec["seed_idx"] = seed_idx
        rec["heads"] = ctrl_heads
        control_results.append(rec)
        dl = rec["loss"] - results["baseline"]["loss"]
        print(f"  seed {seed_idx}: heads={ctrl_heads}")
        print(f"    loss={rec['loss']:.4f} (Δ={dl:+.4f}) acc1={rec['acc_top1']:.4f}")
    results["controls"] = control_results

    b_loss = results['baseline']['loss']
    cand_dl = results['candidate_ablation']['loss'] - b_loss
    ctrl_dls = np.array([c['loss'] - b_loss for c in control_results])
    z = (cand_dl - ctrl_dls.mean()) / max(ctrl_dls.std(), 1e-9)
    p_exceed = float(np.mean(ctrl_dls >= cand_dl))

    cand_dacc = results['candidate_ablation']['acc_top1'] - results['baseline']['acc_top1']
    ctrl_daccs = np.array([c['acc_top1'] - results['baseline']['acc_top1']
                            for c in control_results])
    z_acc = (cand_dacc - ctrl_daccs.mean()) / max(ctrl_daccs.std(), 1e-9)

    cand_dlogit = results['candidate_ablation']['mean_logit_target'] - results['baseline']['mean_logit_target']
    ctrl_dlogits = np.array([c['mean_logit_target'] - results['baseline']['mean_logit_target']
                              for c in control_results])
    z_logit = (cand_dlogit - ctrl_dlogits.mean()) / max(ctrl_dlogits.std(), 1e-9)

    print("\n=== closure summary ===")
    print(f"  baseline loss: {b_loss:.4f}")
    print(f"  candidate Δloss: {cand_dl:+.4f}")
    print(f"  controls Δloss: {ctrl_dls.mean():+.4f} ± {ctrl_dls.std():.4f} "
          f"(min {ctrl_dls.min():+.4f}, max {ctrl_dls.max():+.4f})")
    print(f"  z(candidate vs controls): loss={z:+.2f}σ acc={z_acc:+.2f}σ "
          f"logit_target={z_logit:+.2f}σ")
    print(f"  P[control ≥ candidate]: {p_exceed:.2f}")

    results["summary"] = {
        "baseline_loss": b_loss,
        "candidate_dloss": cand_dl,
        "control_dloss_mean": float(ctrl_dls.mean()),
        "control_dloss_std": float(ctrl_dls.std()),
        "control_dloss_min": float(ctrl_dls.min()),
        "control_dloss_max": float(ctrl_dls.max()),
        "z_dloss": float(z),
        "z_dacc": float(z_acc),
        "z_dlogit": float(z_logit),
        "p_exceed_dloss": p_exceed,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2, default=float)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
