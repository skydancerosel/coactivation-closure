"""
closure_at_intermediate.py

Closure test at an intermediate training checkpoint.

Given a model, a revision (intermediate checkpoint), and a class C
(first-token / previous-token / self / induction / etc), this script:
  1. Loads the end-state supervised classification from --mech-json
  2. Identifies all end-state-classified heads of class C
  3. At the specified intermediate revision, runs a baseline forward pass
     on the natural-text batch
  4. Runs ablation pass with all class-C heads zeroed at the o_proj input
  5. Runs 5 matched-random-head ablation control passes
  6. Computes z-scores of candidate Δloss/Δacc/Δlogit vs control
     distribution

Output: a single JSON with all per-condition metrics and the verdict.

Architecture-aware hook target:
  Pythia/GPT-NeoX: model.gpt_neox.layers[L].attention.dense
  OLMo/OLMoE:      model.model.layers[L].self_attn.o_proj
"""

from __future__ import annotations

import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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


def install_ablation(model, model_name, ablate_dict, head_dim):
    """Install per-head pre-hooks; architecture-aware."""
    handles = []
    nl = model_name.lower()
    for layer_idx, heads in ablate_dict.items():
        if not heads:
            continue
        if "pythia" in nl or "gpt-neox" in nl:
            mod = model.gpt_neox.layers[layer_idx].attention.dense
        else:
            mod = model.model.layers[layer_idx].self_attn.o_proj
        h = mod.register_forward_pre_hook(make_pre_hook(heads, head_dim))
        handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def evaluate(model, tokens, query_pos, targets, device, batch_size=4):
    n = tokens.shape[0]
    losses = np.zeros(n, dtype=np.float32)
    top1 = np.zeros(n, dtype=np.float32)
    logits_at_tgt = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tok = tokens[start:end].to(device)
            out_logits = model(tok).logits.float()
            for j in range(end - start):
                qp = int(query_pos[start + j].item())
                tgt = int(targets[start + j].item())
                row = out_logits[j, qp, :]
                losses[start + j] = F.cross_entropy(
                    row.unsqueeze(0), torch.tensor([tgt], device=device)).item()
                top1[start + j] = float(int(row.argmax().item()) == tgt)
                logits_at_tgt[start + j] = float(row[tgt].item())
            del out_logits
            if device == "mps":
                torch.mps.empty_cache()
            if start > 0 and start % (batch_size * 25) == 0:
                rate = end / (time.time() - t0)
                eta = (n - end) / rate
                print(f"    {end}/{n} ({rate:.1f} ex/s, ETA {eta:.0f}s)",
                      flush=True)
    return losses, top1, logits_at_tgt


def load_model(model_name, revision, device):
    nl = model_name.lower()
    if "olmoe" in nl:
        from transformers import OlmoeForCausalLM
        cls = OlmoeForCausalLM
    elif "olmo" in nl:
        from transformers import OlmoForCausalLM
        cls = OlmoForCausalLM
    elif "pythia" in nl or "gpt-neox" in nl:
        from transformers import GPTNeoXForCausalLM
        cls = GPTNeoXForCausalLM
    else:
        from transformers import AutoModelForCausalLM
        cls = AutoModelForCausalLM
    model = cls.from_pretrained(model_name, revision=revision,
                                  dtype=torch.float16,
                                  attn_implementation="eager")
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--mech-json", required=True,
                    help="End-state supervised classification")
    ap.add_argument("--target-class", required=True,
                    help="Which class's heads to ablate")
    ap.add_argument("--batch-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-controls", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    # Load end-state classification, get heads of target class
    mech = json.load(open(args.mech_json))
    class_map = {(c["layer"], c["head"]): c["classification"]
                 for c in mech.get("classifications", [])}
    target_heads = [(L, H) for (L, H), cls in class_map.items()
                     if cls == args.target_class]
    print(f"target class: {args.target_class}")
    print(f"end-state heads of this class: {len(target_heads)}")
    if not target_heads:
        print("no heads to ablate — exiting")
        return
    by_layer = {}
    for L, H in target_heads:
        by_layer.setdefault(int(L), []).append(int(H))

    # Load batch
    batch = torch.load(args.batch_file, weights_only=False)
    tokens = batch["tokens"]
    query_pos = batch["query_pos"]
    targets = batch["targets"]
    print(f"batch: {tuple(tokens.shape)}")

    # Load model at intermediate revision
    print(f"loading {args.model}@{args.revision}...")
    t0 = time.time()
    model = load_model(args.model, args.revision, device)
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_head
    print(f"  loaded in {time.time()-t0:.0f}s "
          f"({n_layer}L × {n_head}H, head_dim={head_dim})")

    results = {
        "model": args.model,
        "revision": args.revision,
        "target_class": args.target_class,
        "n_target_heads": len(target_heads),
        "n_layer": n_layer,
        "n_head": n_head,
    }

    # Baseline
    print("\n--- baseline ---")
    t0 = time.time()
    L_base, A_base, Z_base = evaluate(model, tokens, query_pos, targets,
                                         device, args.batch_size)
    base_loss = float(L_base.mean())
    base_acc = float(A_base.mean())
    base_logit = float(Z_base.mean())
    print(f"  {time.time()-t0:.0f}s | loss={base_loss:.4f} acc={base_acc:.4f} "
          f"logit={base_logit:.3f}")
    results["baseline"] = {"loss": base_loss, "acc_top1": base_acc,
                           "mean_logit_target": base_logit}

    # Candidate
    print(f"\n--- candidate: {len(target_heads)} {args.target_class} heads ---")
    t0 = time.time()
    handles = install_ablation(model, args.model, by_layer, head_dim)
    L_c, A_c, Z_c = evaluate(model, tokens, query_pos, targets, device,
                               args.batch_size)
    remove_hooks(handles)
    cand_loss = float(L_c.mean())
    cand_acc = float(A_c.mean())
    cand_logit = float(Z_c.mean())
    dL_c = cand_loss - base_loss
    dA_c = cand_acc - base_acc
    dZ_c = cand_logit - base_logit
    print(f"  {time.time()-t0:.0f}s | Δloss={dL_c:+.4f} Δacc={dA_c:+.4f} "
          f"Δlogit={dZ_c:+.3f}")
    results["candidate_ablation"] = {
        "loss": cand_loss, "acc_top1": cand_acc, "mean_logit_target": cand_logit,
        "ablated_heads": [list(p) for p in target_heads],
        "Δloss": dL_c, "Δacc": dA_c, "Δlogit": dZ_c,
    }

    # Controls
    print(f"\n--- {args.n_controls} matched-random controls ---")
    rng = np.random.RandomState(0)
    target_set = set(tuple(p) for p in target_heads)
    all_heads = [(L, H) for L in range(n_layer) for H in range(n_head)]
    available = [hd for hd in all_heads if hd not in target_set]
    ctrl_records = []
    for seed_idx in range(args.n_controls):
        idxs = rng.choice(len(available), size=len(target_heads), replace=False)
        ctrl_heads = [available[i] for i in idxs]
        ctrl_by_layer = {}
        for L, H in ctrl_heads:
            ctrl_by_layer.setdefault(L, []).append(H)
        t0 = time.time()
        handles = install_ablation(model, args.model, ctrl_by_layer, head_dim)
        L_x, A_x, Z_x = evaluate(model, tokens, query_pos, targets, device,
                                    args.batch_size)
        remove_hooks(handles)
        dL = float(L_x.mean()) - base_loss
        dA = float(A_x.mean()) - base_acc
        dZ = float(Z_x.mean()) - base_logit
        print(f"  seed {seed_idx}: Δloss={dL:+.4f} Δacc={dA:+.4f} "
              f"Δlogit={dZ:+.3f} ({time.time()-t0:.0f}s)")
        ctrl_records.append({
            "seed": seed_idx, "heads": ctrl_heads,
            "Δloss": dL, "Δacc": dA, "Δlogit": dZ,
        })
    results["controls"] = ctrl_records

    # Z-scores
    ctrl_dL = np.array([c["Δloss"] for c in ctrl_records])
    ctrl_dA = np.array([c["Δacc"] for c in ctrl_records])
    ctrl_dZ = np.array([c["Δlogit"] for c in ctrl_records])
    def z(x, arr):
        if arr.std() < 1e-9: return 0.0
        return float((x - arr.mean()) / arr.std())
    z_dL = z(dL_c, ctrl_dL)
    z_dA = z(dA_c, ctrl_dA)
    z_dZ = z(dZ_c, ctrl_dZ)
    p_exceed = float(np.mean(ctrl_dL >= dL_c))

    print(f"\n=== closure summary ===")
    print(f"  baseline loss: {base_loss:.4f}")
    print(f"  candidate Δloss: {dL_c:+.4f}  z={z_dL:+.2f}σ")
    print(f"  candidate Δacc:  {dA_c:+.4f}  z={z_dA:+.2f}σ")
    print(f"  candidate Δlogit:{dZ_c:+.3f}  z={z_dZ:+.2f}σ")
    print(f"  P[control ≥ candidate on Δloss]: {p_exceed:.2f}")
    results["summary"] = {
        "z_dloss": z_dL, "z_dacc": z_dA, "z_dlogit": z_dZ,
        "p_exceed_dloss": p_exceed,
        "control_dloss_mean": float(ctrl_dL.mean()),
        "control_dloss_std": float(ctrl_dL.std()),
    }

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
