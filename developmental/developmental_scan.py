"""
developmental_scan.py

Iterate over training checkpoints for a single model. For each checkpoint:
  1. Load model at that revision.
  2. Forward pass on the cached natural-text batch with
     output_attentions=True (+ output_router_logits for OLMoE).
  3. Extract per-(example, layer, head) attention pattern at query positions.
  4. Compute template-free signal (max-attn) + per-head median binarize.
  5. Fit pairwise Ising via per-spin L2-regularized logistic regression.
  6. Spectral-cluster |J| at k ∈ {4, 6, 8, 10, 12}; ARI vs end-state
     supervised labels.
  7. Unsupervised quality metrics: ||J||_F, top-k eigenvalue gaps.
  8. For OLMoE: per-layer routing entropy from softmax of router_logits.
  9. Save J.npy + summary JSON per checkpoint.

Usage:
  python developmental_scan.py \
      --model EleutherAI/pythia-1b \
      --revisions step1,step4,step16,...,step143000 \
      --mechinterp-json /path/to/end-state-supervised-labels.json \
      --batch-file /path/to/natural_induction_batch.pt \
      --batch-size 8 \
      --out-dir results/developmental/pythia_1b

Saves to {out-dir}/{revision}.{npy,json}.
"""

from __future__ import annotations

import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


# ─────────────────────────── helpers ───────────────────────────────────


def template_free_signal(attn):
    return attn.max(axis=-1).reshape(attn.shape[0], -1)


def per_head_median_split(signal):
    medians = np.median(signal, axis=0, keepdims=True)
    return np.where(signal > medians, 1.0, -1.0)


def fit_ising(spins, l2=1e-3):
    spins = np.asarray(spins, dtype=np.float64)
    n, F = spins.shape
    J = np.zeros((F, F))
    C = 1.0 / max(l2, 1e-12)
    for i in range(F):
        y = (spins[:, i] > 0).astype(np.int32)
        if y.min() == y.max():
            continue
        mask = np.ones(F, dtype=bool); mask[i] = False
        X = spins[:, mask]
        clf = LogisticRegression(C=C, fit_intercept=True, solver="lbfgs",
                                  max_iter=500, n_jobs=1)
        clf.fit(X, y)
        idxs = np.where(mask)[0]
        j_row = np.zeros(F)
        j_row[idxs] = clf.coef_[0] / 2.0
        J[i, :] = j_row
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    return J


def spectral_cluster_safe(J, ks):
    A = np.abs(J) + 1e-12
    np.fill_diagonal(A, 0)
    out = {}
    for k in ks:
        try:
            sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                     assign_labels="kmeans", random_state=0,
                                     n_init=1)
            out[k] = sc.fit_predict(A).tolist()
        except Exception:
            out[k] = None
    return out


def score_against_gt(labels, gt_map, n_layer, n_head):
    F = n_layer * n_head
    gt, pred = [], []
    for f in range(F):
        L, H = divmod(f, n_head)
        if (L, H) in gt_map:
            gt.append(gt_map[(L, H)]); pred.append(labels[f])
    if not gt: return None
    classes = sorted(set(gt))
    cls2int = {c: i for i, c in enumerate(classes)}
    gt_int = np.array([cls2int[g] for g in gt])
    pred_arr = np.array(pred)
    correct = 0
    for c in np.unique(pred_arr):
        idx = pred_arr == c
        if idx.sum() == 0: continue
        members = np.array(gt)[idx]
        unique, counts = np.unique(members, return_counts=True)
        correct += counts.max()
    return {
        "n_evaluated": len(gt),
        "purity": float(correct / len(gt)),
        "nmi": float(normalized_mutual_info_score(gt_int, pred_arr)),
        "ari": float(adjusted_rand_score(gt_int, pred_arr)),
    }


def top_eigengaps(absJ, k_max=15):
    """Return the top-k eigenvalues of |J| and the largest eigengaps."""
    A = np.abs(absJ).copy()
    np.fill_diagonal(A, 0)
    eigs = np.linalg.eigvalsh(A)
    eigs = np.sort(eigs)[::-1]  # descending
    top = eigs[:k_max + 1]
    gaps = (top[:-1] - top[1:]).tolist()
    return {
        "top_eigenvalues": top[:k_max].tolist(),
        "top_gaps": gaps,  # gap_i = eig_i - eig_{i+1}
        "argmax_gap_below_15": int(np.argmax(gaps)) + 1,  # the gap at this index
    }


# ─────────────────────────── forward pass ──────────────────────────────


def load_model(model_name, revision, device, want_routes=False):
    name_lower = model_name.lower()
    if "olmoe" in name_lower:
        from transformers import OlmoeForCausalLM
        cls = OlmoeForCausalLM
    elif "olmo" in name_lower:
        from transformers import OlmoForCausalLM
        cls = OlmoForCausalLM
    elif "pythia" in name_lower or "gpt-neox" in name_lower:
        from transformers import GPTNeoXForCausalLM
        cls = GPTNeoXForCausalLM
    else:
        from transformers import AutoModelForCausalLM
        cls = AutoModelForCausalLM

    model = cls.from_pretrained(
        model_name, revision=revision,
        dtype=torch.float16,
        attn_implementation="eager",
    )
    return model.to(device).eval()


def extract_attn_and_routes(model, tokens, query_pos, device,
                              want_routes=False, batch_size=8):
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    n, T = tokens.shape

    attn = torch.zeros(n, n_layer, n_head, T, dtype=torch.float32)
    routes = None
    if want_routes:
        n_expert = cfg.num_experts
        routes = torch.zeros(n, n_layer, n_expert, dtype=torch.float32)

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tok = tokens[start:end].to(device)
            out = model(tok, output_attentions=True,
                          output_router_logits=want_routes)
            for j in range(end - start):
                qp = int(query_pos[start + j].item())
                for L in range(n_layer):
                    attn[start + j, L] = out.attentions[L][j, :, qp, :].float().cpu()
                    if want_routes:
                        rl = out.router_logits[L]
                        if rl.dim() == 2:
                            flat_idx = j * T + qp
                            w = torch.softmax(rl[flat_idx].float(), dim=-1)
                        elif rl.dim() == 3:
                            w = torch.softmax(rl[j, qp].float(), dim=-1)
                        else:
                            raise ValueError(f"router_logits shape: {tuple(rl.shape)}")
                        routes[start + j, L] = w.cpu()
            del out
            if device == "mps":
                torch.mps.empty_cache()
            if start > 0 and start % (batch_size * 40) == 0:
                rate = end / (time.time() - t0)
                eta = (n - end) / rate
                print(f"      {end}/{n} ({rate:.1f} ex/s, ETA {eta:.0f}s)",
                      flush=True)
    return attn, routes


# ─────────────────────────── per-checkpoint pipeline ───────────────────


def process_checkpoint(args, revision, gt_map, n_layer_target, n_head_target,
                        tokens, query_pos, want_routes, device):
    """Run forward + Ising + scoring for one checkpoint."""
    ckpt_t0 = time.time()
    out_path_json = Path(args.out_dir) / f"{revision}.json"
    out_path_J = Path(args.out_dir) / f"{revision}_J.npy"
    out_path_routes = Path(args.out_dir) / f"{revision}_routes.npy"

    if out_path_json.exists() and not args.force:
        print(f"  [{revision}] already done, skipping")
        return

    print(f"  [{revision}] loading model...")
    t0 = time.time()
    try:
        model = load_model(args.model, revision, device, want_routes=want_routes)
    except Exception as e:
        print(f"  [{revision}] LOAD FAILED: {e}")
        return
    print(f"    loaded in {time.time()-t0:.0f}s")

    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    n_expert = getattr(cfg, "num_experts", None)
    assert n_layer == n_layer_target and n_head == n_head_target, (
        f"arch mismatch at {revision}: {n_layer}L x {n_head}H vs "
        f"{n_layer_target}L x {n_head_target}H")

    print(f"  [{revision}] forward pass (batch_size={args.batch_size})...")
    attn, routes = extract_attn_and_routes(
        model, tokens, query_pos, device,
        want_routes=want_routes, batch_size=args.batch_size)
    print(f"    forward done in {time.time()-ckpt_t0:.0f}s")
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # template-free signal -> binarize -> Ising
    sig = template_free_signal(attn.numpy())
    spins = per_head_median_split(sig)
    print(f"  [{revision}] fitting Ising ({spins.shape[1]} spins)...")
    J = fit_ising(spins, l2=args.l2)
    np.save(out_path_J, J)

    # unsupervised quality
    J_norm = float(np.linalg.norm(J))
    eig_info = top_eigengaps(J, k_max=15)

    # spectral clustering + ARI vs end-state labels
    cluster_results = spectral_cluster_safe(J, args.ks_spectral)
    metrics_by_k = {}
    for k, labels in cluster_results.items():
        if labels is None: continue
        m = score_against_gt(labels, gt_map, n_layer, n_head)
        if m: metrics_by_k[str(k)] = {**m, "labels": labels}

    # routing entropy (OLMoE only)
    route_summary = None
    if want_routes and routes is not None:
        p = routes.numpy() + 1e-12
        ent = (-(p * np.log(p)).sum(axis=2))  # (N, L)
        per_layer = ent.mean(axis=0).tolist()  # (L,)
        max_ent = float(np.log(n_expert))
        route_summary = {
            "n_expert": n_expert,
            "max_entropy": max_ent,
            "per_layer_mean_entropy": per_layer,
            "overall_mean_entropy": float(np.mean(per_layer)),
            "fraction_of_max": float(np.mean(per_layer) / max_ent),
        }
        np.save(out_path_routes, routes.numpy())

    summary = {
        "model": args.model,
        "revision": revision,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_examples": int(tokens.shape[0]),
        "ising_norm": J_norm,
        "eig_info": eig_info,
        "metrics_by_k": metrics_by_k,
        "routes_summary": route_summary,
        "elapsed_seconds": time.time() - ckpt_t0,
    }
    json.dump(summary, open(out_path_json, "w"), indent=2)

    # quick report
    best = max((m["ari"] for m in metrics_by_k.values()), default=-1)
    if want_routes:
        ent_frac = route_summary['fraction_of_max']
        print(f"  [{revision}] done in {time.time()-ckpt_t0:.0f}s | "
              f"||J||={J_norm:.2f}, best_ARI={best:.3f}, "
              f"H/Hmax={ent_frac:.3f}")
    else:
        print(f"  [{revision}] done in {time.time()-ckpt_t0:.0f}s | "
              f"||J||={J_norm:.2f}, best_ARI={best:.3f}")


# ─────────────────────────── main ──────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revisions", required=True,
                    help="Comma-separated list of HF revisions to scan.")
    ap.add_argument("--mechinterp-json", required=True,
                    help="Path to end-state supervised classifications.")
    ap.add_argument("--batch-file", required=True,
                    help="Cached natural-text batch (tokens + query_pos).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--ks-spectral", type=int, nargs="+",
                    default=[4, 6, 8, 10, 12])
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if {revision}.json already exists.")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    revisions = [r.strip() for r in args.revisions.split(",") if r.strip()]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    print(f"model: {args.model}")
    print(f"out-dir: {args.out_dir}")
    print(f"revisions ({len(revisions)}): {revisions}\n")

    # Load end-state labels (constant across checkpoints)
    mech = json.load(open(args.mechinterp_json))
    gt_map = {(c["layer"], c["head"]): c["classification"]
              for c in mech.get("classifications", [])}
    print(f"supervised labels: {len(gt_map)} heads classified")

    # Load batch (constant across checkpoints)
    batch = torch.load(args.batch_file, weights_only=False)
    tokens = batch["tokens"]; query_pos = batch["query_pos"]
    print(f"natural-text batch: tokens={tuple(tokens.shape)}")

    # Determine model's expected n_layer, n_head from a quick config load.
    # For OLMoE we want routes; otherwise no.
    want_routes = "olmoe" in args.model.lower()
    print(f"want_routes: {want_routes}")

    # Architecture detection: load main once to grab n_layer/n_head
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    print(f"architecture: {n_layer}L × {n_head}H\n")

    total_t0 = time.time()
    for i, revision in enumerate(revisions):
        print(f"========== {i+1}/{len(revisions)}: {revision} ==========")
        try:
            process_checkpoint(args, revision, gt_map, n_layer, n_head,
                                 tokens, query_pos, want_routes, device)
        except Exception as e:
            import traceback
            print(f"  [{revision}] FAILED: {e}")
            traceback.print_exc()
            continue

    print(f"\n=== ALL CHECKPOINTS DONE in {time.time()-total_t0:.0f}s ===")


if __name__ == "__main__":
    main()
