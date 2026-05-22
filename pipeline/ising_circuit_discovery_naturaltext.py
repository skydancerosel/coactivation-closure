"""
ising_circuit_discovery_naturaltext.py

Variant of ising_circuit_discovery.py that uses the existing
natural_induction_batch.pt (2000 natural-text examples, per-example query
positions) instead of the synthetic [filler] A B [filler] A induction
batch.

Same template-free (option D) pipeline:
  1. forward pass, extract attn[i, L, H, :] at per-example query_pos
  2. max-attn signal, per-head median split → spins
  3. fit pairwise Ising via per-spin L2 logistic regression
  4. spectral cluster on |J|
  5. compare to existing supervised classification

The supervised ground truth here is the model's *_mechinterp_naturaltext.json
classifications (selectivity ≥30× on natural text), which differ from the
synthetic-batch classifications.

Usage:
  python ising_circuit_discovery_naturaltext.py \
      --model EleutherAI/pythia-1b --tag pythia_1b_nat \
      --batch-file natural_induction_batch.pt \
      --mechinterp-json pythia_mechinterp_naturaltext.json \
      --out-dir results
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def load_model(model_name, revision, device):
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
        dtype=torch.float16, attn_implementation="eager",
    )
    return model.to(device).eval()


def extract_attn_per_example_query(model, tokens, query_pos, device,
                                     batch_size=8):
    """For each example, extract attention at its own query position.

    Returns: (N, L, H, T) tensor — for each example i, contains the full
    attention pattern from query_pos[i] to every key position.
    """
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    n, T = tokens.shape
    attn = torch.zeros(n, n_layer, n_head, T, dtype=torch.float32)

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tok = tokens[start:end].to(device)
            out = model(tok, output_attentions=True)
            # for each example in this batch, pull attn at its own query_pos
            for j in range(end - start):
                qp = int(query_pos[start + j].item())
                for L in range(n_layer):
                    # out.attentions[L]: (B, H, T, T)
                    attn[start + j, L] = out.attentions[L][j, :, qp, :].float().cpu()
            del out
            if device == "mps":
                torch.mps.empty_cache()
            if start > 0 and start % (batch_size * 20) == 0:
                rate = (start + batch_size) / (time.time() - t0)
                eta = (n - end) / rate
                print(f"    {end}/{n}  ({rate:.1f} ex/s, ETA {eta:.0f}s)", flush=True)
    print(f"  natural-text attention extraction in {time.time()-t0:.0f}s")
    return attn


def template_free_signal(attn):
    return attn.max(dim=-1).values.reshape(attn.shape[0], -1)


def per_head_median_split(signal):
    medians = signal.median(dim=0, keepdim=True).values
    return torch.where(signal > medians, 1.0, -1.0)


def fit_ising_pseudolikelihood(spins, l2=1e-3):
    from sklearn.linear_model import LogisticRegression
    spins = np.asarray(spins, dtype=np.float64)
    n, F = spins.shape
    J = np.zeros((F, F))
    h = np.zeros(F)
    C = 1.0 / max(l2, 1e-12)
    t0 = time.time()
    for i in range(F):
        y = (spins[:, i] > 0).astype(np.int32)
        if y.min() == y.max():
            continue
        mask = np.ones(F, dtype=bool); mask[i] = False
        X = spins[:, mask]
        clf = LogisticRegression(C=C, fit_intercept=True, solver="lbfgs",
                                  max_iter=500)
        clf.fit(X, y)
        idxs = np.where(mask)[0]
        j_row = np.zeros(F)
        j_row[idxs] = clf.coef_[0] / 2.0
        J[i, :] = j_row
        h[i] = clf.intercept_[0] / 2.0
        if (i + 1) % 32 == 0:
            print(f"    {i+1}/{F} spins ({(time.time()-t0)/(i+1)*1000:.0f} ms/spin)",
                  flush=True)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    return J, h


def spectral_cluster(J, ks):
    from sklearn.cluster import SpectralClustering
    A = np.abs(J) + 1e-12
    np.fill_diagonal(A, 0)
    out = {}
    for k in ks:
        try:
            sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                     assign_labels="kmeans", random_state=0)
            out[k] = sc.fit_predict(A).tolist()
        except Exception as e:
            out[k] = None
    return out


def score_against_gt(labels, mech_json_path, n_layer, n_head):
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    mech = json.load(open(mech_json_path))
    gt_map = {(c["layer"], c["head"]): c["classification"]
              for c in mech.get("classifications", [])}
    F = n_layer * n_head
    gt, pred = [], []
    for f in range(F):
        L, H = divmod(f, n_head)
        if (L, H) in gt_map:
            gt.append(gt_map[(L, H)]); pred.append(labels[f])
    if not gt:
        return None
    classes = sorted(set(gt))
    cls2int = {c: i for i, c in enumerate(classes)}
    gt_int = np.array([cls2int[g] for g in gt])
    pred_arr = np.array(pred)
    correct = 0
    composition = {}
    for c in np.unique(pred_arr):
        idx = pred_arr == c
        if idx.sum() == 0: continue
        members = np.array(gt)[idx]
        unique, counts = np.unique(members, return_counts=True)
        correct += counts.max()
        composition[int(c)] = {str(u): int(v) for u, v in zip(unique, counts)}
    return {
        "n_evaluated": len(gt),
        "purity": float(correct / len(gt)),
        "nmi": float(normalized_mutual_info_score(gt_int, pred_arr)),
        "ari": float(adjusted_rand_score(gt_int, pred_arr)),
        "composition": composition,
    }


def cluster_agreement_with_synthetic(labels_nat, labels_syn, n_layer, n_head):
    """ARI between natural-text and synthetic-text cluster labels —
    how stable are communities across input distributions?"""
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    a = np.asarray(labels_nat); b = np.asarray(labels_syn)
    return {
        "nmi": float(normalized_mutual_info_score(b, a)),
        "ari": float(adjusted_rand_score(b, a)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch-file", default="/Volumes/Brandy/mini_gpt/.claude/worktrees/nostalgic-lederberg-80a58d/natural_induction_batch.pt")
    ap.add_argument("--mechinterp-json", required=True,
                    help="natural-text mechinterp.json for supervised ground truth")
    ap.add_argument("--synthetic-results-dir", default=None,
                    help="results dir of synthetic-batch Ising for cross-distribution stability")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--l2", type=float, default=1e-3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / f"{args.tag}_nat_ising"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    print(f"\nloading {args.model}...")
    t0 = time.time()
    model = load_model(args.model, args.revision, device)
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    print(f"  loaded in {time.time()-t0:.0f}s ({n_layer}L × {n_head}H)")

    print(f"\nloading natural-text batch from {args.batch_file}...")
    batch = torch.load(args.batch_file, weights_only=False)
    tokens = batch["tokens"]
    query_pos = batch["query_pos"]
    print(f"  tokens={tuple(tokens.shape)} dtype={tokens.dtype}; "
          f"query_pos range [{query_pos.min().item()}, {query_pos.max().item()}]")

    print(f"\nextracting attention at per-example query positions (bs={args.batch_size})...")
    attn = extract_attn_per_example_query(model, tokens, query_pos, device,
                                           batch_size=args.batch_size)
    del model
    if device == "mps":
        torch.mps.empty_cache()

    np.save(out_dir / "attn_at_query.npy", attn.numpy())

    sig = template_free_signal(attn)
    spins = per_head_median_split(sig)
    print(f"\nfitting Ising pseudolikelihood (l2={args.l2})...")
    J, h = fit_ising_pseudolikelihood(spins.numpy(), l2=args.l2)
    np.save(out_dir / "J.npy", J); np.save(out_dir / "h.npy", h)
    print(f"  ||J||_F={np.linalg.norm(J):.3f}")

    ks = [4, 6, 8, 10, 12]
    print(f"\nspectral clustering on |J|, k in {ks}...")
    cluster_results = spectral_cluster(J, ks)

    metrics = {}
    for k, labels in cluster_results.items():
        if labels is None: continue
        m = score_against_gt(labels, args.mechinterp_json, n_layer, n_head)
        if m is None:
            continue
        metrics[str(k)] = m
        print(f"  k={k}: n={m['n_evaluated']} purity={m['purity']:.3f} "
              f"nmi={m['nmi']:.3f} ari={m['ari']:.3f}")

    # Cross-distribution stability: how do natural-text clusters compare to
    # synthetic-text clusters from the parallel synthetic Ising run?
    stability = {}
    if args.synthetic_results_dir is not None:
        syn_res = json.load(open(Path(args.synthetic_results_dir) / "results.json"))
        for k in ks:
            if str(k) not in syn_res["cluster_results"]:
                continue
            syn_labels = syn_res["cluster_results"][str(k)]
            if syn_labels is None: continue
            nat_labels = cluster_results.get(k)
            if nat_labels is None: continue
            agr = cluster_agreement_with_synthetic(nat_labels, syn_labels,
                                                     n_layer, n_head)
            stability[str(k)] = agr
        print(f"\ncluster stability across natural ↔ synthetic distributions:")
        for k, a in stability.items():
            print(f"  k={k}: NMI={a['nmi']:.3f}, ARI={a['ari']:.3f}")

    out_json = {
        "model": args.model,
        "tag": args.tag,
        "batch_file": args.batch_file,
        "n_layer": n_layer, "n_head": n_head,
        "cluster_results": {str(k): v for k, v in cluster_results.items()},
        "metrics_vs_supervised_natural": metrics,
        "synthetic_natural_cluster_stability": stability,
    }
    json.dump(out_json, open(out_dir / "results.json", "w"), indent=2)
    print(f"\nsaved {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
