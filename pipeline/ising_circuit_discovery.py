"""
ising_circuit_discovery.py

Template-free unsupervised circuit discovery via Bhalla et al. (2026)
SAE-manifold pipeline, applied to ATTENTION HEAD ACTIVATIONS instead of
SAE codes.

Pipeline (option D, template-free):
  1. Run the induction-batch forward pass, extract per-(L, H, example)
     attention pattern at the last (second-A) query position.
  2. Signal per (example, L, H): max attention weight over key positions.
     This is template-free — no target template is picked. It's just
     "is this head focused on something for this input."
  3. Per-head median split → binary spins s_{i, L, H} in {-1, +1}.
     This is the direct analog of Bhalla's sign(z_i): each head is
     "active" on the 50% of examples where its focus exceeds its own
     median.
  4. Fit pairwise Ising model on the binary spins via per-spin
     L2-regularized logistic regression (pseudolikelihood).
  5. Symmetrize the coupling matrix J. Spectral-cluster heads on |J|.
  6. Compare recovered communities to the known head classifications
     from <model>_mechinterp.json (≥30× selectivity ground truth).

Output: numpy+JSON dump in OUT_DIR/<tag>_ising/

Usage:
  python ising_circuit_discovery.py \
      --model EleutherAI/pythia-1b \
      --tag pythia_1b \
      --mechinterp-json .../pythia_mechinterp.json \
      --out-dir results
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


# ─────────────────────────── induction batch ───────────────────────────


def build_induction_batch(n_examples=2000, seq_len=256, vocab_lo=100, vocab_hi=10000,
                          rng=None):
    """Verbatim port of analyses/.../build_induction_batch.

    Each sequence has structure [filler] ... A B ... [filler] ... A. The
    test (query) position is the last index. Induction head: attend back
    to position-of-B (= ab_idx + 1)."""
    if rng is None:
        rng = np.random.RandomState(0)
    tokens = np.zeros((n_examples, seq_len), dtype=np.int64)
    test_pos = np.zeros(n_examples, dtype=np.int64)
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
        test_pos[i] = seq_len - 1
        targets[i] = int(b)

    return torch.from_numpy(tokens), torch.from_numpy(test_pos), torch.from_numpy(targets)


# ─────────────────────────── model + attention ─────────────────────────


def load_model(model_name, revision="main", device="mps"):
    """Load HF causal LM with eager attention so output_attentions works."""
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
        model_name,
        revision=revision,
        dtype=torch.float16,
        attn_implementation="eager",
    )
    model = model.to(device).eval()
    return model


def extract_attn_at_query(model, tokens, query_pos, device, batch_size=8):
    """For each example, extract the full attention pattern at query_pos
    from every (layer, head) to every key position.

    Returns: tensor of shape (n_examples, n_layers, n_heads, seq_len), float32.
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
            for L in range(n_layer):
                attn[start:end, L] = out.attentions[L][:, :, query_pos, :].float().cpu()
            del out
            if device == "mps":
                torch.mps.empty_cache()
            if start > 0 and start % (batch_size * 20) == 0:
                rate = (start + batch_size) / (time.time() - t0)
                eta = (n - end) / rate
                print(f"    {end}/{n}  ({rate:.1f} ex/s, ETA {eta:.0f}s)", flush=True)
    print(f"  attention extraction done in {time.time()-t0:.0f}s")
    return attn


# ─────────────────────────── binarization ──────────────────────────────


def template_free_signal(attn):
    """attn: (n, L, H, T). Returns (n, L*H) signal = max over key positions.

    Template-free: no chosen target. High = focused, low = uniform.
    """
    n, L, H, T = attn.shape
    sig = attn.max(dim=-1).values  # (n, L, H)
    return sig.reshape(n, L * H)


def per_head_median_split(signal):
    """signal: (n, F). Per-head median split → spins in {-1, +1}.

    Each head is "active" (+1) on the 50% of examples where its
    template-free signal exceeds its own median.
    """
    medians = signal.median(dim=0, keepdim=True).values  # (1, F)
    spins = torch.where(signal > medians, 1.0, -1.0)
    return spins, medians


# ─────────────────────────── Ising fit ─────────────────────────────────


def fit_ising_pseudolikelihood(spins, l2=1e-3):
    """spins: (n, F) numpy or tensor with values in {-1, +1}.

    Fit pairwise Ising by per-spin L2-regularized logistic regression:
      P(s_i | s_{-i}) = sigmoid(2 * (h_i + Σ_j J_ij s_j) * s_i)
    Equivalently, regress (s_i == +1) on s_{-i} via logistic regression
    with coefficients = 2 * J_i,: and intercept = 2 * h_i.

    Symmetrize J ← (J + J.T) / 2.

    Returns: J of shape (F, F), h of shape (F,), both numpy.
    """
    from sklearn.linear_model import LogisticRegression

    spins = np.asarray(spins, dtype=np.float64)
    n, F = spins.shape
    J = np.zeros((F, F))
    h = np.zeros(F)
    print(f"  fitting Ising over F={F} spins × n={n} examples (l2={l2})...")
    t0 = time.time()

    # sklearn logistic regression with l2; C = 1/l2 in their convention
    # of the regularization being summed over weights.
    C = 1.0 / max(l2, 1e-12)

    for i in range(F):
        y = (spins[:, i] > 0).astype(np.int32)
        # remove spin i from features
        mask = np.ones(F, dtype=bool)
        mask[i] = False
        X = spins[:, mask]
        if y.min() == y.max():
            # all same — head is constant; skip
            continue
        clf = LogisticRegression(C=C, fit_intercept=True, solver="lbfgs",
                                  max_iter=500)
        clf.fit(X, y)
        coef = clf.coef_[0]   # (F-1,)
        intercept = clf.intercept_[0]
        # coef[k] = 2 * J_{i, idx} where idx is the k-th non-i column
        j_row = np.zeros(F)
        idxs = np.where(mask)[0]
        j_row[idxs] = coef / 2.0
        J[i, :] = j_row
        h[i] = intercept / 2.0
        if (i + 1) % 32 == 0:
            print(f"    fit {i+1}/{F} spins ({(time.time()-t0)/(i+1)*1000:.0f} ms/spin)",
                  flush=True)

    # symmetrize
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    print(f"  Ising fit done in {time.time()-t0:.0f}s")
    return J, h


# ─────────────────────────── community recovery ────────────────────────


def spectral_cluster_J(J, n_clusters_list=(4, 6, 8, 10, 12)):
    """Cluster spins by spectral clustering on |J|.

    Returns: dict mapping n_clusters → labels (F,).
    """
    from sklearn.cluster import SpectralClustering
    A = np.abs(J)
    out = {}
    for k in n_clusters_list:
        try:
            sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                     assign_labels="kmeans", random_state=0)
            labels = sc.fit_predict(A + 1e-12)
            out[k] = labels.tolist()
        except Exception as e:
            print(f"  spectral_cluster(k={k}) failed: {e}")
            out[k] = None
    return out


# ─────────────────────────── ground truth load ─────────────────────────


def load_ground_truth(mech_json_path):
    """Load (layer, head) → class assignment from mechinterp JSON.

    Top-K classified heads get their `classification` (one of:
    induction, previous-token, duplicate-token, first-token, self,
    local, unclassified). Heads not in top-K are labeled "other".
    """
    d = json.load(open(mech_json_path))
    cls_map = {}
    for entry in d.get("classifications", []):
        L = entry["layer"]
        H = entry["head"]
        cls_map[(L, H)] = entry.get("classification", "unclassified")
    return cls_map, d.get("selectivity_threshold"), d.get("top_k")


def comparison_metrics(labels, gt_class_map, n_layer, n_head):
    """Compute purity / NMI / ARI of community labels against ground truth.

    Heads not in gt_class_map are excluded from comparison.
    Returns: dict with purity, ARI, NMI, per-cluster class composition.
    """
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    # build per-head ground truth
    gt = []
    pred = []
    F = n_layer * n_head
    for f in range(F):
        L, H = divmod(f, n_head)
        if (L, H) in gt_class_map:
            gt.append(gt_class_map[(L, H)])
            pred.append(labels[f])
    if len(gt) == 0:
        return {"n_evaluated": 0}

    # purity: for each predicted cluster, take majority class
    pred_arr = np.array(pred)
    gt_arr = np.array(gt)
    clusters = np.unique(pred_arr)
    correct = 0
    composition = {}
    for c in clusters:
        idx = pred_arr == c
        members = gt_arr[idx]
        if len(members) == 0:
            continue
        unique, counts = np.unique(members, return_counts=True)
        maj_count = counts.max()
        correct += maj_count
        composition[int(c)] = {
            str(u): int(v) for u, v in zip(unique, counts)
        }
    purity = float(correct / len(gt))

    # NMI/ARI need integer labels for gt — assign per-class integer
    classes = sorted(set(gt))
    cls2int = {c: i for i, c in enumerate(classes)}
    gt_int = np.array([cls2int[g] for g in gt])
    nmi = float(normalized_mutual_info_score(gt_int, pred_arr))
    ari = float(adjusted_rand_score(gt_int, pred_arr))

    return {
        "n_evaluated": len(gt),
        "purity": purity,
        "nmi": nmi,
        "ari": ari,
        "cluster_composition": composition,
        "gt_classes": classes,
    }


# ─────────────────────────── main ──────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mechinterp-json", required=True,
                    help="Path to existing <model>_mechinterp.json with classifications")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--l2", type=float, default=1e-3,
                    help="L2 penalty for per-spin logistic regression")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / f"{args.tag}_ising"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    # 1. Load model
    print(f"\nloading {args.model}@{args.revision}...")
    t0 = time.time()
    model = load_model(args.model, args.revision, device)
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    print(f"  loaded in {time.time()-t0:.0f}s "
          f"({n_layer}L × {n_head}H, hidden={cfg.hidden_size})")

    # 2. Build induction batch
    rng = np.random.RandomState(args.seed)
    tokens, _, _ = build_induction_batch(n_examples=args.n_examples,
                                           seq_len=args.seq_len, rng=rng)
    query_pos = args.seq_len - 1
    print(f"  batch: {tokens.shape}, query_pos={query_pos}")

    # 3. Extract attention at query position
    print(f"\nextracting attn at query position (batch_size={args.batch_size})...")
    attn = extract_attn_at_query(model, tokens, query_pos, device,
                                   batch_size=args.batch_size)
    del model
    if device == "mps":
        torch.mps.empty_cache()

    # save the raw attention tensor (compressed) for reuse
    attn_path = out_dir / "attn_at_query.npy"
    np.save(attn_path, attn.numpy())
    print(f"  saved {attn_path} ({attn_path.stat().st_size/1e6:.1f} MB)")

    # 4. Template-free signal + binarization
    signal = template_free_signal(attn)            # (n, F)
    spins, medians = per_head_median_split(signal)
    print(f"  signal shape: {signal.shape}, spins shape: {spins.shape}")
    print(f"  mean spin (should be ~0): {spins.float().mean().item():.4f}")

    # 5. Fit Ising
    print("\nfitting Ising pseudolikelihood...")
    J, h = fit_ising_pseudolikelihood(spins.numpy(), l2=args.l2)

    np.save(out_dir / "J.npy", J)
    np.save(out_dir / "h.npy", h)
    print(f"  J.shape={J.shape}, ||J||_F={np.linalg.norm(J):.3f}")
    print(f"  J spectral range: [{J.min():.3f}, {J.max():.3f}]")

    # 6. Spectral cluster
    print("\nspectral clustering on |J|...")
    cluster_results = spectral_cluster_J(J, n_clusters_list=(4, 6, 8, 10, 12))

    # 7. Load ground truth + compare
    print("\nloading ground truth + comparing...")
    gt_map, gt_thresh, gt_topk = load_ground_truth(args.mechinterp_json)
    print(f"  ground truth: {len(gt_map)} heads classified "
          f"(top-{gt_topk}, threshold={gt_thresh})")

    metrics_by_k = {}
    for k, labels in cluster_results.items():
        if labels is None:
            continue
        m = comparison_metrics(labels, gt_map, n_layer, n_head)
        metrics_by_k[k] = m
        print(f"  k={k}: purity={m.get('purity', 0):.3f}, "
              f"nmi={m.get('nmi', 0):.3f}, ari={m.get('ari', 0):.3f}, "
              f"n_eval={m.get('n_evaluated', 0)}")

    # 8. Save everything
    out_json = {
        "model": args.model,
        "revision": args.revision,
        "tag": args.tag,
        "n_examples": args.n_examples,
        "seq_len": args.seq_len,
        "l2": args.l2,
        "n_layer": n_layer,
        "n_head": n_head,
        "signal": "max_attn_at_query",
        "binarization": "per_head_median_split",
        "ising_norm": float(np.linalg.norm(J)),
        "cluster_results": {str(k): v for k, v in cluster_results.items()},
        "metrics_by_k": {str(k): v for k, v in metrics_by_k.items()},
        "ground_truth_threshold": gt_thresh,
        "ground_truth_topk": gt_topk,
    }
    out_path = out_dir / "results.json"
    json.dump(out_json, open(out_path, "w"), indent=2)
    print(f"\nsaved {out_path}")
    print("done.")


if __name__ == "__main__":
    main()
