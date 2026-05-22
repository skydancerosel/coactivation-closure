"""
route_conditioned_ising_olmoe.py

Snapshot route-conditioned Ising on OLMoE-1B-7B-0924 natural text.

Hypothesis: marginal Ising on OLMoE natural-text head co-activation gives
ARI ≈ 0 (vs 0.193 on synthetic) because routing to different experts
makes head co-activation conditionally varying. Conditioning on routes
should recover structure if any exists.

Two complementary tests:

(A) ROUTE-STRATIFIED Ising. Cluster examples by their routing pattern
    (k-means on per-layer routing weights). Within each route cluster,
    fit a separate Ising on the 256 head spins. Score within-cluster
    ARI vs supervised natural-text classification. Compare per-cluster
    mean and max ARI to the marginal baseline.

(B) ROUTE-COVARIATE pooled Ising. Single per-spin logistic regression
    with route-cluster membership as additional one-hot covariates
    (K-1 features). The coefficients on other head spins give
    conditional couplings *after accounting for route cluster*. Spectral
    cluster |J|, score vs supervised classification.

Output: results/olmoe_route_conditioned/
  - attn_at_query.npy
  - routes_at_query.npy  (softmax of router_logits at query position)
  - route_cluster_assignments.json
  - results.json  (per-K, per-strategy ARI)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def template_free_signal(attn):
    return attn.max(dim=-1).values.reshape(attn.shape[0], -1)


def per_head_median_split(signal):
    medians = signal.median(dim=0, keepdim=True).values
    return torch.where(signal > medians, 1.0, -1.0)


def fit_ising_pseudolikelihood(spins, extra_X=None, l2=1e-3, verbose=False):
    """Fit pairwise Ising; if extra_X is provided, augment each per-spin LR
    with those covariates (so the head-spin coefficients are conditional
    on extra_X).

    spins: (N, F) in {-1, +1}
    extra_X: (N, M) covariates (one-hot route cluster) or None.
    Returns J (F, F), h (F,) — only the head-spin part.
    """
    from sklearn.linear_model import LogisticRegression
    spins = np.asarray(spins, dtype=np.float64)
    n, F = spins.shape
    J = np.zeros((F, F))
    h = np.zeros(F)
    C = 1.0 / max(l2, 1e-12)
    if extra_X is not None:
        extra_X = np.asarray(extra_X, dtype=np.float64)
    for i in range(F):
        y = (spins[:, i] > 0).astype(np.int32)
        if y.min() == y.max():
            continue
        mask = np.ones(F, dtype=bool); mask[i] = False
        X_spins = spins[:, mask]
        if extra_X is not None:
            X = np.hstack([X_spins, extra_X])
        else:
            X = X_spins
        clf = LogisticRegression(C=C, fit_intercept=True, solver="lbfgs",
                                  max_iter=500)
        clf.fit(X, y)
        idxs = np.where(mask)[0]
        j_row = np.zeros(F)
        j_row[idxs] = clf.coef_[0][:F - 1] / 2.0
        J[i, :] = j_row
        h[i] = clf.intercept_[0] / 2.0
        if verbose and (i + 1) % 32 == 0:
            print(f"    fit {i+1}/{F} spins", flush=True)
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
        except Exception:
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


def extract_attn_and_routes(model, tokens, query_pos, device, batch_size=4):
    """For each example, extract attention pattern at query_pos AND the
    softmaxed router weights at query_pos per layer.

    Returns:
        attn: (N, L, H, T) float32
        routes: (N, L, E) float32 — softmax of router_logits at query position
    """
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    n_expert = cfg.num_experts
    n, T = tokens.shape

    attn = torch.zeros(n, n_layer, n_head, T, dtype=torch.float32)
    routes = torch.zeros(n, n_layer, n_expert, dtype=torch.float32)

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tok = tokens[start:end].to(device)
            out = model(tok, output_attentions=True, output_router_logits=True)
            for j in range(end - start):
                qp = int(query_pos[start + j].item())
                for L in range(n_layer):
                    attn[start + j, L] = out.attentions[L][j, :, qp, :].float().cpu()
                    rl = out.router_logits[L]
                    # OLMoE returns router_logits per layer as (B*T, n_expert) flattened
                    if rl.dim() == 2:
                        flat_idx = j * T + qp
                        weights = torch.softmax(rl[flat_idx].float(), dim=-1)
                    elif rl.dim() == 3:
                        weights = torch.softmax(rl[j, qp].float(), dim=-1)
                    else:
                        raise ValueError(f"Unexpected router_logits shape: {tuple(rl.shape)}")
                    routes[start + j, L] = weights.cpu()
            del out
            if device == "mps":
                torch.mps.empty_cache()
            if start > 0 and start % (batch_size * 20) == 0:
                rate = (start + batch_size) / (time.time() - t0)
                eta = (n - end) / rate
                print(f"    {end}/{n}  ({rate:.1f} ex/s, ETA {eta:.0f}s)", flush=True)
    print(f"  attn+routes extracted in {time.time()-t0:.0f}s")
    return attn, routes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--batch-file", default="/Volumes/Brandy/mini_gpt/.claude/worktrees/nostalgic-lederberg-80a58d/natural_induction_batch.pt")
    ap.add_argument("--mechinterp-json", default="/Volumes/Brandy/mini_gpt/.claude/worktrees/nostalgic-lederberg-80a58d/olmoe_mechinterp_naturaltext.json")
    ap.add_argument("--out-dir", default="results/olmoe_route_conditioned")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--ks-cluster", type=int, nargs="+", default=[4, 8, 16],
                    help="Number of route clusters for stratification")
    ap.add_argument("--ks-spectral", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--min-cluster-size", type=int, default=80)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    print(f"output dir: {out_dir}")

    print(f"\nloading {args.model}...")
    from transformers import OlmoeForCausalLM
    t0 = time.time()
    model = OlmoeForCausalLM.from_pretrained(args.model, revision=args.revision,
                                              dtype=torch.float16,
                                              attn_implementation="eager")
    model = model.to(device).eval()
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    n_expert = cfg.num_experts
    print(f"  loaded in {time.time()-t0:.0f}s "
          f"({n_layer}L × {n_head}H × {n_expert}E)")

    print(f"\nloading natural-text batch...")
    batch = torch.load(args.batch_file, weights_only=False)
    tokens = batch["tokens"]; query_pos = batch["query_pos"]
    print(f"  tokens={tuple(tokens.shape)} query_pos range "
          f"[{query_pos.min().item()}, {query_pos.max().item()}]")

    print(f"\nextracting attention + routes (batch_size={args.batch_size})...")
    attn, routes = extract_attn_and_routes(model, tokens, query_pos, device,
                                             batch_size=args.batch_size)
    del model
    if device == "mps":
        torch.mps.empty_cache()

    np.save(out_dir / "attn_at_query.npy", attn.numpy())
    np.save(out_dir / "routes_at_query.npy", routes.numpy())
    print(f"  saved {out_dir / 'attn_at_query.npy'} and routes_at_query.npy")

    # Build route features for clustering: flatten (N, L, E) -> (N, L*E)
    N = routes.shape[0]
    route_features = routes.reshape(N, -1).numpy().astype(np.float32)
    print(f"  route_features shape: {route_features.shape}")

    # Mean routing entropy as a summary statistic
    mean_route_entropy_per_layer = []
    for L in range(n_layer):
        p = routes[:, L, :].numpy() + 1e-12
        ent = -np.sum(p * np.log(p), axis=1).mean()
        mean_route_entropy_per_layer.append(float(ent))
    overall_mean_entropy = float(np.mean(mean_route_entropy_per_layer))
    print(f"  mean per-layer routing entropy: {overall_mean_entropy:.3f} "
          f"(max={np.log(n_expert):.3f})")

    # Build template-free signal + spins from attention
    sig = template_free_signal(attn)
    spins = per_head_median_split(sig)
    print(f"  spins shape: {spins.shape}")

    # --- BASELINE: marginal Ising (no route conditioning) ---
    print("\n[BASELINE] marginal Ising (no route conditioning)...")
    J_marg, _ = fit_ising_pseudolikelihood(spins.numpy(), extra_X=None,
                                             l2=args.l2)
    np.save(out_dir / "J_marginal.npy", J_marg)
    marg_clusters = spectral_cluster(J_marg, args.ks_spectral)
    marg_metrics = {}
    for k, labels in marg_clusters.items():
        if labels is None: continue
        m = score_against_gt(labels, args.mechinterp_json, n_layer, n_head)
        if m: marg_metrics[str(k)] = m
        print(f"  marginal k={k}: purity={m['purity']:.3f} nmi={m['nmi']:.3f} "
              f"ari={m['ari']:.3f}")

    # --- STRATEGY A: stratified Ising per route cluster ---
    print("\n[STRATEGY A] route-stratified Ising...")
    from sklearn.cluster import KMeans
    strategy_A = {}
    cluster_assignments = {}
    for K in args.ks_cluster:
        print(f"\n  K_route_clusters={K}:")
        km = KMeans(n_clusters=K, random_state=0, n_init=10)
        cluster_labels = km.fit_predict(route_features)
        cluster_assignments[str(K)] = cluster_labels.tolist()

        per_cluster = []
        for c in range(K):
            mask = cluster_labels == c
            n_in = int(mask.sum())
            if n_in < args.min_cluster_size:
                print(f"    cluster {c}: n={n_in} (skip, below min {args.min_cluster_size})")
                per_cluster.append({"cluster": c, "n_examples": n_in, "skipped": True})
                continue
            sub_spins = spins[mask]
            J_sub, _ = fit_ising_pseudolikelihood(sub_spins.numpy(),
                                                   extra_X=None, l2=args.l2)
            sub_clusters = spectral_cluster(J_sub, args.ks_spectral)
            best = {"k_spectral": None, "ari": -1, "nmi": 0, "purity": 0}
            for ks, labels in sub_clusters.items():
                if labels is None: continue
                m = score_against_gt(labels, args.mechinterp_json,
                                       n_layer, n_head)
                if m and m["ari"] > best["ari"]:
                    best = {"k_spectral": ks, **m}
            per_cluster.append({"cluster": c, "n_examples": n_in,
                                 "best": best})
            print(f"    cluster {c}: n={n_in} | "
                  f"best k_spectral={best['k_spectral']} "
                  f"ari={best['ari']:.3f} nmi={best['nmi']:.3f}")

        valid = [pc["best"]["ari"] for pc in per_cluster
                 if "best" in pc and pc["best"]["ari"] is not None]
        if valid:
            print(f"  mean within-cluster ARI: {np.mean(valid):.3f}")
            print(f"  max  within-cluster ARI: {np.max(valid):.3f}")
        strategy_A[str(K)] = {
            "per_cluster": per_cluster,
            "mean_within_cluster_ari": float(np.mean(valid)) if valid else None,
            "max_within_cluster_ari": float(np.max(valid)) if valid else None,
        }

    # --- STRATEGY B: pooled Ising with route-cluster one-hot covariates ---
    print("\n[STRATEGY B] pooled Ising with route-cluster covariates...")
    strategy_B = {}
    for K in args.ks_cluster:
        print(f"\n  K_route_clusters={K} (as one-hot covariates):")
        km = KMeans(n_clusters=K, random_state=0, n_init=10)
        cl = km.fit_predict(route_features)
        # one-hot encode, drop first column to avoid colinearity
        onehot = np.zeros((N, K - 1), dtype=np.float64)
        for k in range(1, K):
            onehot[cl == k, k - 1] = 1.0
        J_cov, _ = fit_ising_pseudolikelihood(spins.numpy(), extra_X=onehot,
                                                l2=args.l2)
        np.save(out_dir / f"J_cov_K{K}.npy", J_cov)
        cov_clusters = spectral_cluster(J_cov, args.ks_spectral)
        cov_metrics = {}
        for ks, labels in cov_clusters.items():
            if labels is None: continue
            m = score_against_gt(labels, args.mechinterp_json, n_layer, n_head)
            if m: cov_metrics[str(ks)] = m
            print(f"    k_spectral={ks}: purity={m['purity']:.3f} "
                  f"nmi={m['nmi']:.3f} ari={m['ari']:.3f}")
        strategy_B[str(K)] = cov_metrics

    # --- save everything ---
    json.dump(cluster_assignments, open(out_dir / "route_cluster_assignments.json", "w"))
    summary = {
        "model": args.model,
        "n_layer": n_layer, "n_head": n_head, "n_expert": n_expert,
        "n_examples": N,
        "marginal_baseline_ari": marg_metrics,
        "strategy_A_stratified": strategy_A,
        "strategy_B_route_covariate": strategy_B,
        "mean_routing_entropy_per_layer": mean_route_entropy_per_layer,
        "overall_mean_routing_entropy": overall_mean_entropy,
    }
    json.dump(summary, open(out_dir / "results.json", "w"), indent=2)
    print(f"\nsaved {out_dir / 'results.json'}")

    print("\n=== HEADLINE COMPARISON ===")
    marg_best = max((m["ari"] for m in marg_metrics.values()), default=-1)
    print(f"  marginal best ARI: {marg_best:.3f}")
    for K in args.ks_cluster:
        sA = strategy_A.get(str(K), {})
        if sA.get("max_within_cluster_ari") is not None:
            print(f"  Strategy A K={K}: max within-cluster ARI = "
                  f"{sA['max_within_cluster_ari']:.3f}, "
                  f"mean = {sA['mean_within_cluster_ari']:.3f}")
        sB = strategy_B.get(str(K), {})
        if sB:
            best_b = max((m["ari"] for m in sB.values()), default=-1)
            print(f"  Strategy B K={K}: best ARI with route-covariate = {best_b:.3f}")


if __name__ == "__main__":
    main()
