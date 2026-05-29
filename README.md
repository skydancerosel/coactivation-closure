# Closure-Validated Circuit Discovery in Attention Heads

**Co-activation proposes, ablation disposes — closure tests in dense and MoE transformers at 1B scale.**

📄 **Draft:** [`draft.pdf`](draft.pdf) (working draft) · companion notes: [`developmental_notes.md`](developmental_notes.md)

We study **attention-head** circuits in 1B-scale language models and ask a
single question of every cheap signal that gets used as evidence for a
circuit: *does it predict closure-validated function?* We adapt
Bhalla et al.'s (2026) SAE-feature Ising-clustering recipe to attention
heads, but our object and validation differ from that line of work — we
cluster **attention heads** (not SAE features) and validate by **causal
ablation** (not manifold reconstruction). This repository is *not* an
evaluation of SAE-feature manifold clustering, which is a complementary
object with a complementary validation criterion.

The recurring answer: a cheap signal — co-activation cluster membership,
attention-to-target selectivity, or participation ratio — is a circuit
**proposal**, not a confirmed circuit. Closure (ablate-and-measure) is what
separates the two.

## Headline result: co-activation communities under closure

| # | Model | Distribution | Cluster source | Heads | $z_{\Delta\ell}$ | $z_{\Delta\text{acc}}$ | $z_{\Delta z^{\text{tgt}}}$ | Verdict |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | OLMo 1B | synthetic | Ising | 5 | +1.83σ | −1.97 | −2.68 | pass |
| 2 | Pythia 1B | synthetic | Ising, diffuse | 25 | +2.05σ | −1.16 | −2.06 | weak pass |
| 3 | **OLMo 1B** | **natural** | Ising | 10 | **+36.4σ** | **−50.6** | **−50.9** | **pass** |
| 4 | **Pythia 1B** | **natural** | Ising | 9 | +1.95σ | **−9.9** | **−52.5** | **pass (redundancy)** |
| 5 | **OLMoE 1B-7B** | natural | route-stratified Ising | 22 | **−5.64σ** | −0.22 | −2.97 | **does not pass (wrong dir.)** |

Each row: a candidate community discovered unsupervisedly, ablated, and
compared against 5 matched-random-head controls on the same batch. Pass
requires $\Delta\ell > 0$ (loss rises under ablation) with
$\Delta\text{acc}, \Delta z^{\text{tgt}} < 0$, at least one metric well
outside the control distribution.

**Dense models pass closure across both synthetic and natural
distributions.** On MoE, the marginal Ising collapses on natural text
(ARI 0.006); route-conditional stratification recovers a statistical signal
(+3.05σ above a random-partition null) but that community **does not pass
closure** — ablating it *helps* loss, in the opposite direction from a real
circuit on the discovery stratum. The MoE outcome is a cautionary case for
applying co-activation clustering to MoE attention, not a claim that "MoE
breaks interpretability."

## Second result: closure across the training axis

Using cached intermediate checkpoints (Pythia 1B, OLMo 1B, OLMoE), we ask
the same question of two more proxies — attention-to-target selectivity and
participation ratio (PR) — over the course of training. They do not track
closure-validated function, and they fail in **both directions**:

- **Function without form.** At OLMo 1B step 1000 (2B tokens), BOS heads'
  attention-to-BOS selectivity is 0.6 (50× below threshold) yet ablating
  them costs −6.1σ in accuracy. Function is load-bearing before the
  attention pattern forms. Replicates in Pythia (step 1000) and OLMoE
  (step 5000).
- **Form without function.** At Pythia 1B step 512 (1B tokens),
  previous-token heads have selectivity 74 *and* PR 32 — both well past any
  threshold — yet ablation does not damage prediction (loss moves −5.4σ in
  the *wrong* direction).

The bidirectional crossing shows attention pattern and function are distinct
constructs, not the same construct at different sensitivities. Details and
honest caveats in [`developmental_notes.md`](developmental_notes.md) and §7
of the draft.

## Four claims (from the paper)

1. **Co-activation clustering of attention-head focus statistics can
   propose communities that survive causal closure as load-bearing
   circuits in the two dense 1B-scale models tested.** 4 of 4 dense
   closure tests pass; OLMo 1B synthetic cluster 2 (5 heads, layer-0
   self-attention) saturates the ablate-all-of-layer-0 upper bound.
2. **Recovered communities are distribution-conditioned but not merely
   synthetic artifacts.** Natural-text dense communities pass closure with
   target-logit and accuracy effect sizes comparable to or exceeding the
   synthetic cases.
3. **Route-conditional co-activation clustering in MoE recovers a
   statistical signal that does not pass closure.** OLMoE natural-text
   marginal Ising collapses (ARI 0.006); route-stratification recovers
   ARI 0.191 (+3.05σ above a random-partition null, 0/10 random seeds
   reached it), but the recovered community fails closure in direction.
4. **A cheap signal — cluster membership, attention selectivity, or PR —
   is a circuit *proposal*, not a confirmed circuit.** Closure is what
   confirms; it is necessary in the dense models here and not yet
   sufficient in the MoE setting even after route conditioning. *(We make
   no claim about co-activation clustering of SAE-feature manifolds, a
   different object validated differently and not tested here.)*

## A methodological observation worth flagging

The three closure-effect metrics (cross-entropy loss, top-1 accuracy, mean
target-token logit) rank results differently and each has a known failure
mode. The **Pythia 1B natural-text test shows a 25× divergence within the
same test on the same ablation**: $z_{\Delta\ell} = +1.95\sigma$
(borderline) but $z_{\Delta z^{\text{tgt}}} = -52.5\sigma$ (largest single
z-score in the study) — a *downstream-redundancy signature*. We recommend
**multi-metric closure reporting**: direction first, then target-logit and
accuracy as primary indicators, then loss as a conservative floor.

## Figures

![Five-test verdict](figures/fig1_verdict.png)
*Figure: multi-metric verdict across the five community-closure tests. Pass
tests have the candidate bar in the damage direction (above 0 on
$\Delta$loss; below 0 on $\Delta$accuracy and $\Delta$target-logit); the MoE
result flips direction.*

![Pythia redundancy](figures/fig2_redundancy.png)
*Figure: Pythia 1B natural-text redundancy signature — near-zero
$\Delta$loss but large negative $\Delta$target-logit, ~18× more downstream
reconstruction than OLMo on the natural batch.*

![MoE story arc](figures/fig3_moe_story.png)
*Figure: MoE arc — (a) marginal Ising collapses on OLMoE natural text;
(b) route-stratification recovers a cluster-1 signal above the
random-partition null band; (c) closure on that community fails in
direction.*

## Method (one paragraph)

For each model and batch, forward with `output_attentions=True` and extract
the attention pattern at a per-example query position. For each head
$(L, H)$, take the maximum attention weight to any key position as a
template-free *focus* signal, then binarize across the batch at the head's
own median. Fit a pairwise Ising model on the $N \times F$ spin matrix by
per-spin $L_2$-regularized logistic regression (pseudolikelihood);
symmetrize to get $J$. Spectral-cluster $|J|$ for $k \in \{4,6,8,10,12\}$
and select $k$ by ARI against the prior probe-circuit supervised
classification. Pick the candidate by `purity × isolation × size-clip`.
Ablate via forward-pre-hook on the attention output projection; compare
$\Delta\ell$, $\Delta\text{acc}$, $\Delta z^{\text{tgt}}$ against 5
matched-random-head controls. For MoE, additionally stratify examples by
k-means on per-layer routing weights ($K=4$) and fit Ising per stratum;
verify route-specificity with a random-partition null (10 uniform $K=4$
partitions). For the developmental results, run the same closure protocol
on end-state-classified head sets at cached intermediate checkpoints.

## Reproducing

```bash
pip install -r requirements.txt

# --- Final-checkpoint closure (the headline table) ---
# 1. Discovery pipeline (synthetic batch, ~5 min on MPS per model)
python pipeline/ising_circuit_discovery.py \
    --model EleutherAI/pythia-1b --tag pythia_1b \
    --mechinterp-json <supervised-classification-json> --out-dir results
# 2. Natural-text variant
python pipeline/ising_circuit_discovery_naturaltext.py --model EleutherAI/pythia-1b ...
# 3. Route-conditional Ising for MoE
python pipeline/route_conditioned_ising_olmoe.py
# 4. Random-partition null (CPU, ~1 min)
python pipeline/random_partition_null.py
# 5-6. Candidate selection + closure
python closure/select_closure_candidate_pythia_natural.py
python closure/closure_pythia_natural.py

# --- Developmental closure (across training checkpoints) ---
python developmental/developmental_scan.py --model EleutherAI/pythia-1b --revisions step1,step256,... ...
python developmental/closure_at_intermediate.py --model EleutherAI/pythia-1b --revision step1000 --target-class first-token ...
python developmental/combined_pr_attn_emergence.py

# --- Figures ---
python figures/generate_figures.py
```

Required: PyTorch with MPS or CUDA, HuggingFace `transformers`, sklearn,
numpy, matplotlib.

## Repository layout

```
draft.pdf / draft.tex   - the working draft (compile with figures/ at repo root)
developmental_notes.md  - companion notes for the training-axis results
pipeline/               - discovery: Ising fits, route conditioning, null control
closure/                - candidate selection + closure tests per (model, distribution)
developmental/          - checkpoint scans, per-class emergence, intermediate closures
analysis/               - supplementary diagnostics (per-class recall, viz)
figures/                - figure generation script + PDFs/PNGs
results/                - JSON outputs + small .npy artifacts
```

Large tensors are not tracked and regenerate by re-running the pipeline:
per-test attention tensors (`attn_at_query.npy`, 250-500MB each) and OLMoE
routing weights (`*_routes.npy`, ~8MB each). Tracked: Ising coupling
matrices `J.npy` (<=512KB), and all JSON outputs (closure results, candidate
metadata, null distributions, developmental trajectories).

## Limitations

Two dense models (OLMo 1B, Pythia 1B) and one MoE (OLMoE-1B-7B), two input
distributions (synthetic induction, Pile-derived natural text). The
dense-vs-MoE closure asymmetry is unambiguous within this sample but
cross-MoE replication (Mixtral, DeepSeek-MoE) is needed before treating it
as general. Per-head ablation is destructive; alternative interventions may
give different effect sizes. The developmental results are on a few
architectures with sparse checkpoints; the "~2B tokens to BOS function"
observation is suggestive, not a law. The cluster-discovery step uses
pairwise Ising; mutual information on binary spins gives ~2x stronger ARI on
the dense models, but closure results depend on the clusters, not the
affinity. We make no claim about SAE-feature manifold clustering.

## License

MIT (see `LICENSE`).
