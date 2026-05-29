# Closure-Validated Circuit Discovery in Attention Heads

*Co-activation proposes, ablation disposes.* Closure tests in dense and MoE transformers at 1B scale.

📄 The full draft is in [`draft.pdf`](draft.pdf), but this README is meant to be
self-contained — everything below is the complete account.

---

## What this is

A growing line of interpretability work treats *groups* of components, not
individual units, as the basic object, and proposes to find those groups by
clustering co-activation statistics. We study one concrete instance:
**attention-head circuits** in 1B-scale language models. We adapt
Bhalla et al.'s (2026) SAE-feature Ising-clustering recipe to attention
heads, but our **object and validation differ** from that line of work — we
cluster **attention heads** (not SAE features) and validate by **causal
ablation** (not manifold reconstruction). Nothing here evaluates SAE-feature
manifold clustering; that is a complementary object with a complementary
validation criterion.

The question we ask of every cheap signal that gets read as evidence for a
circuit — co-activation cluster membership, attention-to-target selectivity,
participation ratio (PR) — is the same: **does it predict closure-validated
function?** The recurring answer is that a cheap signal is a circuit
*proposal*, not a confirmed circuit. **Closure** — ablate the candidate head
set, measure damage against matched-random controls — is what separates the
two.

We report two bodies of evidence: (1) closure on co-activation communities at
the final checkpoint, across three models and two input distributions; and
(2) closure across the training axis, which shows that two further proxies
(attention selectivity and PR) decouple from function in *both* directions.

---

## Result 1 — co-activation communities under closure

For each model we cluster attention heads by co-activation, pick the cleanest
community, ablate it, and compare per-example damage to five matched-random
head sets of the same size on the same batch.

| # | Model | Distribution | Cluster source | Heads | $z_{\Delta\ell}$ | $z_{\Delta\text{acc}}$ | $z_{\Delta z^{\text{tgt}}}$ | Verdict |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | OLMo 1B | synthetic | Ising | 5 | +1.83 | −1.97 | −2.68 | pass |
| 2 | Pythia 1B | synthetic | Ising, diffuse | 25 | +2.05 | −1.16 | −2.06 | weak pass |
| 3 | **OLMo 1B** | **natural** | Ising | 10 | **+36.4** | **−50.6** | **−50.9** | **pass** |
| 4 | **Pythia 1B** | **natural** | Ising | 9 | +1.95 | **−9.9** | **−52.5** | **pass (redundancy)** |
| 5 | **OLMoE 1B-7B** | natural | route-stratified Ising | 22 | **−5.64** | −0.22 | −2.97 | **does not pass** |

z-scores are the candidate's effect relative to the 5-control distribution,
in σ. **Pass** requires the candidate's damage in the predicted direction
($\Delta\ell > 0$ — loss rises under ablation; $\Delta\text{acc},
\Delta z^{\text{tgt}} < 0$) and outside the control distribution on at least
one metric.

**Per test:**

- **OLMo 1B synthetic (the cleanest case).** Spectral clustering at $k=10$
  yields a 5-head, layer-0 self-attention community, isolation ratio 5.8×,
  100% pure. Ablating it raises loss +1.85 nats — slightly *more* than
  ablating all 16 layer-0 heads (+1.78). Those five heads carry essentially
  all of layer-0's contribution: a minimal, load-bearing circuit recovered
  with no template specified.
- **Pythia 1B synthetic (diffuse).** A 25-head community over 7 layers
  (isolation 1.45×) that captures 10/13 previous-token heads but mixes in
  others. Closure passes in direction (+2.05σ) but the cluster is not minimal
  — 45% of the heads do 72% of the all-layer damage. A calibration anchor for
  what a less-clean proposal looks like under closure.
- **OLMo 1B natural (the strongest signal).** A 10-head L0+L1 community
  (isolation 3.01×). Ablation raises loss +1.44 nats while every random
  control *lowers* it by 0.07–0.18; the tight control distribution yields a
  +36σ z-score, with accuracy and target-logit both ≥50σ. Reading the
  *magnitude* (+1.44 vs the synthetic +1.85) it is comparable to the
  synthetic case — the 36σ reflects unusually tight controls, not a larger
  effect.
- **Pythia 1B natural (a redundancy signature).** A 9-head L0–L3 community.
  $\Delta\ell$ z is only +1.95σ (borderline) but $\Delta z^{\text{tgt}}$ z is
  −52.5σ (the largest single z-score in the study) and accuracy is −9.9σ.
  The community is load-bearing for the specific computation — the
  target-token logit drops 3.0 units — while the model's redundant downstream
  pathways keep cross-entropy roughly calibrated. See "multi-metric" below.
- **OLMoE 1B-7B natural (does not pass).** The marginal Ising collapses on
  natural text (ARI 0.006 vs 0.193 synthetic). Stratifying examples by
  per-layer routing pattern ($k$-means, $K=4$) and fitting Ising within each
  stratum recovers a within-stratum signal in cluster 1 (ARI 0.191, **+3.05σ
  above a random-partition null**, 0/10 random seeds reached it). But closure
  on that recovered 22-head community fails *in direction*: ablating it
  *improves* loss (−5.6σ on cluster-1 inputs, −12.9σ on the rest), more so
  outside the discovery stratum than inside — the opposite of a real
  route-conditional circuit. The candidate is route-modulated noise, not a
  load-bearing computation. **A cautionary case for co-activation
  interpretability of MoE attention, not a claim that MoE breaks
  interpretability.**

**Takeaway:** dense models pass closure across both distributions and both
architectures; the MoE route-conditional community recovers a real
*statistical* signal that does not survive closure. The MoE failure is
qualitative (wrong direction), so no choice of metric rescues it.

---

## Result 2 — closure across the training axis

We run the **same closure protocol** on cached intermediate checkpoints,
ablating the heads a model classifies into a capability class *at its final
checkpoint* and asking when that class becomes load-bearing. This lets us
compare two more cheap proxies — attention-to-target selectivity and PR —
against closure over training.

| Model / checkpoint | Class | Tokens | PR | Attn | $z_{\Delta\ell}$ | $z_{\Delta\text{acc}}$ | Reading |
|---|---|---:|---:|---:|---:|---:|---|
| Pythia step 1 (random init) | BOS | 0.002B | 2.1 | 1.3 | −1.1 | −0.8 | null (control) |
| Pythia step 256 | BOS | 0.5B | 10.5 | 1.0 | −1.8 | +0.9 | not load-bearing |
| **Pythia step 512** | **prev-tok** | 1.0B | **32** | **74** | **−5.4** (wrong) | +1.4 | **form without function** |
| Pythia step 1000 | BOS | 2B | 12.2 | 3.1 | +2.6 | −1.9 | load-bearing |
| Pythia step 3000 | BOS | 6B | 39.7 | 8.7 | +3.5 | −8.0 | load-bearing |
| **OLMo step 1000** | **BOS** | 2B | 11.4 | **0.6** | +0.4 | **−6.1** | **function without form** |
| OLMoE step 5000 | BOS | 20B | 54.5 | 4.6 | +2.4 | −8.4 | load-bearing |
| OLMoE step 25000 | BOS | 104B | 51.1 | 13.9 | +0.5 | −9.1 | load-bearing |
| OLMoE step 50000 | BOS | 209B | 42.4 | 19.7 | −0.3 | −4.9 | load-bearing |
| OLMoE step 100000 | BOS | 419B | 37.9 | 29.6 | −0.1 | −6.1 | load-bearing |
| OLMoE step 200000 | BOS | 838B | 31.7 | 187 | −0.1 | −5.8 | load-bearing |

("Attn" is mean selectivity to the class's canonical target; ≥30 is the
classification threshold. Accuracy z is the most reliable metric here — see
multi-metric below.)

**The two anchor rows point in opposite directions:**

- **Function without form** — at OLMo step 1000 (2B tokens) the BOS heads'
  attention-to-BOS selectivity is 0.6, ~50× below threshold (no attention
  pattern), yet ablating them costs −6.1σ in accuracy. Function is
  load-bearing *before* the attention pattern forms. Replicates in Pythia
  (step 1000) and OLMoE (step 5000).
- **Form without function** — at Pythia step 512 (1B tokens) the
  previous-token heads have selectivity 74 *and* PR 32, both well past any
  threshold one would use to declare the circuit present, yet ablation does
  *not* damage prediction (loss moves −5.4σ in the wrong direction).

The **bidirectional crossing** is the point: if attention selectivity were
merely a less-sensitive proxy for function we would see only one order
(closure firing before the cheap signal). Seeing both — function before the
pattern for BOS, the pattern before function for previous-token — means
attention pattern and function are *distinct constructs*, not the same
construct at different sensitivities. PR does not rescue the cheap-signal
reading either (high at Pythia step 256/512 while closure is null/negative).

**Supporting trajectories.** On the attention metric alone, per-class
emergence has the same ordering in all three models — previous-token and
self cross threshold first, BOS last — and BOS attention emerges ~25× later
(in tokens) in the Allen-AI models than in Pythia:

| Class crosses 30× | Pythia 1B | OLMo 1B | OLMoE 1B-7B |
|---|---|---|---|
| previous-token | ~0.5–1B | <2B (earliest ckpt) | <20B (earliest ckpt) |
| self | ~2–6B | 23–52B | 41–104B |
| first-token (BOS) | 10–20B | 264–597B | 419–838B |

But closure shows this is a fact about *attention-pattern sharpening*, not
function onset: BOS function is load-bearing by ~2B tokens in both Pythia
and OLMo regardless. In OLMoE, BOS function is load-bearing at every cached
checkpoint (from 20B on); the 6× sharpening of BOS selectivity between 419B
and 838B leaves closure damage unchanged — refinement of an
already-load-bearing head set, not function onset.

**What does not pin down:** whether ~2B tokens is a meaningful threshold
(two architectures agreeing is suggestive, not a law); why Pythia step 512
previous-token has high PR *and* selectivity yet no function (tokens vs class
vs architecture are confounded); whether function-before-form holds for
classes other than BOS (untested at intermediate checkpoints in OLMo/OLMoE).
Full trajectories and caveats: [`developmental_notes.md`](developmental_notes.md).

---

## Multi-metric closure (a reporting recommendation)

The three closure metrics rank results differently and each has a known
failure mode:

- **Cross-entropy loss** is the noisiest. It under-states real positives by
  aggregation slack (Pythia natural: +1.95σ on loss vs −52.5σ on
  target-logit) and over-states by control-variance collapse (OLMo natural:
  +36σ reflects tight controls as much as effect size).
- **Top-1 accuracy** is more stable but saturates in low-accuracy regimes.
- **Mean target-token logit** is the most stable across our tests — it tracks
  confidence in the correct token before the softmax, free of both the
  aggregation slack and the saturation issue.

**Recommendation:** report all three; read **direction first**, then
**target-logit and accuracy** as primary indicators, then **loss** as a
conservative floor. Under this reading the verdicts are internally
consistent and the one "does not pass" (MoE) stands out because it fails on
*direction*, which no metric choice rescues.

---

## Figures

![Five-test verdict](figures/fig1_verdict.png)
*Multi-metric verdict across the five community-closure tests. Pass tests
have the candidate bar in the damage direction (above 0 on Δloss; below 0 on
Δaccuracy and Δtarget-logit); the MoE result flips direction.*

![Pythia redundancy](figures/fig2_redundancy.png)
*Pythia 1B natural-text redundancy signature: near-zero Δloss but large
negative Δtarget-logit — ~18× more downstream reconstruction than OLMo on the
natural batch. The ratio |Δlogit|/|Δloss| is the signature.*

![MoE story arc](figures/fig3_moe_story.png)
*MoE arc: (a) marginal Ising collapses on OLMoE natural text; (b)
route-stratification recovers a cluster-1 signal above the random-partition
null band; (c) closure on that community fails in direction — ablation helps
loss on both subsets.*

---

## Method (full protocol)

**Forward + signal.** For each model and batch, forward with
`output_attentions=True` and extract the attention pattern at a per-example
query position. For each head $(L,H)$ take the maximum attention weight to
any key position as a template-free *focus* signal (no canonical target
chosen). For MoE, also extract per-layer softmax routing weights at the query
position.

**Binarize.** Threshold each head's signal across the batch at the head's own
median → spins $s_{i,(L,H)} \in \{-1,+1\}$. Every head fires 50% of the time
by construction, so cross-head co-activation is the only signal driving
couplings. (This is the attention-head analog of Bhalla et al.'s sign-of-code
binarization, but with no always-on background features.)

**Ising fit.** Fit a pairwise Ising model on the $N\times F$ spin matrix by
per-spin $L_2$-regularized logistic regression (pseudolikelihood, $\lambda =
10^{-3}$); symmetrize to get coupling matrix $J$.

**Community recovery.** Spectral-cluster $|J|$ for $k\in\{4,6,8,10,12\}$;
pick the $k$ maximizing adjusted Rand index against a prior probe-circuit
supervised classification (used only for hyperparameter selection and
candidate identification, never in the clustering itself). Choose the
candidate community by `purity × isolation × size-clip`.

**Closure test.** Ablate the candidate by zeroing its per-head slices of the
attention output projection via a forward-pre-hook. Record per-example loss,
top-1 accuracy, and target-token logit at the query position. Repeat with 5
matched-random head sets of the same size drawn from heads not in the
candidate. Report the candidate's Δ on each metric as a z-score against the
control distribution, plus $P[\text{control} \ge \text{candidate}]$.

**MoE extension.** When the marginal Ising collapses, stratify examples by
$k$-means ($K=4$) on the flattened per-layer routing weights and fit a
separate Ising within each route stratum. Verify any recovered signal is
route-specific (not a sample-size artifact) with a **random-partition null**:
10 uniform $K=4$ partitions of the same batch, refit per group, compare max
within-group ARI.

**Developmental.** Run the same closure protocol on end-state-classified head
sets at cached intermediate checkpoints; alongside, track attention-to-target
selectivity and the participation ratio of the per-head output across
training.

---

## Claims

1. Co-activation clustering of attention-head focus statistics can propose
   communities that survive causal closure as load-bearing circuits in the
   two dense 1B-scale models tested (4/4 dense closure tests pass; OLMo
   synthetic cluster 2 saturates the ablate-all-of-layer-0 upper bound).
2. Recovered communities are distribution-conditioned but not merely
   synthetic artifacts — natural-text dense communities pass closure with
   target-logit/accuracy effect sizes comparable to or exceeding synthetic.
3. Route-conditional clustering in MoE recovers a statistical signal (+3.05σ
   above a careful null) that does not pass closure: ablation helps loss, in
   the wrong direction.
4. Across the training axis, attention selectivity and PR decouple from
   closure-validated function in both directions (function-without-form,
   form-without-function) — they are distinct constructs.
5. Therefore a cheap signal — cluster membership, attention selectivity, or
   PR — is a circuit *proposal*, not a confirmed circuit. Closure is what
   confirms.

**We do not claim:** anything about co-activation clustering of SAE-feature
manifolds (a different object, validated by reconstruction, not tested here);
that the method "works for dense models" in general (two architectures, two
distributions); that MoE "breaks interpretability" (one MoE model, one
natural-text distribution, one route-conditioning protocol); or that ~2B
tokens is a universal function-emergence threshold.

---

## Limitations

Two dense models (OLMo 1B, Pythia 1B) and one MoE (OLMoE-1B-7B), two input
distributions (synthetic induction, Pile-derived natural text). The
dense-vs-MoE closure asymmetry is unambiguous within this sample but cross-MoE
replication (Mixtral, DeepSeek-MoE) is needed before treating it as general.
Per-head ablation is destructive; mean-ablation / activation-patching /
counterfactual ablation may give different effect sizes. The developmental
results use sparse checkpoints; the ~2B-token observation is suggestive, not a
law. The cluster-discovery step uses pairwise Ising — mutual information on
binary spins gives ~2× stronger ARI on the dense models, but closure results
depend on the clusters, not the affinity used to find them.

---

## Reproducing

```bash
pip install -r requirements.txt

# --- Final-checkpoint closure (Result 1) ---
python pipeline/ising_circuit_discovery.py \
    --model EleutherAI/pythia-1b --tag pythia_1b \
    --mechinterp-json <supervised-classification-json> --out-dir results
python pipeline/ising_circuit_discovery_naturaltext.py --model EleutherAI/pythia-1b ...
python pipeline/route_conditioned_ising_olmoe.py        # MoE route-conditional
python pipeline/random_partition_null.py                # null control (CPU, ~1 min)
python closure/select_closure_candidate_pythia_natural.py
python closure/closure_pythia_natural.py

# --- Developmental closure (Result 2) ---
python developmental/developmental_scan.py --model EleutherAI/pythia-1b --revisions step1,step256,... ...
python developmental/closure_at_intermediate.py --model EleutherAI/pythia-1b --revision step1000 --target-class first-token ...
python developmental/combined_pr_attn_emergence.py

python figures/generate_figures.py
```

Required: PyTorch with MPS or CUDA (forward passes on 1B-scale models),
HuggingFace `transformers`, scikit-learn, numpy, matplotlib.

## Repository layout

```
draft.pdf / draft.tex   the working draft (compile with figures/ at repo root)
developmental_notes.md  companion notes for the training-axis results
pipeline/               Ising discovery, route conditioning, null control
closure/                candidate selection + closure tests per (model, distribution)
developmental/          checkpoint scans, per-class emergence, intermediate closures
analysis/               supplementary diagnostics (per-class recall, viz)
figures/                figure generation script + PDFs/PNGs
results/                JSON outputs + small .npy artifacts
```

Large tensors are not tracked and regenerate from the pipeline: per-test
attention tensors (`attn_at_query.npy`, 250–500MB each) and OLMoE routing
weights (`*_routes.npy`, ~8MB each). Tracked: Ising coupling matrices
`J.npy` (≤512KB) and all JSON outputs (closure results, candidate metadata,
null distributions, developmental trajectories).

## License

MIT (see `LICENSE`).
