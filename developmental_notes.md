# Developmental notes: attention-head circuit emergence across 1B-scale LMs

*Companion to the closure paper.*
**Date:** 2026-05-23
**Status:** Wrapped. ~10 closure tests, 4 per-checkpoint signals across 3
architectures, robustness checked on 2 batches. Reported honestly with
caveats.

## TL;DR

We applied four developmental signals to three 1B-scale language models
(Pythia 1B, OLMo 1B, OLMoE-1B-7B) across their cached training
checkpoints, asking when attention-head circuits actually become
load-bearing during training. Three findings emerge with varying
strength:

1. **Attention pattern and load-bearing function are independent
   constructs — they decouple in both directions.** Across 10 closure
   tests, we observe both *function-without-form* (OLMo and Pythia
   BOS heads are load-bearing at 2B tokens despite attention selectivity
   essentially absent) and *form-without-function* (Pythia step 512
   prev-token heads have PR=32 and attention=74 both well past their
   thresholds, yet closure goes in the wrong direction — z=−5.4σ on
   loss). The bidirectional crossing rules out "the functional metric
   is just more sensitive than the pattern metric"; pattern and
   function are genuinely distinct constructs measured by these tools.
   *Scope*: 9 of 10 tests are on BOS heads; the form-without-function
   datum is on prev-token. The bidirectional claim is from these 10
   tests combined, with most of the data on BOS.
2. **The OLMoE attention phase transition (419B → 838B tokens) does not
   correspond to functional emergence** — closure damage on BOS heads
   is essentially identical pre- and post-jump. The phase transition
   is attention-pattern refinement of already-load-bearing heads.
3. **The same phase-transition window coincides with a feature in
   OLMoE's natural-batch loss trajectory** — the only interval in the
   trajectory where Δloss is non-positive, replicated on two
   independently-sampled batches. Robust within OLMoE; doesn't
   generalize (no comparable attention phase transitions in Pythia or
   OLMo to test).

Combined, this extends the closure paper's "attention metrics don't
predict load-bearing function" lesson along the time axis: at every
training stage tested, neither attention-pattern selectivity nor PR
alone reliably predicts when a head set has wired in.

## What we measured

Three checkpointed 1B-scale language models from publicly released
training runs:

- **Pythia 1B** (EleutherAI), 14 checkpoints from step 1 to step 143000.
- **OLMo 1B** (Allen AI), 10 checkpoints from 2B to 3048B tokens.
- **OLMoE-1B-7B** (Allen AI), 10 checkpoints from 20B to 5117B tokens.

For each cached checkpoint, four signals per attention head:

1. **Attention-to-canonical-target selectivity** — for each end-state
   "first-token / previous-token / self / induction" head, the mean
   attention to its canonical target position, normalized against a
   random-position baseline. Same protocol as the probe-circuit
   methodology paper.
2. **Participation Ratio (PR) of per-head attention output** — measures
   content-dependent computation across the batch (from the
   spectral-probe-circuits work).
3. **Closure damage** — at 10 selected intermediate checkpoints, ablate
   the end-state-classified heads of a class, compare per-example loss /
   accuracy / target-logit damage against five matched-random head
   ablations. Same protocol as the closure paper.
4. **Ising co-activation cluster membership** — spectral cluster of the
   pairwise Ising fit on binarized per-head focus, at each checkpoint.

Four signals on three architectures with 10-14 checkpoints each. The
honest framing throughout: we have a handful of architectures and a
handful of metrics, none of which fully predicts the others.

## Per-class attention-pattern emergence

Tracking *end-state-classified heads*' mean selectivity to their
canonical target across training:

| Class | Pythia 1B | OLMo 1B | OLMoE 1B-7B |
|---|---|---|---|
| previous-token | crosses 30× at ~500M–1B tokens | already crossed at earliest ckpt (2B) | already crossed at earliest ckpt (20B) |
| self | ~2–6B | 23–52B | 41–104B |
| first-token (BOS) | **10–20B** | **264–597B** | **419–838B** |

Three observations:

1. **Same per-class ordering in all 3 architectures.** prev-token (and
   self) emerge before BOS on the attention metric.
2. **BOS attention emerges ~25× slower (in tokens) in OLMo/OLMoE than in
   Pythia.** Real observation on the attention metric.
3. **Induction heads don't appear in the natural-text end-state
   classification for any of the 3 models** — only synthetic-induction
   batches surface induction heads. (Synthetic-batch data:
   `results/dev_per_class_synth/`. Induction phase transitions: ~1–2B
   Pythia, ~4–10B OLMo, <20B OLMoE — Olsson-style emergence replicates
   at 1B scale across all 3 architectures.)

The 25× cross-architecture delay on attention disappears when we look
at function instead (next two sections). The static interp literature,
working only with the final checkpoint, can see the per-class ordering
but not the cross-architecture variance or the function-vs-pattern
distinction.

## PR vs attention-pattern emergence: they decouple

Combining the existing PR trajectory data (from
`nostalgic-lederberg-80a58d/*_phase1_trajectory.json`) with our
attention-to-target data, per end-state-classified head:

| Model | Class | n heads | PR crosses 5 (tokens) | Attn crosses 30× (tokens) | Gap |
|---|---|---:|---:|---:|---:|
| Pythia 1B | first-token | 19 | 0.5B | 20B | **40×** |
| Pythia 1B | previous-token | 14 | 0.5B | 1.0B | 2× |
| OLMo 1B | first-token | 21 | ≤2B (earliest) | 597B | **≥300×** |
| OLMo 1B | previous-token | 7 | ≤2B | 2B | ~1× |
| OLMo 1B | self | 5 | ≤2B | 52B | ~26× |
| OLMoE | first-token | 28 | ≤20B (earliest) | 838B | **≥40×** |
| OLMoE | previous-token | 5 | ≤20B | 20B | ~1× |
| OLMoE | self | 6 | ≤20B | 20B | ~1× |

The PR threshold (5) is arbitrary; the qualitative point is robust to
threshold choice. **For BOS heads in all 3 architectures, PR crosses
any reasonable threshold dramatically before the attention pattern
crosses its 30× selectivity threshold.** For prev-token and self the
two signals are more synchronous.

Caveat: for OLMo and OLMoE the earliest checkpoint already has PR past
threshold, so PR-emergence time is upper-bounded by the earliest
cached checkpoint. For Pythia we have step 1, 4, 16, 64, 256 and PR
for BOS heads is {2.09, 2.09, 1.21, 2.02, 10.48} — PR emerges between
step 64 and step 256 in Pythia.

## Closure at 10 intermediate checkpoints

The actual test of function emergence. For each checkpoint, ablate the
end-state-classified heads of one class, compare per-example damage to
five matched-random head sets.

| Test | Tokens | PR | Attn | Δloss z | Δacc z | Δlogit z | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Pythia step 1 BOS (random init, neg. control) | 0.002B | 2.1 | 1.3 | −1.11 | −0.82 | +1.03 | NULL (correct — no function yet) |
| Pythia step 256 BOS | 0.5B | 10.5 | 1.0 | −1.75 | +0.85 | +1.01 | NOT load-bearing |
| Pythia step 512 prev-token | 1.0B | 32.4 | 74 | **−5.39 (wrong dir)** | +1.43 | −1.45 | NOT load-bearing |
| **Pythia step 1000 BOS** | **2B** | 12.2 | 3.1 | **+2.61** | **−1.90** | **−2.27** | **PASS** (all 3) |
| Pythia step 3000 BOS | 6B | 39.7 | 8.7 | **+3.45** | **−7.95** | **−2.86** | PASS (all 3) |
| **OLMo step 1000 BOS** | **2B** | 11.4 | 0.6 | +0.37 | **−6.10** | +0.13 | **PASS** (acc) |
| **OLMoE step 5000 BOS** | **20B** | 54.5 | 4.6 (max 17) | **+2.42** | **−8.40** | **−6.40** | **PASS** (all 3) |
| OLMoE step 25000 BOS | 104B | 51.1 | 13.9 | +0.49 | **−9.14** | **−5.66** | PASS (acc + logit) |
| OLMoE step 50000 BOS | 209B | 42.4 | 19.7 | −0.34 | **−4.85** | **−4.72** | PASS (acc + logit) |
| OLMoE step 100000 BOS (pre-jump) | 419B | 37.9 | 29.6 | −0.11 | **−6.07** | **−3.11** | PASS (acc + logit) |
| OLMoE step 200000 BOS (post-jump) | 838B | 31.7 | 187 | −0.07 | **−5.83** | −1.97 | PASS (acc) |

### What's clear from these 11 closures

1. **Random init has no functional structure** (Pythia step 1 negative
   control: all metrics near zero).
2. **BOS function is load-bearing at 2B tokens in Pythia AND OLMo**
   (Pythia step 1000 + OLMo step 1000 both pass) — function emerges by
   2B tokens regardless of whether the attention pattern is formed
   (OLMo BOS attn = 0.6, Pythia BOS attn = 3.1 — both far below 30×).
   **BOS-function-before-BOS-form replicates across 2 dense
   architectures at the same training-token count.** (Other classes
   not tested at intermediate checkpoints in OLMo or OLMoE; the
   replication is specifically for BOS.)
3. **In OLMoE, BOS function is load-bearing at the earliest cached
   checkpoint** (20B), and at every subsequent checkpoint tested. We
   can't resolve the OLMoE BOS function emergence point earlier than
   20B (no earlier checkpoints).
4. **The OLMoE attention phase transition between 419B and 838B is
   not function emergence.** Closure damage is essentially identical
   pre-jump (step 100k: Δacc z=−6.1, Δlogit z=−3.1) and post-jump
   (step 200k: Δacc z=−5.8, Δlogit z=−2.0). The 6× sharpening of
   attention selectivity (29.6 → 187) is attention-pattern refinement,
   not function onset.
5. **At Pythia step 256 (PR=10) and step 512 (PR=32 + attn=74), heads
   are not load-bearing.** PR and attention can be substantial without
   closure damage being in the load-bearing direction. The
   developmental signals don't cleanly predict closure.

### What's not clear

- **Whether 2B tokens is a universal threshold or coincidence.** Two
  Allen-AI / EleutherAI architectures hit BOS function-emergence around
  step 1000 = 2B tokens, but n=2 architectures is not "universal."
- **Why Pythia step 512 prev-token isn't load-bearing despite PR=32
  and attn=74.** Both metrics are well past the thresholds that OLMoE
  step 100000 (PR=38, attn=30) cleared while being load-bearing. We
  don't have a single-factor explanation — training tokens (1B vs 419B)
  is the obvious candidate but unverified.
- **Whether the "function before form" finding extends to other classes
  or other architectures.** All 10 closure tests are on first-token
  (BOS) heads except one (Pythia step 512 prev-token, which failed).
  We did not test prev-token or self closures at intermediate
  checkpoints in OLMo or OLMoE.

### Multi-metric closure observation

Consistent with the closure paper: the **loss z-score is the noisiest
metric**. Several closure-passes had null Δloss z but strong Δacc /
Δlogit z (OLMo step 1k, OLMoE step 25k/50k/100k/200k). One closure
case showed strong loss + accuracy + logit all in damage direction
(Pythia step 1k, step 3k, OLMoE step 5k). One showed wrong-direction
loss (Pythia step 512 prev-token: z=−5σ on loss but +1.4σ on acc).

When summarizing closure across many tests, target-logit and accuracy
are more reliable than loss. Loss can swing both ways depending on
control variance and downstream redundancy.

## OLMoE loss-curve alignment with the attention phase transition

The strongest of the developmental findings. We documented an attention
phase transition between OLMoE step 100k (419B tokens) and step 200k
(838B): mean BOS selectivity for end-state BOS heads jumped from 29.6
to 187 (6× sharpening). We checked whether the natural-batch LM loss
trajectory shows any feature in the same window.

### Loss trajectory across 10 cached OLMoE checkpoints

Forward pass at each checkpoint on the natural batch (2000 examples at
per-example query positions), recording mean loss / acc / target-logit:

| Interval | Δloss |
|---|---:|
| 20B → 41B | +0.206 (climbing) |
| 41B → 104B | +0.142 |
| 104B → 209B | +0.099 |
| 209B → 419B | +0.019 (decelerating) |
| **419B → 838B** | **−0.004 (flat)** ← phase-transition window |
| 838B → 1677B | +0.053 (climbing again) |
| 1677B → 2516B | +0.075 |
| 2516B → 3355B | +0.041 |
| 3355B → 5117B | +0.099 |

**Out of 9 inter-checkpoint intervals, only the 419B → 838B window has
non-positive Δloss** — exactly the documented phase-transition window.
Loss decelerates approaching the window, flat through it, reaccelerates
after.

Target-token logit hits its global minimum (8.92) at 838B, the window's
right boundary, then climbs back to 10.07 by end-of-training. Accuracy
stays roughly constant throughout (no feature).

### Robustness check on a second batch

We ran the same scan on `natural_induction_batch_midseq.pt` (different
query-position distribution: starts at 41 not 21; different
first_T_pos distribution: starts at 20 not 0). Baseline losses are
~+0.6 to +0.8 nats higher (more difficult batch), but the trajectory
structure replicates:

| Interval | Regular Δloss | Midseq Δloss |
|---|---:|---:|
| 20→41 | +0.21 | +0.20 |
| 41→104 | +0.14 | +0.20 |
| 104→209 | +0.10 | +0.08 |
| 209→419 | +0.02 | +0.06 |
| **419→838** | **−0.004** | **−0.018** |
| 838→1677 | +0.053 | +0.095 |
| 1677→2516 | +0.075 | +0.036 |
| 2516→3355 | +0.041 | +0.052 |
| 3355→5117 | +0.099 | +0.096 |

**The 419 → 838B interval is the only negative-Δloss interval on both
batches.** Target-logit minimum also lands at 838B on both batches
(regular: 8.92, midseq: 7.89). Under a null where the flat-Δloss
interval lands at random across 9 intervals (~11% per batch),
replication on a second independent batch with very different baseline
trajectories drops the joint probability to ~1%.

### Reading the alignment

**What the alignment is**: a within-OLMoE empirical observation that the
attention-pattern phase transition window coincides with a feature in
the natural-batch loss trajectory, robust across two independently-
sampled batches.

**What the alignment is not**:
- *Not predictive science.* One architecture, one phase transition. We
  cannot claim that mechanism events generally have loss-curve
  signatures.
- *Not causation.* The phase transition might cause the loss feature,
  both might result from a third event (data-mixture shift,
  LR-schedule event, expert-routing specialization point), or the
  alignment might still be coincidence at the within-batch level.
- *Not generalizable across architectures.* Pythia and OLMo don't show
  comparable attention phase transitions in our data, so the
  cross-architecture replication question is not testable from what we
  have.

Three possible mechanisms for the alignment:
1. Attention sharpening reorganizes downstream computation in a way
   that briefly arrests whatever process is making natural-batch loss
   climb at this resolution.
2. A third training event (data shift, LR schedule, expert
   specialization milestone) simultaneously triggers both the
   attention sharpening and the loss feature.
3. Coincidence at the within-OLMoE level, even after batch replication.

We can't distinguish these without access to Allen AI's training
schedule or interventional experiments. The within-OLMoE finding
stands; the mechanistic story is open.

The most relevant connection to the closure paper: the OLMoE
loss-trajectory feature is most cleanly visible on target-logit, not
loss — the same metric that the closure paper recommended as primary
for closure tests. The pattern that aggregate cross-entropy obscures
finer signals shows up at both the per-test and the trajectory level.

## Continuous drift in co-activation clusters

From the earlier developmental Ising scans (not closure-based): the
spectral cluster assignments of head co-activation patterns are
unstable across training in all 3 architectures. Adjacent-checkpoint
NMI@k=6 is about 0.25 (Pythia), 0.16 (OLMo), 0.15 (OLMoE). The
asymptotic floor (distant checkpoints) is 0.06-0.10. **Communities
never stabilize during training.**

Caveat: NMI = 0.25 for adjacent checkpoints is not "near 1.0" stable
but also not "near 0" random — the metric sits in the middle of its
range and the cluster relabeling isn't well characterized. The
observation is qualitatively clean but quantitatively coarse.

## What this writeup is and is not

**Is**: a developmental companion to the closure paper. The closure
paper documented at end-state that attention metrics and load-bearing
function can diverge (the MoE failure). This work extends the same
observation across the training axis with a stronger form of the
divergence: attention pattern and closure-validated function decouple
in *both* directions across training — function arrives before
attention pattern for BOS heads in 2 dense architectures, and
attention pattern arrives before function for prev-token heads at
Pythia step 512. The bidirectional crossing is what makes the
"distinct constructs" reading necessary, beyond what a unidirectional
"function before form" claim would support. The OLMoE phase
transition is also documented as loss-silent at end-state in the
closure paper but coincides with a loss-trajectory feature here.

**Is not**: a standalone paper with a clean architectural contrast.
Three architectures × sparse checkpoints × multiple metrics that don't
cleanly align is not enough to support the kind of headline claim the
closure paper carries. The phase-transition / loss-curve alignment is
the cleanest single finding here and it's still one architecture × one
phase transition × two batches.

### Defensible claims

- **Attention pattern and closure-validated function decouple in both
  directions across training.** BOS-function-without-BOS-form
  (Pythia step 1000 + OLMo step 1000 + OLMoE step 5000, all
  load-bearing with attn far below threshold) and form-without-function
  (Pythia step 512 prev-token, PR=32 + attn=74 both well past
  thresholds, closure z=−5σ wrong direction on loss). The
  bidirectional crossing rules out "one metric is just more sensitive."
- BOS function is load-bearing by ~2B tokens in both Pythia and OLMo,
  before attention pattern is formed. Replicated across 2 dense
  architectures at the same token count. (Not tested for other
  classes at intermediate checkpoints in OLMo or OLMoE — this is
  specifically a BOS-class observation.)
- OLMoE BOS function is load-bearing at every cached checkpoint we
  tested (20B onwards). The 419B-838B attention phase transition is
  attention-pattern refinement, not function onset.
- The OLMoE attention phase transition window coincides with the only
  flat-Δloss interval in the natural-batch loss trajectory, replicated
  on two independently-sampled batches.
- Target-token logit is the most informative single metric for
  closure-style measurements, both per-test and per-trajectory.

### Non-defensible claims (we tried and walked back)

- "Universal 2B-token threshold for BOS function emergence across
  architectures." n=2 architectures with same threshold is suggestive
  but not universal.
- "PR (functional signal) emerges first, attention pattern second."
  Pythia step 256 and step 512 closures show PR-high ≠ load-bearing.
  The decoupling story is real for specific clean cases (OLMo step 1k,
  OLMoE step 5k) but PR alone does not reliably predict closure
  damage.
- "Internal mechanism transitions forecast training events" (the
  "predictive science" framing for the OLMoE alignment). One model,
  one phase transition, two batches is not "forecasting." Coincidence
  at the within-batch level is also possible.

## Files

- `results/developmental/{pythia_1b,olmo_1b,olmoe_1b_7b}/` — Ising
  scan per checkpoint (J matrices + summaries).
- `results/dev_per_class/{pythia_1b,olmo_1b,olmoe_1b_7b}/` — per-class
  attention-to-target per checkpoint (natural batch).
- `results/dev_per_class_synth/{pythia_1b,olmo_1b,olmoe_1b_7b}/` — same
  on synthetic induction batch.
- `results/dev_closure/` — 10 closure tests' raw data + summaries +
  loss-curve scans on regular and midseq batches.
- `figures/` — per-class emergence plots, PR-attn combined plots, OLMoE
  phase-alignment plot, per-layer emergence plots.
- `nostalgic-lederberg-80a58d/{pythia,olmo,olmoe}_phase1_trajectory.json`
  — pre-existing PR trajectory data from spectral-probe-circuits work.

## Followups deliberately not pursued

(For someone picking this back up.)

- **Closure on Pythia step 3000 prev-token at PR-peak** (PR=61, attn=29k).
  Would test whether Pythia prev-token ever passes closure. Would
  resolve the "Pythia step 512 prev-token failure: too-early or
  Pythia-specific" question.
- **Closure on OLMo at later checkpoints** for other classes (prev-token,
  self). Tests whether function-before-form holds beyond BOS.
- **Cross-architecture loss-curve scan**: same protocol on Pythia and
  OLMo. Would test whether their loss trajectories show analogous
  features at any mechanism-event point, even though they don't have
  the OLMoE-style attention phase transition.
- **Allen AI's official OLMoE training loss curve** (needs wandb auth).
  Would let us compare our 2000-example natural-batch loss against the
  full training curve at sub-checkpoint resolution.
- **Multiple seeds for each closure test**. Current closures use 5
  matched-random controls; closure verdicts depend on a single
  control-distribution variance, not characterized across batches.

All of these are bounded compute (~few hours each). The decision to
stop here is about scope, not feasibility.
