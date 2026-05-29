# Closure-Validated Circuit Discovery in Attention Heads

*Co-activation proposes, ablation disposes.* Closure tests in dense and MoE
transformers at 1B scale.

📄 **[Read the draft → `draft.pdf`](draft.pdf)**

A cheap signal — co-activation cluster membership, attention-to-target
selectivity, participation ratio — is a circuit **proposal**, not a
confirmed circuit. We adapt Bhalla et al.'s (2026) SAE-feature Ising
clustering to **attention heads** and validate by **causal ablation**
(closure), asking of each proxy: does it predict load-bearing function?
*(Different object and validation from SAE-feature manifold work — not an
evaluation of that.)*

## Headline: co-activation communities under closure

| Model | Distribution | Heads | $z_{\Delta\ell}$ | $z_{\Delta\text{acc}}$ | $z_{\Delta z^{\text{tgt}}}$ | Verdict |
|---|---|---:|---:|---:|---:|---|
| OLMo 1B | synthetic | 5 | +1.83 | −1.97 | −2.68 | pass |
| Pythia 1B | synthetic | 25 | +2.05 | −1.16 | −2.06 | weak pass |
| **OLMo 1B** | **natural** | 10 | **+36.4** | **−50.6** | **−50.9** | **pass** |
| **Pythia 1B** | **natural** | 9 | +1.95 | **−9.9** | **−52.5** | **pass** |
| **OLMoE 1B-7B** | natural | 22 | **−5.64** | −0.22 | −2.97 | **does not pass** |

Candidate community discovered unsupervisedly, ablated, compared to 5
matched-random-head controls (z-scores in σ). **Dense models pass closure
on both distributions.** On MoE natural text the marginal Ising collapses
(ARI 0.006); route-conditional stratification recovers a +3.05σ signal but
that community fails closure in direction — ablating it *helps* loss. A
cautionary case for co-activation interpretability of MoE attention, not a
claim that MoE breaks interpretability.

## Second result: across training, two more proxies decouple from function

Running the same closure test on intermediate checkpoints, attention
selectivity and PR fail to track function in **both directions**: BOS-head
function is load-bearing ~2B tokens *before* its attention pattern forms
(OLMo + Pythia), while Pythia previous-token heads at step 512 show a sharp
attention pattern and high PR with *no* closure signal. Pattern and function
are distinct constructs. → [`developmental_notes.md`](developmental_notes.md),
§7 of the draft.

## Repo

```
draft.pdf / draft.tex   the working draft (compile with figures/ at root)
developmental_notes.md  companion notes for the training-axis results
pipeline/               Ising discovery, route conditioning, null control
closure/                candidate selection + closure tests
developmental/          checkpoint scans, per-class emergence, intermediate closures
analysis/  figures/  results/
```

Reproduce: `pip install -r requirements.txt`, then the pipeline →
candidate-selection → closure scripts (PyTorch+MPS/CUDA, `transformers`,
sklearn). The full protocol, claims, and limitations are in `draft.pdf`;
large attention tensors (`attn_at_query.npy`, `*_routes.npy`) are not
tracked and regenerate from the pipeline.

## License

MIT.
