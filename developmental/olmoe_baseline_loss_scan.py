"""
olmoe_baseline_loss_scan.py

Lightweight pass to compute mean LM loss at OLMoE cached checkpoints
not covered by closure tests. Loops over checkpoints, runs forward pass
on natural-text batch, records baseline loss per checkpoint.

For the OLMoE 419B → 838B phase transition story.
"""

from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def evaluate_loss(model, tokens, query_pos, targets, device, batch_size=4):
    n = tokens.shape[0]
    losses = []
    accs = []
    logits_target = []
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
                losses.append(F.cross_entropy(
                    row.unsqueeze(0),
                    torch.tensor([tgt], device=device)).item())
                accs.append(float(int(row.argmax().item()) == tgt))
                logits_target.append(float(row[tgt].item()))
            del logits
            if device == "mps":
                torch.mps.empty_cache()
            if start > 0 and start % (batch_size * 25) == 0:
                rate = end / (time.time() - t0)
                eta = (n - end) / rate
                print(f"    {end}/{n} ({rate:.1f} ex/s, ETA {eta:.0f}s)",
                      flush=True)
    return {
        "n": n,
        "loss": float(np.mean(losses)),
        "loss_std": float(np.std(losses)),
        "acc_top1": float(np.mean(accs)),
        "mean_logit_target": float(np.mean(logits_target)),
        "elapsed_seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--revisions", required=True,
                    help="Comma-separated list of revisions to scan.")
    ap.add_argument("--batch-file",
                    default="/Volumes/Brandy/mini_gpt/.claude/worktrees/nostalgic-lederberg-80a58d/natural_induction_batch.pt")
    ap.add_argument("--out", default="results/dev_closure/olmoe_baseline_loss_scan.json")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    revisions = [r.strip() for r in args.revisions.split(",") if r.strip()]
    print(f"device: {device}")
    print(f"revisions ({len(revisions)}): {revisions}")

    batch = torch.load(args.batch_file, weights_only=False)
    tokens = batch["tokens"]; query_pos = batch["query_pos"]; targets = batch["targets"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.out).exists():
        results = json.load(open(args.out))
    else:
        results = {}

    from transformers import OlmoeForCausalLM
    for revision in revisions:
        print(f"\n=== {revision} ===")
        if revision in results:
            print(f"  already done: loss={results[revision]['loss']:.4f}, skipping")
            continue
        t0 = time.time()
        try:
            model = OlmoeForCausalLM.from_pretrained(
                args.model, revision=revision,
                dtype=torch.float16,
                attn_implementation="eager")
            model = model.to(device).eval()
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            results[revision] = {"error": str(e)}
            continue
        print(f"  loaded in {time.time()-t0:.0f}s")
        r = evaluate_loss(model, tokens, query_pos, targets, device,
                            batch_size=args.batch_size)
        results[revision] = r
        print(f"  done in {r['elapsed_seconds']:.0f}s | loss={r['loss']:.4f} "
              f"acc={r['acc_top1']:.4f} logit={r['mean_logit_target']:.3f}")
        del model
        if device == "mps":
            torch.mps.empty_cache()
        json.dump(results, open(args.out, "w"), indent=2)

    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
