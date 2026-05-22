"""
select_closure_candidate_olmo_natural.py

Pick a closure-test candidate from OLMo 1B's natural-text Ising clusters.
Best ARI was at k=12 (ARI=0.199). Score each cluster by:
  - size (prefer 5-30 heads, like OLMo synthetic cluster 2 size)
  - isolation ratio (within-cluster mean |J| / outside mean |J|)
  - class purity (from natural-text mechinterp)

Pick the highest combined score.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.cluster import SpectralClustering

K_SPECTRAL = 12
RESULTS_DIR = Path("results/olmo_1b_nat_ising")
MECH_JSON = "/Volumes/Brandy/mini_gpt/.claude/worktrees/nostalgic-lederberg-80a58d/olmo_mechinterp_naturaltext.json"
N_LAYER = 16
N_HEAD = 16
MIN_SIZE = 4
MAX_SIZE = 12

J = np.load(RESULTS_DIR / "J.npy")
print(f"loaded J shape={J.shape} ||J||_F={np.linalg.norm(J):.3f}")

A = np.abs(J) + 1e-12
np.fill_diagonal(A, 0)
sc = SpectralClustering(n_clusters=K_SPECTRAL, affinity="precomputed",
                         assign_labels="kmeans", random_state=0, n_init=1)
labels = sc.fit_predict(A)

mech = json.load(open(MECH_JSON))
gt_map = {(c["layer"], c["head"]): c["classification"]
          for c in mech.get("classifications", [])}
print(f"natural-text supervised classifications: {len(gt_map)}")

F = N_LAYER * N_HEAD
clusters = []
for c in sorted(set(labels)):
    members = np.where(labels == c)[0]
    if len(members) == 0:
        continue
    head_list = [(int(m // N_HEAD), int(m % N_HEAD)) for m in members]
    layers = sorted(set(L for L, H in head_list))

    composition = {}
    n_classified = 0
    for f in members:
        L, H = divmod(f, N_HEAD)
        cls = gt_map.get((L, H))
        if cls is not None:
            composition[cls] = composition.get(cls, 0) + 1
            n_classified += 1
    purity = max(composition.values()) / n_classified if n_classified else 0.0

    member_set = set(int(m) for m in members)
    within, outside = [], []
    for i in members:
        for j in members:
            if int(i) >= int(j): continue
            within.append(abs(J[i, j]))
        for j in range(F):
            if j in member_set: continue
            outside.append(abs(J[i, j]))
    within_mean = float(np.mean(within)) if within else 0.0
    outside_mean = float(np.mean(outside)) if outside else 1e-12
    iso_ratio = within_mean / max(outside_mean, 1e-12)

    rec = {
        "cluster": int(c), "size": int(len(members)),
        "layers": layers,
        "heads": head_list,
        "composition": composition,
        "n_classified": n_classified,
        "purity": float(purity),
        "isolation_ratio": iso_ratio,
        "within_mean_J": within_mean,
        "outside_mean_J": outside_mean,
    }
    clusters.append(rec)
    comp_str = ", ".join(f"{k}={v}" for k, v in sorted(composition.items(), key=lambda x: -x[1])) or "—"
    print(f"  c{c}: size={len(members)} layers={layers[:5]}{'...' if len(layers)>5 else ''}")
    print(f"       comp(n_class={n_classified}): {comp_str}")
    print(f"       purity={purity:.2f}  isolation={iso_ratio:.2f}x")

candidates = [c for c in clusters if MIN_SIZE <= c["size"] <= MAX_SIZE]
if not candidates:
    print(f"\nNo cluster in size range, using all clusters")
    candidates = clusters

for c in candidates:
    size_factor = min(c["size"], MAX_SIZE) / MAX_SIZE
    c["combined_score"] = c["purity"] * c["isolation_ratio"] * size_factor

best = max(candidates, key=lambda x: x["combined_score"])

print(f"\n=== CHOSEN CANDIDATE ===")
print(f"  cluster {best['cluster']}, size={best['size']}, layers {best['layers']}")
print(f"  heads: {best['heads']}")
print(f"  composition: {best['composition']}")
print(f"  purity={best['purity']:.2f}, isolation={best['isolation_ratio']:.2f}x, "
      f"score={best['combined_score']:.3f}")

out = {"clusters": clusters, "candidate": best, "k_spectral": K_SPECTRAL}
json.dump(out, open(RESULTS_DIR / "closure_candidate.json", "w"), indent=2)
print(f"\nsaved {RESULTS_DIR / 'closure_candidate.json'}")
