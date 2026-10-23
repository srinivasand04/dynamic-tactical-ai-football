"""A clean 'how unusual is this match' number, from models that never saw the match.

For every real match M (fold models made by run_folds.py):
    reference = the Key States of the OTHER real matches, in the vector space of the model that held M out
    real      = how far the Key States of M are from that reference         (a real match the model never saw)
    simulated = how far each simulated match is from the same reference      (never seen either)
So real and simulated are measured by the same ruler, and neither was in training.

Distance = average, over Key States, of the distance to the closest reference Key State (after standardising the vector
by the reference matches). Larger = more unusual.

Input : <name>_embeddings_fold_<M>.npz for every real match M (run_folds.py)
Output: a table per fold and the average over folds, outputs/novelty_holdout.csv

Usage :  python scripts\\novelty_holdout.py
"""
import os

import numpy as np
import pandas as pd
import torch

import runlog
import settings

REALS = [m for m in settings.available(settings.REAL) if os.path.exists(settings.base(m) + f"_embeddings_fold_{m}.npz")]
SIMS = settings.available(settings.SIM)


def emb(m, tag):
    e = np.load(settings.base(m) + f"_embeddings_{tag}.npz")["emb"]
    return e[~np.isnan(e).any(axis=1)]


def nearest(query, ref, n_q=1500, n_r=5000, seed=0):
    rng = np.random.default_rng(seed)
    q = torch.from_numpy(query[rng.choice(len(query), min(n_q, len(query)), replace=False)]).float()
    r = torch.from_numpy(ref[rng.choice(len(ref), min(n_r, len(ref)), replace=False)]).float()
    return float(torch.cdist(q, r).min(dim=1).values.mean())


def main():
    rows = []
    for m in REALS:
        tag = f"fold_{m}"
        ref = np.concatenate([emb(x, tag) for x in REALS if x != m])
        mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-6
        z = lambda e: (e - mu) / sd
        ref_z = z(ref)
        row = {"held_out_real": m, "real": round(nearest(z(emb(m, tag)), ref_z), 3)}
        sims = [nearest(z(emb(s, tag)), ref_z) for s in SIMS if os.path.exists(settings.base(s) + f"_embeddings_{tag}.npz")]
        row["simulated_mean"] = round(float(np.mean(sims)), 3)
        row["simulated_min"] = round(float(np.min(sims)), 3)
        row["ratio_sim/real"] = round(row["simulated_mean"] / row["real"], 2)
        rows.append(row)
    t = pd.DataFrame(rows)
    print("Distance to the closest Key State in the other real matches (larger = more unusual)\n")
    print(t.to_string(index=False))
    print(f"\nreal {t['real'].mean():.2f} (spread {t['real'].std():.2f})   simulated {t['simulated_mean'].mean():.2f} (spread {t['simulated_mean'].std():.2f})"
          f"   ratio {t['ratio_sim/real'].mean():.2f}")
    t.to_csv("outputs/novelty_holdout.csv", index=False)
    runlog.log({"script": "novelty_holdout", "folds": len(t), "real_novelty": round(t["real"].mean(), 3),
                "sim_novelty": round(t["simulated_mean"].mean(), 3), "ratio": round(t["ratio_sim/real"].mean(), 3)})


if __name__ == "__main__":
    main()
