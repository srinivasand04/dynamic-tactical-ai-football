"""Stage E: a learned Key-State distance, compared with the hand-made one.

The hand-made distance is D_KS (0.4 players moved + 0.4 arrangement changed + 0.2 ball moved). The learned distance is
how far apart the network's vectors of two consecutive Key States are. Two questions:

  1. Do they agree?   (rank correlation between the two, per match)
  2. Which one points at moments that matter in football?  The test uses something neither of them saw: the moment
     the ball goes dead or play restarts (from the game mode). For each distance we ask: if we keep only the largest
     jumps, how many of those moments do we keep?  A distance that finds them with fewer Key States is more useful.
     Random choice is the yardstick.

Input : <name>_adaptive_ks.csv (d_ks), embeddings made by a model that never saw the match
        (<name>_embeddings_fold_<match>.npz from run_folds.py; simulated matches use the first real fold)
Output: a table and outputs/learned_distance.png

Usage :  python scripts\\learned_distance.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import runlog
import settings

REALS = [m for m in settings.available(settings.REAL) if os.path.exists(settings.base(m) + f"_embeddings_fold_{m}.npz")]
SIMS = settings.available(settings.SIM)
KEEP = [0.05, 0.10, 0.20, 0.30, 0.50]


def ranks(x):
    return pd.Series(x).rank().to_numpy()


def spearman(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(ranks(a[ok]), ranks(b[ok]))[0, 1])


def auc(score, label):
    ok = ~np.isnan(score)
    s, y = score[ok], label[ok].astype(bool)
    r = ranks(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def recall_at(score, label, frac):
    """Share of the football moments that sit among the top `frac` largest jumps."""
    ok = ~np.isnan(score)
    s, y = score[ok], label[ok].astype(bool)
    k = max(1, int(len(s) * frac))
    top = np.argsort(-s)[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def load(m, tag):
    ks = pd.read_csv(settings.base(m) + "_adaptive_ks.csv")
    e = np.load(settings.base(m) + f"_embeddings_{tag}.npz")["emb"]
    g = np.load(settings.base(m) + "_graphs.npz")
    mode = pd.read_csv(settings.csv(m), usecols=["frame", "game_mode"]).drop_duplicates("frame").set_index("frame").loc[g["frame"], "game_mode"].to_numpy()
    sd = np.nanstd(e, axis=0) + 1e-6
    z = e / sd
    jump = np.full(len(z), np.nan)
    jump[1:] = np.linalg.norm(z[1:] - z[:-1], axis=1)                    # learned distance between consecutive Key States
    hand = {c: ks[c].to_numpy() for c in ("d_ks", "d_abs", "d_rel", "d_ball", "gap_s")}   # hand-made measures (NaN for the first)
    dead = np.zeros(len(mode), bool)
    dead[1:] = (mode[1:] == "normal") != (mode[:-1] == "normal")          # the ball goes dead or play restarts between k-1 and k
    return hand, jump, dead


def main():
    first_fold = f"fold_{REALS[0]}"
    rows, curves = [], {"hand-made D_KS": [], "learned distance": [], "random": []}
    pooled = {"j": [], "y": []}
    for m in REALS + SIMS:
        tag = f"fold_{m}" if m in REALS else first_fold
        if not os.path.exists(settings.base(m) + f"_embeddings_{tag}.npz"):
            continue
        hand, j, y = load(m, tag)
        kind = "real" if m in REALS else "simulated"
        row = {"match": m, "kind": kind, "events": int(y.sum()), "agree_(rank_corr)": round(spearman(hand["d_ks"], j), 2)}
        for c in ("d_ks", "d_abs", "d_rel", "d_ball", "gap_s"):
            row[f"AUC_{c}"] = round(auc(hand[c], y), 3)
        row["AUC_learned"] = round(auc(j, y), 3)
        rows.append(row)
        if kind == "real":
            for c in hand:
                pooled.setdefault(c, []).append(hand[c])
            pooled["j"].append(j); pooled["y"].append(y)
    t = pd.DataFrame(rows)
    print("1. Do they agree, and 2. how well does each point at 'ball goes dead / play restarts'?  (AUC: 0.5 = no better than chance, 1 = perfect)\n")
    print(t.to_string(index=False))
    for kind in ("real", "simulated"):
        s = t[t.kind == kind]
        if len(s):
            print(f"\n{kind}: agreement {s['agree_(rank_corr)'].mean():.2f}   AUC learned {s['AUC_learned'].mean():.3f}   hand-made: " +
                  "  ".join(f"{c[4:]} {s[c].mean():.3f}" for c in s.columns if c.startswith("AUC_") and c != "AUC_learned"))

    j, y = np.concatenate(pooled["j"]), np.concatenate(pooled["y"])
    hand_all = {c: np.concatenate(pooled[c]) for c in ("d_ks", "d_abs", "d_rel", "d_ball", "gap_s")}
    best = max(hand_all, key=lambda c: auc(hand_all[c], y))               # the strongest hand-made measure, so the comparison is fair
    d = hand_all[best]
    rng = np.random.default_rng(0)
    print("\n3. Keep only the largest jumps: share of the 'ball dead / restart' moments that are kept (real matches pooled)\n")
    out = []
    for f in KEEP:
        row = {"keep_top_%": int(f * 100), "D_KS": round(recall_at(hand_all["d_ks"], y, f), 2), f"best hand-made ({best})": round(recall_at(d, y, f), 2),
               "learned": round(recall_at(j, y, f), 2), "random": round(recall_at(rng.random(len(d)), y, f), 2)}
        out.append(row)
        for k, v in (("hand-made D_KS", row["D_KS"]), (f"best hand-made ({best})", row[f"best hand-made ({best})"]), ("learned distance", row["learned"]), ("random", row["random"])):
            curves.setdefault(k, []).append(v)
    print(pd.DataFrame(out).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.2, 4))
    for name, col in (("hand-made D_KS", "#EFA566"), (f"best hand-made ({best})", "#E8833A"), ("learned distance", "#1B7F4B"), ("random", "#9AA5AF")):
        ax.plot([f * 100 for f in KEEP], curves[name], marker="o", color=col, label=name)
    ax.set_xlabel("share of Key States kept (largest jumps first), %")
    ax.set_ylabel("share of 'ball dead / restart' moments kept")
    ax.set_title("Which distance picks the moments that matter?", loc="left", fontsize=10)
    ax.legend(frameon=False)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/learned_distance.png", dpi=150)
    t.to_csv("outputs/learned_distance.csv", index=False)
    real = t[t.kind == "real"]
    runlog.log({"script": "learned_distance", "AUC_d_ks": round(float(real["AUC_d_ks"].mean()), 3), "AUC_best_hand_made": round(float(max(real[c].mean() for c in real.columns if c.startswith("AUC_") and c != "AUC_learned")), 3),
                "AUC_learned": round(float(real["AUC_learned"].mean()), 3)})
    print("\nsaved outputs/learned_distance.png and outputs/learned_distance.csv")


if __name__ == "__main__":
    main()
