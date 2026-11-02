"""Combine the learned distance with the hand-made ones, and test whether the mix points at important moments better.

Question: which Key States mark "the ball goes dead or play restarts"? (A moment no distance was told about.)

Each Key State gets six numbers:
    learned  how far the network's vector jumped since the previous Key State (from a model that never saw that match)
    d_ks, d_abs, d_rel, d_ball, gap_s   the hand-made measures (total change, players moved, arrangement changed, ball moved,
                                        time since the previous Key State)
Inside each match every number is turned into a rank between 0 and 1, so matches and models are comparable.

Candidates
    D_KS only          the 3 m rule's own value
    time gap only      the best simple hand-made signal so far
    hand-made (all 5)  a small logistic regression on the five hand-made numbers
    learned only       the network's jump
    simple mix         average of the learned rank and the time-gap rank (nothing to train)
    combined (all 6)   a small logistic regression on all six numbers

Fair test: for each real match, the regression is fitted on the OTHER six real matches and tested on the one left out.
Also tested on the simulated games (fitted on all real matches).

Input : outputs of run_folds.py (<name>_embeddings_fold_<match>.npz), <name>_adaptive_ks.csv, <name>_graphs.npz
Output: tables, outputs/combined_distance.png, outputs/combined_distance.csv, a line in outputs/run_log.csv

Usage :  python scripts\\combine_distances.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import runlog
import settings

REALS = [m for m in settings.available(settings.REAL) if os.path.exists(settings.base(m) + f"_embeddings_fold_{m}.npz")]
SIMS = settings.available(settings.SIM)
HAND = ["d_ks", "d_abs", "d_rel", "d_ball", "gap_s"]
SETS = {"D_KS only": ["d_ks"], "time gap only": ["gap_s"], "hand-made (all 5)": HAND, "learned only": ["learned"],
        "combined (all 6)": HAND + ["learned"]}
KEEP = [0.05, 0.10, 0.20, 0.30]


def pct(x):
    """Rank inside the match, scaled to 0..1 (missing values stay missing)."""
    return pd.Series(x).rank(pct=True).to_numpy()


def features(m, tag):
    ks = pd.read_csv(settings.base(m) + "_adaptive_ks.csv")
    e = np.load(settings.base(m) + f"_embeddings_{tag}.npz")["emb"]
    g = np.load(settings.graphs(m))
    mode = pd.read_csv(settings.csv(m), usecols=["frame", "game_mode"]).drop_duplicates("frame").set_index("frame").loc[g["frame"], "game_mode"].to_numpy()
    z = e / (np.nanstd(e, axis=0) + 1e-6)
    learned = np.full(len(z), np.nan)
    learned[1:] = np.linalg.norm(z[1:] - z[:-1], axis=1)
    y = np.zeros(len(mode), bool)
    y[1:] = (mode[1:] == "normal") != (mode[:-1] == "normal")                 # the ball goes dead or play restarts between k-1 and k
    raw = {"learned": learned, **{c: ks[c].to_numpy() for c in HAND}}
    df = pd.DataFrame({c: pct(v) for c, v in raw.items()})
    df["y"] = y
    return df.dropna().reset_index(drop=True)


def auc(score, y):
    r = pd.Series(score).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recall_at(score, y, frac):
    k = max(1, int(len(score) * frac))
    return float(y[np.argsort(-score)[:k]].sum() / max(y.sum(), 1))


def fit(X, y, l2=1e-2):
    """Small logistic regression, events weighted up because they are rare."""
    X = torch.tensor(X, dtype=torch.float32)
    t = torch.tensor(y.astype(np.float32))
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pw = torch.tensor((1 - y.mean()) / max(y.mean(), 1e-6), dtype=torch.float32)
    opt = torch.optim.LBFGS([w, b], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, t, pos_weight=pw) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(b.item())


def score_set(name, cols, train, test):
    if len(cols) == 1:
        return test[cols[0]].to_numpy()
    w, b = fit(pd.concat(train)[cols].to_numpy(), pd.concat(train)["y"].to_numpy())
    return test[cols].to_numpy() @ w + b


def main():
    data = {m: features(m, f"fold_{m}") for m in REALS}
    first = f"fold_{REALS[0]}"
    sim = {s: features(s, first) for s in SIMS if os.path.exists(settings.base(s) + f"_embeddings_{first}.npz")}
    print(f"{len(REALS)} real matches (each left out in turn), {len(sim)} simulated games\n")

    rows, rec, coefs = [], {k: {f: [] for f in KEEP} for k in list(SETS) + ["simple mix"]}, []
    for m in REALS:
        train = [data[x] for x in REALS if x != m]
        test = data[m]
        y = test["y"].to_numpy()
        row = {"held_out": m, "events": int(y.sum())}
        for name, cols in SETS.items():
            s = score_set(name, cols, train, test)
            row[name] = round(auc(s, y), 3)
            for f in KEEP:
                rec[name][f].append(recall_at(s, y, f))
        mix = test["learned"].to_numpy() + test["gap_s"].to_numpy()
        row["simple mix"] = round(auc(mix, y), 3)
        for f in KEEP:
            rec["simple mix"][f].append(recall_at(mix, y, f))
        rows.append(row)
        coefs.append(fit(pd.concat(train)[HAND + ["learned"]].to_numpy(), pd.concat(train)["y"].to_numpy())[0])
    t = pd.DataFrame(rows)
    order = ["D_KS only", "time gap only", "hand-made (all 5)", "learned only", "simple mix", "combined (all 6)"]
    print("1. How well does each pick the 'ball dead / restart' moments? (score on the match left out; 0.5 = chance, 1 = perfect)\n")
    print(t[["held_out", "events"] + order].to_string(index=False))
    mean = t[order].mean()
    print("\nmean over the 7 matches:  " + "   ".join(f"{k} {v:.3f}" for k, v in mean.items()))
    print(f"spread of the combined score over matches: {t['combined (all 6)'].std():.3f}")

    print("\n2. Keep only the top jumps: share of those moments that are kept (mean over the 7 matches)\n")
    r = pd.DataFrame({k: [np.mean(rec[k][f]) for f in KEEP] for k in order}, index=[f"top {int(f * 100)}%" for f in KEEP])
    r["random"] = KEEP
    print(r.round(2).to_string())

    print("\n3. What the combined model leans on (average weight; larger = more important)\n")
    cw = pd.Series(np.mean(coefs, axis=0), index=HAND + ["learned"]).sort_values(ascending=False)
    print(cw.round(2).to_string())

    allreal = list(data.values())
    simrows = []
    for s, d in sim.items():
        y = d["y"].to_numpy()
        row = {"match": s}
        for name, cols in SETS.items():
            row[name] = auc(score_set(name, cols, allreal, d), y)
        simrows.append(row)
    st = pd.DataFrame(simrows)
    print("\n4. On the simulated games (fitted on all real matches), mean score:  " + "   ".join(f"{c} {st[c].mean():.3f}" for c in SETS))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#EFA566", "#E8833A", "#C98A5A", "#1B7F4B", "#6B9E7F", "#14532D"]
    ax[0].barh(order[::-1], [mean[k] for k in order[::-1]], color=colors[::-1])
    ax[0].axvline(0.5, color="#9AA5AF", ls="--")
    ax[0].set_xlim(0.4, 1)
    ax[0].set_title("Finding 'ball dead / restart' moments, real matches (higher is better)", loc="left", fontsize=10)
    for k, v in zip(order[::-1], [mean[k] for k in order[::-1]]):
        ax[0].text(v + 0.005, k, f"{v:.2f}", va="center", fontsize=9)
    for k, c in zip(order, colors):
        ax[1].plot([f * 100 for f in KEEP], [np.mean(rec[k][f]) for f in KEEP], marker="o", color=c, label=k)
    ax[1].plot([f * 100 for f in KEEP], KEEP, color="#9AA5AF", ls="--", label="random")
    ax[1].set_xlabel("share of Key States kept, %")
    ax[1].set_ylabel("share of those moments kept")
    ax[1].legend(frameon=False, fontsize=7)
    for a in ax:
        for side in ("right", "top"):
            a.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/combined_distance.png", dpi=150)
    t.to_csv("outputs/combined_distance.csv", index=False)
    runlog.log({"script": "combine_distances", **{f"AUC {k}": round(float(mean[k]), 3) for k in order}})
    print("\nsaved outputs/combined_distance.png and outputs/combined_distance.csv")


if __name__ == "__main__":
    main()
