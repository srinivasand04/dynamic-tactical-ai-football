"""Do the pass edges really help the network? The paired comparison of the three networks of run_pass_proof.py.

    A  ours           B  pass_raw (real pass numbers)           C  pass_shuffle (the numbers shuffled: same size, no information)

For every held-out real match (a 'fold') the error of each network is averaged over its random starts (seeds). The comparison is
paired: the same matches for all three. For each comparison (B - A, B - C, C - A) and each error (players, ball, in metres; a
NEGATIVE difference means the first network is better):

    mean difference, its 95% interval (bootstrap over matches), the number of matches where it is better, and an exact sign-flip
    test (how often random +/- signs on the 7 differences give a mean as extreme: the p-value, two-sided).

What counts as proof
    B - A and B - C both clearly below zero (interval entirely below zero)  -> the pass INFORMATION helps
    B - A below zero but B - C not                                          -> the gain is just extra capacity
    neither                                                                  -> no evidence that the pass edges help
It also prints the random-start noise (how much one network varies between seeds on the same match): a difference smaller than
that noise cannot be trusted.

Input : outputs/run_log.csv (written by train_gnn.py, run_pass_proof.py)
Output: the tables, outputs/prove_pass_edges.csv, outputs/prove_pass_edges.png
Usage :  python scripts\\prove_pass_edges.py
"""
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARMS = {"ours": "A ours", "pass_raw": "B pass edges", "pass_shuffle": "C shuffled pass edges"}
METRICS = {"real_player_model_m": "player error (m)", "real_ball_model_m": "ball error (m)"}
COMPARE = [("pass_raw", "ours"), ("pass_raw", "pass_shuffle"), ("pass_shuffle", "ours")]
RNG = np.random.default_rng(0)


def load():
    log = pd.read_csv("outputs/run_log.csv")
    log = log[(log["script"] == "train_gnn") & log["holdout"].notna() & (log["holdout"] != "none") & log["real_player_model_m"].notna()].copy()
    log = log[~log["tag"].astype(str).str.contains("test|smoke", case=False)]
    log["arch"] = log["arch"].fillna("ours") if "arch" in log.columns else "ours"
    log["seed"] = log["seed"].fillna(0).astype(int)
    log = log[log["arch"].isin(ARMS)].sort_values("time").drop_duplicates(["arch", "holdout", "seed"], keep="last")
    return log


def sign_flip_p(d):
    d = np.asarray(d, float)
    if len(d) <= 12:
        flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
        means = (flips * d).mean(axis=1)
    else:
        means = (RNG.choice([1, -1], (20000, len(d))) * d).mean(axis=1)
    return float((np.abs(means) >= abs(d.mean()) - 1e-12).mean())


def boot(d, n=5000):
    return np.percentile(RNG.choice(d, (n, len(d)), replace=True).mean(axis=1), [2.5, 97.5])


def main():
    if not os.path.exists("outputs/run_log.csv"):
        raise SystemExit("no run log yet")
    log = load()
    have = {a: set(log[log["arch"] == a]["holdout"]) for a in ARMS}
    folds = sorted(set.intersection(*have.values()))
    print("runs per network and held-out match (seeds):")
    cnt = log.groupby(["arch", "holdout"]).size().unstack(0).reindex(columns=list(ARMS))
    print(cnt.fillna(0).astype(int).to_string())
    if len(folds) < 3:
        raise SystemExit("\nneed at least 3 held-out matches with all three networks: run scripts/run_pass_proof.py")
    seeds = log[log["holdout"].isin(folds)].groupby("arch")["seed"].nunique().to_dict()
    print(f"\n{len(folds)} matches with all three networks; seeds per network: {seeds}\n")

    fold_mean = log[log["holdout"].isin(folds)].groupby(["arch", "holdout"])[list(METRICS)].mean()
    rows = []
    for metric, mname in METRICS.items():
        print(f"== {mname}: error of each network (mean over seeds) ==")
        print(fold_mean[metric].unstack(0).reindex(columns=list(ARMS)).round(3).to_string())
        noise = log[log["holdout"].isin(folds)].groupby(["arch", "holdout"])[metric].std().groupby("arch").mean()
        if noise.notna().any():
            print("random-start noise (average std between seeds on one match):", noise.round(3).to_dict())
        print()
        for first, second in COMPARE:
            d = (fold_mean[metric].loc[first] - fold_mean[metric].loc[second]).reindex(folds).to_numpy()
            lo, hi = boot(d)
            wins = int((d < 0).sum())
            p = sign_flip_p(d)
            rows.append({"metric": mname, "comparison": f"{ARMS[first]} minus {ARMS[second]}", "mean_diff": d.mean(), "ci_low": lo, "ci_high": hi,
                         "better_in_matches": f"{wins}/{len(d)}", "p_sign_flip": p, **{f: v for f, v in zip(folds, d)}})
    t = pd.DataFrame(rows)
    show = t[["metric", "comparison", "mean_diff", "ci_low", "ci_high", "better_in_matches", "p_sign_flip"]]
    print("PAIRED DIFFERENCES (negative = the first network is better)\n")
    print(show.round(3).to_string(index=False))

    print("\nVERDICT")
    for mname in METRICS.values():
        s = t[t["metric"] == mname].set_index("comparison")
        ba, bc = s.loc["B pass edges minus A ours"], s.loc["B pass edges minus C shuffled pass edges"]
        if ba["ci_high"] < 0 and bc["ci_high"] < 0:
            v = "the pass information helps (B beats A and the shuffled control C)"
        elif ba["ci_high"] < 0:
            v = "B beats A but not the shuffled control: the gain is extra capacity, not pass information"
        elif ba["mean_diff"] < 0:
            v = "B is better on average but the interval includes zero: not proven (more seeds or more matches needed)"
        else:
            v = "no evidence that the pass edges help"
        print(f"  {mname:18s} {v}")

    os.makedirs("outputs", exist_ok=True)
    t.round(4).to_csv("outputs/prove_pass_edges.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for a, (metric, mname) in zip(ax, METRICS.items()):
        s = t[t["metric"] == mname].reset_index(drop=True)
        for i, r in s.iterrows():
            y = len(s) - 1 - i
            vals = [r[f] for f in folds]
            a.scatter(vals, [y] * len(vals), color="#b4b2a9", s=18, zorder=2)
            a.plot([r["ci_low"], r["ci_high"]], [y, y], color="#2a78d6", lw=3, zorder=3)
            a.scatter([r["mean_diff"]], [y], color="#2a78d6", s=50, zorder=4)
        a.axvline(0, color="#73726c", lw=1, ls=(0, (4, 3)))
        a.set_yticks(range(len(s)), list(s["comparison"])[::-1])
        a.set_xlabel("error difference in metres (left of the line = first network better)")
        a.set_title(mname + "   (gray dots = single matches, blue = mean and 95% interval)", loc="left", fontsize=9)
        for side in ("right", "top"):
            a.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/prove_pass_edges.png", dpi=150)
    print("\nsaved outputs/prove_pass_edges.csv and outputs/prove_pass_edges.png")


if __name__ == "__main__":
    main()
