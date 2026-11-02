"""Do teams have a recognisable SHAPE? A cheap second test of 'team models' that needs no network training.

For every team-match of a dataset (registry with home/away names) the shape of the team is summarised at its Key States, with the team
always turned to attack towards x = 105:
    line    mean x of the 4 deepest outfield players (height of the back line, m from own goal)
    mean_x  mean x of the ten outfield players (how far up the pitch the team lives)
    width   standard deviation of y over the outfield players (m)
    length  deepest to highest outfield player (m)
each measured separately while the ball is in the own / middle / opponent third (12 numbers per team-match).

Test: leave one match out, assign it to the team whose other matches have the nearest average shape (z-scored numbers); accuracy
against chance (1 / number of teams). A one-way ANOVA F per number says how much of its variation is between teams rather than
between matches of the same team. Teams with at least --min-matches matches are used.

A second variant uses only Key States where at least --min-seen of the ten outfield players of that team were really seen (PFF
estimates players outside the camera view, and an estimated player may be placed by the model's idea of a formation).

Input : the common-format files and the Key State files of the dataset (run_pipeline.py --dataset NAME)
Output: outputs/team_style_<dataset>.csv (one row per team-match) and a printed summary

Usage :  python scripts\\team_style.py --dataset pff
"""
import argparse
import numpy as np
import pandas as pd
import settings

L = 105.0


def features(m, side, min_seen):
    d = pd.read_csv(settings.csv(m), usecols=lambda c: c in ("frame", "team", "player_id", "role", "x_m", "y_m", "visible", "ball_x_m"))
    ks = pd.read_csv(settings.base(m) + "_adaptive_ks.csv")["frame"]
    d = d[d["frame"].isin(ks)]
    own = d[(d["team"] == side) & (d["role"] != "GK")]
    flip = side == "away"
    x = (L - own["x_m"]) if flip else own["x_m"]
    y = own["y_m"]
    bx = d.drop_duplicates("frame").set_index("frame")["ball_x_m"]
    bx = (L - bx) if flip else bx
    own = own.assign(x=x.to_numpy(), y=y.to_numpy())
    rows = []
    for f, g in own.groupby("frame"):
        if len(g) < 10:
            continue
        if min_seen and "visible" in g and g["visible"].sum() < min_seen:
            continue
        xs = np.sort(g["x"].to_numpy())
        rows.append((bx.loc[f], xs[:4].mean(), xs.mean(), g["y"].std(), xs[-1] - xs[0]))
    r = pd.DataFrame(rows, columns=["ball_x", "line", "mean_x", "width", "length"])
    out = {"n_key_states": len(r)}
    for third, (lo, hi) in {"own": (0, L / 3), "mid": (L / 3, 2 * L / 3), "opp": (2 * L / 3, L + 1)}.items():
        s = r[(r["ball_x"] >= lo) & (r["ball_x"] < hi)]
        for c in ("line", "mean_x", "width", "length"):
            out[f"{c}_{third}"] = s[c].mean() if len(s) >= 20 else np.nan
    return out


def anova_f(X, y):
    """One-way ANOVA F for each column of X (samples x numbers) over the labels y."""
    labels = np.unique(y)
    n, k = len(y), len(labels)
    F = []
    for j in range(X.shape[1]):
        col = X[:, j]
        grand = col.mean()
        ssb = sum((y == l).sum() * (col[y == l].mean() - grand) ** 2 for l in labels)
        ssw = sum(((col[y == l] - col[y == l].mean()) ** 2).sum() for l in labels)
        F.append((ssb / (k - 1)) / (ssw / (n - k) + 1e-12))
    return np.array(F)


def classify(T, cols):
    X = T[cols].to_numpy(float)
    X = X[:, ~np.isnan(X).any(axis=0)] if np.isnan(X).any() else X
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = T["team"].to_numpy()
    hit = 0
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        cent = {t: X[keep & (y == t)].mean(0) for t in np.unique(y) if (keep & (y == t)).any()}
        pred = min(cent, key=lambda t: np.linalg.norm(X[i] - cent[t]))
        hit += pred == y[i]
    return hit / len(y), X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pff")
    ap.add_argument("--min-matches", type=int, default=5)
    ap.add_argument("--min-seen", type=int, default=8)
    a = ap.parse_args()
    reg = settings.registry(a.dataset)
    have = set(settings.dataset_matches(a.dataset))
    import os
    reg = reg[reg["match_id"].isin(have) & reg["match_id"].map(lambda m: os.path.exists(settings.base(m) + "_adaptive_ks.csv"))]
    count = pd.concat([reg["home"], reg["away"]]).value_counts()
    teams = list(count[count >= a.min_matches].index)
    print(f"teams with >= {a.min_matches} matches: " + ", ".join(f"{t} ({count[t]})" for t in teams), flush=True)

    res = {}
    for label, seen in (("all positions", 0), (f"only Key States with >= {a.min_seen} of 10 players seen", a.min_seen)):
        rows = []
        for r in reg.itertuples():
            for side, team in (("home", r.home), ("away", r.away)):
                if team in teams:
                    rows.append({"team": team, "match": r.match_id, "side": side, **features(r.match_id, side, seen)})
            print(f"  [{label}] {r.match_id}", flush=True) if len(rows) % 10 == 0 else None
        T = pd.DataFrame(rows).dropna(axis=1, how="all").dropna()
        res[label] = T
        T.to_csv(f"outputs/team_style_{a.dataset}{'' if not seen else '_seen'}.csv", index=False)
    print("\nRESULT\n")
    for label, T in res.items():
        cols = [c for c in T.columns if c not in ("team", "match", "side", "n_key_states")]
        acc, X, y = classify(T, cols)
        F = anova_f(X, y)
        chance = 1 / len(np.unique(y))
        print(f"{label}: {len(T)} team-matches, {len(np.unique(y))} teams, {len(cols)} numbers")
        print(f"  leave-one-match-out team recognition {acc * 100:.0f}%  (chance {chance * 100:.0f}%)")
        top = sorted(zip(cols, F), key=lambda z: -z[1])[:4]
        print("  most team-specific numbers (ANOVA F, 1 = no difference between teams, bigger = more team-specific): " +
              ", ".join(f"{c} {f:.1f}" for c, f in top))
        print(f"  mean key states per team-match {T['n_key_states'].mean():.0f}")


if __name__ == "__main__":
    main()
