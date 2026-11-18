"""Weak spot 6: are the team styles real, and does GRF keep them?

A. Richer real style fingerprint per team and match (World Cup 2022 tracking, PFF; attacking towards x = 105):
     line_height_m      out of possession: how far the 4 deepest outfield players stand from their own goal
     press_dist_m       out of possession: distance from the ball to the nearest defender (lower = closer pressing)
     regain_height_m    where the team wins the ball back, metres from its own goal (higher = presses high)
     length_out_m       out of possession: team length (deepest to highest outfield player, 10th-90th percentile)
     width_in_m         in possession: team width (10th-90th percentile of y)
     width_out_m        out of possession: team width
     opp_half_share     in possession: share of time the ball is in the opponent half
     possession         share of live time with the ball
   Recognisable? Nearest-centroid classifier (standardised numbers), leave one match out: can it tell Argentina,
   France and Croatia apart from one match's numbers? Chance = 1/3.

B. The same test on the GRF league games of the team models (shots, xG, passes, pass completion, possession, time in
   the opponent half per game). If GRF teams are recognisable, the models carry a style into the game.

C. Real vs GRF ordering on the two numbers both have (possession, opponent-half share): does the team that has more of it
   in reality also have more in GRF?

Output: outputs/team_league/style_real.csv, style_check.csv (new files only)
Usage : .venv-analysis\\Scripts\\python.exe scripts\\team_style_check.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "team_league")
TEAMS = ["Argentina", "France", "Croatia"]
L, W = 105.0, 68.0
REAL = ["line_height_m", "press_dist_m", "regain_height_m", "length_out_m", "width_in_m", "width_out_m", "opp_half_share", "possession"]
GRF = ["shots", "xg", "passes", "pass_completion", "possession", "opp_half"]


def style(match, side):
    d = pd.read_csv(settings.csv(match), usecols=["frame", "team", "role", "x_m", "y_m", "ball_x_m", "ball_y_m", "ball_owner_team", "game_mode", "period"])
    d = d[(d["game_mode"] == "normal") & (d["frame"] % 5 == 0)]
    rows = []
    for per, g in d.groupby("period"):
        gk = g[(g["team"] == side) & (g["role"] == "GK")]["x_m"].mean()
        flip = gk > L / 2                                                   # own goal on the right: turn so the team attacks x = 105
        g = g.copy()
        if flip:
            g["x_m"], g["y_m"], g["ball_x_m"], g["ball_y_m"] = L - g["x_m"], W - g["y_m"], L - g["ball_x_m"], W - g["ball_y_m"]
        rows.append(g)
    g = pd.concat(rows)
    own = g[(g["team"] == side) & (g["role"] != "GK")]
    other = "away" if side == "home" else "home"
    fr = g.drop_duplicates("frame").set_index("frame")
    out_f = fr.index[fr["ball_owner_team"] == other]
    in_f = fr.index[fr["ball_owner_team"] == side]
    o_out, o_in = own[own["frame"].isin(out_f)], own[own["frame"].isin(in_f)]
    q = lambda s, a, b: s.quantile(b) - s.quantile(a)
    deep4 = o_out.groupby("frame")["x_m"].apply(lambda x: np.sort(x.to_numpy())[:4].mean())
    bx = fr.loc[o_out["frame"], ["ball_x_m", "ball_y_m"]].to_numpy()
    dist = np.hypot(o_out["x_m"].to_numpy() - bx[:, 0], o_out["y_m"].to_numpy() - bx[:, 1])
    press = pd.Series(dist, index=o_out["frame"].to_numpy()).groupby(level=0).min()
    owner = fr["ball_owner_team"].to_numpy()
    won = np.flatnonzero((owner[1:] == side) & (owner[:-1] == other)) + 1
    return {"line_height_m": deep4.mean(), "press_dist_m": press.mean(),
            "regain_height_m": float(fr["ball_x_m"].to_numpy()[won].mean()) if len(won) else np.nan,
            "length_out_m": o_out.groupby("frame")["x_m"].apply(lambda x: q(x, .1, .9)).mean(),
            "width_in_m": o_in.groupby("frame")["y_m"].apply(lambda x: q(x, .1, .9)).mean(),
            "width_out_m": o_out.groupby("frame")["y_m"].apply(lambda x: q(x, .1, .9)).mean(),
            "opp_half_share": float((fr.loc[in_f, "ball_x_m"] > L / 2).mean()),
            "possession": len(in_f) / max(len(in_f) + len(out_f), 1)}


def loo_centroid(X, y):
    """Leave-one-out nearest-centroid accuracy on standardised numbers."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    hit = 0
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z, z = (X[tr] - mu) / sd, (X[i] - mu) / sd
        cent = {c: Z[y[tr] == c].mean(0) for c in np.unique(y[tr])}
        hit += min(cent, key=lambda c: np.sum((z - cent[c]) ** 2)) == y[i]
    return hit / len(y)


def perm_p(X, y, acc, n=500, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([loo_centroid(X, rng.permutation(y)) >= acc for _ in range(n)]))


def main():
    reg = settings.registry("pff")
    rows = []
    for _, r in reg.iterrows():
        for side in ("home", "away"):
            team = r[side]
            if team in TEAMS:
                rows.append({"team": team, "match": str(r["match_id"]), **style(str(r["match_id"]), side)})
                print(f"  {team} in {r['match_id']} done", flush=True)
    real = pd.DataFrame(rows)
    real.round(4).to_csv(os.path.join(OUT, "style_real.csv"), index=False)

    g = pd.read_csv(os.path.join(OUT, "league_games.csv"))
    grf = pd.concat([g.rename(columns={f"{s}_{k}": k for k in GRF}).assign(team=g[s])[["team"] + GRF] for s in ("left", "right")])
    grf = grf[grf["team"].isin(TEAMS)].dropna(subset=GRF).reset_index(drop=True)          # games v GRF AI miss some numbers

    checks = []
    for name, X, y in (("real matches, richer style (8 numbers)", real[REAL].fillna(real[REAL].mean()), real["team"]),
                       ("real matches, only possession + opp. half", real[["possession", "opp_half_share"]], real["team"]),
                       ("GRF league games (6 numbers)", grf[GRF], grf["team"])):
        acc = loo_centroid(X, y)
        checks.append({"test": name, "rows": len(y), "accuracy": acc, "chance": 1 / 3, "p_value": perm_p(X, y, acc)})
    chk = pd.DataFrame(checks)
    means_real = real.groupby("team")[["possession", "opp_half_share"]].mean()
    means_grf = grf.groupby("team")[["possession", "opp_half"]].mean().rename(columns={"opp_half": "opp_half_share"})
    order = pd.DataFrame({"real_possession": means_real["possession"], "grf_possession": means_grf["possession"],
                          "real_opp_half": means_real["opp_half_share"], "grf_opp_half": means_grf["opp_half_share"]})
    chk.round(4).to_csv(os.path.join(OUT, "style_check.csv"), index=False)
    order.round(4).to_csv(os.path.join(OUT, "style_real_vs_grf.csv"))
    pd.set_option("display.width", 200)
    print("\nA. Real style per team (mean over its World Cup matches):")
    print(real.groupby("team")[REAL].mean().round(2).to_string())
    print("\nB. Can a simple classifier tell the teams apart? (leave one out, chance 33%)")
    print(chk.round(3).to_string(index=False))
    print("\nC. Real vs GRF, team means:")
    print(order.round(3).to_string())
    print("\nwrote outputs/team_league/style_real.csv, style_check.csv, style_real_vs_grf.csv")


if __name__ == "__main__":
    main()
