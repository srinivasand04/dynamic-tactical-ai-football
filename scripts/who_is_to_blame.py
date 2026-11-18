"""Who is to blame? For real dangerous moments, how much did each defender's spot add to the danger?

Moments: Key States of the 7 Bundesliga matches where the team on the ball has threat >= 15% (threat model,
value_models: chance of a shot within 10 s), at most one per 5 s of play (the peak).
For each defender (all 11 of the defending team), only HE is put somewhere else, everybody else stays real, and threat
is computed again:
    typical ghost  where our movement model (3-seed ensemble, held-out predictions) expected him one Key State
                   (~1.1 s) earlier: "did he do what a normal defender does here?"
    best ghost     the spot he could have reached in the same time (7 m/s from where he was a Key State earlier, 1 m
                   grid) that makes the moment least dangerous: "could he have stopped it?"
Blame = real threat - threat with his ghost   (in %-points; > 0: his real spot made the moment more dangerous).

Outputs (outputs/blame/, new files only):
    blame_moments.csv   one row per defender per moment (both blames, name, role, match, time)
    blame_players.csv   per player: chances defended, mean / total blame, how often he is the most to blame
    blame_roles.csv     mean blame per role
    blame_chart.png     most-blamed players + an example moment
Usage : .venv-analysis\\Scripts\\python.exe scripts\\who_is_to_blame.py             (about 30 min)
        .venv-analysis\\Scripts\\python.exe scripts\\who_is_to_blame.py --report    (tables + chart from the saved rows)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings  # noqa: E402
import value_models as V  # noqa: E402
from football_scorecard import THREAT, PREDS, START, SUFFIX, ENSEMBLE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "blame")
L, W = 105.0, 68.0
MIN_THREAT, GAP_S, SPEED, HOLD_M = 0.15, 5.0, 7.0, 3.0
ROLE = {0: "keeper", 1: "centre-back", 2: "full-back", 3: "full-back", 4: "central mid", 5: "central mid",
        6: "wide mid", 7: "wide mid", 8: "attacking mid", 9: "striker"}
MEAN, STD, WT = np.array(THREAT["mean"]), np.array(THREAT["std"]), np.array(THREAT["weights"])
FEAT = [f[2:] for f in THREAT["features"]]


def threat(ball, att, dfs):
    x = V.state_features(ball, att, dfs)
    z = WT[0] + ((np.array([x[f] for f in FEAT]) - MEAN) / STD) @ WT[1:]
    return 1 / (1 + np.exp(-z))


def oriented(pos, team):
    """ball, attackers, defenders with the attacking team going towards x = 105 (away from its own keeper)."""
    a0, d0 = (0, 11) if team == 0 else (11, 0)
    flip = np.isfinite(pos[a0, 0]) and pos[a0, 0] > L / 2
    f = (lambda p: np.stack([L - p[..., 0], W - p[..., 1]], -1)) if flip else (lambda p: p)
    return f(pos[22]), f(pos[a0:a0 + 11]), f(pos[d0:d0 + 11]), f


def best_spot(ball, att, dfs, j, start, reach):
    """The reachable spot (disc of radius reach around start, 1 m grid) with the lowest threat for defender j."""
    g = np.arange(-reach, reach + 0.01, 1.0)
    gx, gy = np.meshgrid(g, g)
    keep = gx ** 2 + gy ** 2 <= reach ** 2
    cand = np.c_[start[0] + gx[keep], start[1] + gy[keep]]
    cand = cand[(cand[:, 0] >= 0) & (cand[:, 0] <= L) & (cand[:, 1] >= 0) & (cand[:, 1] <= W)]
    cand = np.r_[cand, np.asarray(dfs[j])[None]]                         # staying where he is is always possible
    best, spot = np.inf, start
    d2 = dfs.copy()
    for c in cand:
        d2[j] = c
        t = threat(ball, att, d2)
        if t < best:
            best, spot = t, c
    return best, spot


def names_for(m):
    """(frame, team, player_id) -> (name, shirt)."""
    d = pd.read_csv(settings.csv(m), usecols=lambda c: c in ("frame", "team", "player_id", "dfl_id", "jersey"))
    pfile = os.path.join(ROOT, "data", "real", "raw", f"{m}_players.csv")
    nm = pd.read_csv(pfile, encoding="utf-8").set_index("player_id")["name"].to_dict() if os.path.exists(pfile) else {}
    d["name"] = d["dfl_id"].map(nm).fillna("#" + d["jersey"].astype("Int64").astype(str))
    return d.set_index(["frame", "team", "player_id"])[["name", "jersey"]]


def match_blame(m):
    g = np.load(settings.base(m) + f"_graphs{SUFFIX}.npz")
    xy, role, frame, t = g["xy"], g["role"], g["frame"], g["time_s"]
    preds = [np.load(os.path.join(PREDS, f"{s}_{m}.npz")) for s in ENSEMBLE]
    pred = np.mean([p["pred"] for p in preds], axis=0)
    ok = preds[0]["ok"]
    names = names_for(m)
    rows, team, cands = [], None, []
    for i in range(len(pred)):                         # moment = Key State k+1, ghost from Key State k
        k = START + i + 1
        now = xy[k]
        if not np.isfinite(now[22, 0]):
            continue
        d = np.hypot(*(now[:22] - now[22]).T)
        if np.isfinite(d).any() and np.nanmin(d) < HOLD_M:
            team = 0 if np.nanargmin(d) < 11 else 1
        if team is None:
            continue
        b, a, df, _ = oriented(now, team)
        th = threat(b, a, df)
        if th >= MIN_THREAT:
            cands.append((t[k], th, i, k, team))
    keep, last = [], -1e9                              # peak of each dangerous spell, at least 5 s apart
    for c in sorted(cands, key=lambda c: -c[1]):
        if all(abs(c[0] - q[0]) >= GAP_S for q in keep):
            keep.append(c)
    for tk, th, i, k, team in sorted(keep):
        now, before = xy[k], xy[k - 1]
        b, a, df, f = oriented(now, team)
        d0 = 11 if team == 0 else 0
        dt = max(float(t[k] - t[k - 1]), 0.3)
        ghost_typ = f(before[d0:d0 + 11] + pred[i][d0:d0 + 11])            # typical: model's spot from Key State k-1
        start_pos = f(before[d0:d0 + 11])
        side = "away" if d0 == 11 else "home"
        for j in range(11):
            if not np.isfinite(df[j, 0]):
                continue
            d2 = df.copy()
            t_typ = np.nan
            if ok[i, d0 + j] and np.isfinite(ghost_typ[j, 0]):
                d2[j] = ghost_typ[j]
                t_typ = threat(b, a, d2)
            st = start_pos[j] if np.isfinite(start_pos[j, 0]) else df[j]
            t_best, spot = best_spot(b, a, df, j, st, SPEED * dt)
            try:
                nm, shirt = names.loc[(int(frame[k]), side, j)].iloc[0] if isinstance(names.loc[(int(frame[k]), side, j)], pd.DataFrame) \
                    else names.loc[(int(frame[k]), side, j)]
            except KeyError:
                nm, shirt = f"{side} slot {j}", -1
            rows.append({"match": m, "time_s": round(float(tk), 1), "ks": int(k), "defending": side, "slot": j, "name": nm,
                         "shirt": shirt, "role": ROLE.get(int(role[k, d0 + j]), "unknown"), "threat": round(float(th), 4),
                         "blame_typical": round(float(th - t_typ) * 100, 2) if np.isfinite(t_typ) else np.nan,
                         "blame_best": round(float(th - t_best) * 100, 2),
                         "dist_ball_m": round(float(np.hypot(*(df[j] - b))), 1),
                         "best_x": round(float(spot[0]), 1), "best_y": round(float(spot[1]), 1)})
    return pd.DataFrame(rows), len(keep)


def chart(df, pl, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1, 1.25]})
    top = pl[pl.chances >= 20].sort_values("mean_share", ascending=False).head(15).iloc[::-1]
    ax[0].barh(top["name"] + " (" + top["team"] + ")", top["mean_share"], color="#e0483b")
    ax[0].axvline(100 / 11, color="0.4", ls="--", lw=0.8)
    ax[0].set_xlabel("mean share of his team's blame per chance (%; dashed = equal share 9%)")
    ax[0].set_title("Most to blame (players in 20+ chances)")
    # example: the moment with the single biggest blame
    big = df[(df.blame_best >= 3) & (df.groupby(["match", "ks"])["blame_best"].transform("count") >= 10)]
    r = big.loc[big["blame_share"].idxmax()]                 # the clearest case: one defender carries most of the blame
    g = np.load(settings.base(r["match"]) + f"_graphs{SUFFIX}.npz")
    pos = g["xy"][int(r["ks"])]
    team = 0 if r["defending"] == "away" else 1
    b, a, dfs, f = oriented(pos, team)
    mm = df[(df.match == r["match"]) & (df.ks == r["ks"])].set_index("slot")
    a_ = ax[1]
    a_.add_patch(plt.Rectangle((0, 0), L, W, fill=False, color="0.5"))
    a_.add_patch(plt.Rectangle((L - 16.5, W / 2 - 20.16), 16.5, 40.32, fill=False, color="0.5"))
    a_.scatter(a[:, 0], a[:, 1], s=60, c="#3a7bd5", label="attackers")
    for j in range(11):
        if not np.isfinite(dfs[j, 0]):
            continue
        bl = float(mm.loc[j, "blame_best"]) if j in mm.index else 0.0
        a_.scatter(*dfs[j], s=60 + 25 * max(bl, 0), c="#e0483b", edgecolors="k", zorder=3)
        if j in mm.index and bl >= 1:
            a_.annotate(f"{mm.loc[j, 'name']} {bl:.0f}", dfs[j], xytext=(4, 6), textcoords="offset points", fontsize=8)
            a_.annotate("", xy=(mm.loc[j, "best_x"], mm.loc[j, "best_y"]), xytext=dfs[j], arrowprops=dict(arrowstyle="->", color="#e0483b", ls="--"))
    a_.scatter(*b, s=40, c="k", marker="o", label="ball", zorder=4)
    a_.set_xlim(40, 106)
    a_.set_ylim(-1, 69)
    a_.set_aspect("equal")
    a_.set_title(f"{r['match']} {r['time_s']:.0f} s: threat {r['threat'] * 100:.0f}%. Red size = blame; arrow = best reachable spot")
    a_.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def report(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    pos = df["blame_best"].clip(lower=0)
    df["blame_share"] = (pos / pos.groupby([df.match, df.ks]).transform("sum").replace(0, np.nan) * 100).round(1)
    top = df.loc[df.dropna(subset=["blame_best"]).groupby(["match", "ks"])["blame_best"].idxmax()]
    pl = df.groupby(["name", "team"]).agg(chances=("ks", "size"), mean_blame_best=("blame_best", "mean"),
                                          mean_share=("blame_share", "mean"), total_blame_best=("blame_best", "sum"),
                                          mean_blame_typical=("blame_typical", "mean"),
                                          main_role=("role", lambda s: s.mode().iat[0])).reset_index()
    pl["most_to_blame_in"] = pl.set_index(["name", "team"]).index.map(top.groupby(["name", "team"]).size()).fillna(0).astype(int)
    pl = pl.sort_values("mean_share", ascending=False)
    pl.round(2).to_csv(os.path.join(OUT, "blame_players.csv"), index=False)
    ro = df.groupby("role").agg(rows=("ks", "size"), mean_blame_best=("blame_best", "mean"), mean_share=("blame_share", "mean"),
                                mean_blame_typical=("blame_typical", "mean")).reset_index()
    ro["most_to_blame"] = ro["role"].map(top["role"].value_counts()).fillna(0).astype(int)
    ro = ro.sort_values("mean_share", ascending=False)
    ro.round(2).to_csv(os.path.join(OUT, "blame_roles.csv"), index=False)
    tm = df.drop_duplicates(["match", "ks"]).groupby("team").agg(chances=("ks", "size"), mean_threat=("threat", "mean"))
    tm["blame_removable_pp"] = df.groupby(["team", "match", "ks"])["blame_best"].max().groupby(level=0).mean()
    n = df.groupby(["match", "ks"]).ngroups
    print(f"\n{n} dangerous moments, mean threat {df.drop_duplicates(['match', 'ks'])['threat'].mean() * 100:.0f}%")
    print(f"most-blamed defender could remove on average {top['blame_best'].mean():.1f} %-points of threat on his own")
    print(f"typical ghost blame: mean {df['blame_typical'].mean():.2f}, |blame| > 1 pp in {(df['blame_typical'].abs() > 1).mean() * 100:.0f}% of rows")
    pd.set_option("display.width", 200)
    print("\nby role (share = % of the team's blame in the moment; 11 players -> 9% each if equal):\n" + ro.round(2).to_string(index=False))
    print("\nby team:\n" + tm.round(3).to_string())
    print("\nmost to blame (share, 20+ chances):\n" + pl[pl.chances >= 20].head(10).round(2).to_string(index=False))
    df.to_csv(os.path.join(OUT, "blame_moments.csv"), index=False)
    return df, pl


def main():
    os.makedirs(OUT, exist_ok=True)
    if "--report" in sys.argv:                          # only the tables and chart, from the saved rows
        df, pl = report(pd.read_csv(os.path.join(OUT, "blame_moments.csv")))
        chart(df, pl, os.path.join(OUT, "blame_chart.png"))
        return
    parts = []
    for m in settings.available(settings.REAL):
        if not all(os.path.exists(os.path.join(PREDS, f"{s}_{m}.npz")) for s in ENSEMBLE):
            continue
        df, n = match_blame(m)
        parts.append(df)
        print(f"  {m}: {n} dangerous moments, {len(df)} defender rows", flush=True)
    df = pd.concat(parts, ignore_index=True)
    from export_viewer import team_names
    tn = {m: team_names(m) for m in df.match.unique()}
    df["team"] = [tn[m][0] if s == "home" else tn[m][1] for m, s in zip(df.match, df.defending)]
    df, pl = report(df)
    try:
        chart(df, pl, os.path.join(OUT, "blame_chart.png"))
        print("\nwrote outputs/blame/blame_moments.csv, blame_players.csv, blame_roles.csv, blame_chart.png")
    except ImportError:
        print("chart needs matplotlib (.venv-analysis)")


if __name__ == "__main__":
    main()
