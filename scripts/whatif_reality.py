"""Weak spot 2: is the what-if believable? Two checks against REAL football, no new GRF runs.

A. Does GRF's 'no change' run look like what really happened?
   For each of the 20 batch moments (whatif_batch.py) the real 12 s after the start are scored exactly like a GRF run
   (attacking team's view, stop when the ball is lost): max threat, max xT, furthest point, ball lost or not.
   Compared with the 10 'no change' GRF runs of the same moment:
     rank correlation over moments  (do dangerous real moments stay dangerous in GRF?)
     bias                           (is GRF more or less dangerous on average?)
     real inside GRF's seed range   (in how many moments the real value lies between the lowest and highest GRF run)

B. Does pressing help in REAL matches, as the what-if says?   (World Cup 2022, 61,595 open-play ball positions)
   Natural experiment: compare positions that are alike (same distance-to-goal band and same wideness band), with and
   without pressure, on the real outcome 'this team shoots within 10 s'.
     carrier pressed     nearest defender within 2 m        vs  no defender within 5 m      (what-if: double_team, press)
     crowd at the ball   4+ defenders within 10 m           vs  0-1 within 10 m             (what-if: team_press)
   Difference in shot rate, weighted over the bands; 95% interval by resampling whole matches (1,000 times).
   Positions are not randomised (teams choose when to press), so this says whether the direction agrees, not the size.

Output: outputs/whatif/reality_moments.csv, reality_summary.csv, reality_pressing.csv, reality_chart.png (new files only)
Usage : .venv-analysis\\Scripts\\python.exe scripts\\whatif_reality.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import value_models as V  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WI = os.path.join(ROOT, "outputs", "whatif")
L, W, FPS, SECONDS = 105.0, 68.0, 10, 12.0
METRICS = ["max_threat", "max_xt", "furthest_m", "lost"]


def real_moment(seg, start_s, attack):
    z = np.load(os.path.join(ROOT, seg), allow_pickle=True)
    t, ball, home, away, own = z["time_s"], z["ball"].astype(float), z["home"].astype(float), z["away"].astype(float), z["owner"].astype(str)
    if float(np.nanmean(home[:, 0, 0])) > L / 2:                     # same turn as whatif_batch.pick_moments: home attacks x = 105
        ball, home, away = V_flip(ball), V_flip(home), V_flip(away)
    i0 = int(np.argmin(np.abs(t - start_s)))
    i1 = min(len(t), i0 + int(SECONDS * FPS))
    other = "away" if attack == "home" else "home"
    lost = np.flatnonzero(own[i0:i1] == other)
    keep = i0 + (lost[0] if len(lost) else i1 - i0)
    keep = max(keep, i0 + 1)
    b = ball[i0:keep]
    A, D = (home, away) if attack == "home" else (away, home)
    if attack == "away":                                             # turn so the attacking team attacks x = 105
        b, A, D = V_flip(b), V_flip(A), V_flip(D)
    A, D = A[i0:keep], D[i0:keep]
    th = [V.threat(b[k], A[k], D[k]) for k in range(0, len(b), 5)]
    return {"max_threat": float(max(th)), "max_xt": float(V.xt(b).max()), "furthest_m": float(np.nanmax(b[:, 0])),
            "lost": float(len(lost) > 0)}


def V_flip(p):
    return np.stack([L - p[..., 0], W - p[..., 1]], axis=-1)


def check_moments():
    mom = pd.read_csv(os.path.join(WI, "batch_moments.csv"))
    runs = pd.read_csv(os.path.join(WI, "batch_runs.csv"))
    base = runs[runs["scenario"] == "baseline"].copy()
    base["lost"] = base["lost_ball_s"].notna().astype(float)
    rows = []
    for _, m in mom.iterrows():
        r = real_moment(m["segment"], m["start_s"], m["attack"])
        g = base[base["moment"] == m["moment"]]
        row = {"moment": m["moment"], "seeds": len(g)}
        for k in METRICS:
            row[f"real_{k}"] = r[k]
            row[f"grf_{k}"] = g[k].mean()
            row[f"grf_{k}_lo"], row[f"grf_{k}_hi"] = g[k].min(), g[k].max()
        rows.append(row)
    df = pd.DataFrame(rows)
    summ = []
    for k in METRICS:
        real, grf = df[f"real_{k}"], df[f"grf_{k}"]
        inside = ((real >= df[f"grf_{k}_lo"]) & (real <= df[f"grf_{k}_hi"])).sum()
        summ.append({"metric": k, "real_mean": real.mean(), "grf_mean": grf.mean(), "bias_grf_minus_real": (grf - real).mean(),
                     "rank_corr": real.rank().corr(grf.rank()) if k != "lost" else real.corr(grf),
                     "real_inside_grf_range": f"{int(inside)}/{len(df)}"})
    return df, pd.DataFrame(summ)


def pressing():
    moves = pd.read_csv(os.path.join(V.OUT, "pff_moves.csv.gz"))
    ev = V.threat_labels(moves)
    ev["band"] = pd.cut(ev["s_dist"], [0, 12, 20, 30, 45, 200], labels=False).astype(str) + "_" + \
        pd.cut(ev["s_by_c"], [-1, 10, 20, 40], labels=False).astype(str)
    tests = {"carrier pressed (defender within 2 m vs none within 5 m)": (ev["s_near_def"] <= 2, ev["s_near_def"] > 5),
             "crowd at the ball (4+ defenders within 10 m vs 0-1)": (ev["s_def_10m"] >= 4, ev["s_def_10m"] <= 1)}
    matches = ev["match"].unique()
    rng = np.random.default_rng(0)

    def effect(e, treat, ctrl):
        d, n = 0.0, 0
        for _, g in e.groupby("band"):
            t, c = g[treat.loc[g.index]], g[ctrl.loc[g.index]]
            if len(t) >= 20 and len(c) >= 20:
                w = len(t)
                d += w * (t["shot10"].mean() - c["shot10"].mean())
                n += w
        return d / n if n else np.nan, n

    rows = []
    for name, (treat, ctrl) in tests.items():
        est, n = effect(ev, treat, ctrl)
        by = {m: ev.index[ev["match"] == m] for m in matches}
        boot = []
        for _ in range(1000):
            pick = rng.choice(matches, len(matches))
            idx = np.concatenate([by[m] for m in pick])
            e = ev.loc[idx].reset_index(drop=True)
            boot.append(effect(e, treat.loc[idx].reset_index(drop=True), ctrl.loc[idx].reset_index(drop=True))[0])
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        rows.append({"test": name, "positions_pressed": int(treat.sum()), "positions_free": int(ctrl.sum()), "matched_pressed": n,
                     "shot_rate_pressed": ev.loc[treat, "shot10"].mean(), "shot_rate_free": ev.loc[ctrl, "shot10"].mean(),
                     "matched_difference": est, "ci95_lo": lo, "ci95_hi": hi,
                     "agrees_with_whatif": bool(hi < 0)})
        print(f"  {name}: {est:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
    return pd.DataFrame(rows)


def chart(df, summ, press, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    for a, k, title in ((ax[0], "max_threat", "Max threat in 12 s"), (ax[1], "furthest_m", "Furthest the ball got (m)")):
        r = summ.set_index("metric").loc[k]
        a.errorbar(df[f"real_{k}"], df[f"grf_{k}"], yerr=np.clip([df[f"grf_{k}"] - df[f"grf_{k}_lo"], df[f"grf_{k}_hi"] - df[f"grf_{k}"]], 0, None),
                   fmt="o", c="#2a7ad5", capsize=3, alpha=0.8)
        lim = [min(df[f"real_{k}"].min(), df[f"grf_{k}_lo"].min()), max(df[f"real_{k}"].max(), df[f"grf_{k}_hi"].max())]
        a.plot(lim, lim, "--", c="#777", lw=1)
        a.set_xlabel("real match")
        a.set_ylabel("GRF 'no change' (mean, min-max of 10 runs)")
        a.set_title(f"{title}: rank corr {r['rank_corr']:.2f}, real inside GRF range {r['real_inside_grf_range']}")
        a.grid(alpha=0.3)
    a = ax[2]
    y = np.arange(len(press))
    a.barh(y, press["matched_difference"], xerr=np.clip([press["matched_difference"] - press["ci95_lo"], press["ci95_hi"] - press["matched_difference"]], 0, None),
           color=["#1baf7a" if v else "#b5b3ab" for v in press["agrees_with_whatif"]], capsize=4)
    a.set_yticks(y, [t.split(" (")[0] for t in press["test"]])
    a.axvline(0, c="k", lw=0.8)
    a.set_xlabel("change in 'shot within 10 s' (real World Cup, like-for-like positions)")
    a.set_title("Real football: does pressure lower the shot chance?\ngreen = yes (95% interval below 0), as the what-if says")
    a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=105)


def main():
    df, summ = check_moments()
    print("A. GRF 'no change' vs the real match (20 moments):")
    print(summ.round(3).to_string(index=False))
    print("\nB. Pressing in real matches:")
    press = pressing()
    df.round(4).to_csv(os.path.join(WI, "reality_moments.csv"), index=False)
    summ.round(4).to_csv(os.path.join(WI, "reality_summary.csv"), index=False)
    press.round(4).to_csv(os.path.join(WI, "reality_pressing.csv"), index=False)
    chart(df, summ, press, os.path.join(WI, "reality_chart.png"))
    pd.set_option("display.width", 220)
    print(press.round(4).to_string(index=False))
    print("\nwrote outputs/whatif/reality_moments.csv, reality_summary.csv, reality_pressing.csv, reality_chart.png")


if __name__ == "__main__":
    main()
