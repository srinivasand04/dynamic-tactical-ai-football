"""What-if on many real moments: every scenario of whatif.py, played on attacking moments from all 7 Bundesliga matches.

One moment says little (the Phase 1 coach test: no clear effect). Here each scenario is played on many moments, each with
the same seeds as its 'no change' twin, and the effect is pooled: the mean change over all moments with a 95% interval
(moments as the unit), and in how many moments the change helped.

Moments: from each match's 3-minute segment (export_segment.py, the busiest open play), times where a team has had the ball
for at least 1 s 6-40 m from the goal it attacks (not on the goal line), at least 20 s apart; the --per-match moments nearest goal.

Scores (attacking team's view, in --seconds): goal, shots, xG (value_models), max xT reached, max threat (state model:
chance of a shot within 10 s, sees all players), furthest point reached, ball lost.

Input : data/real/processed/<match>_segment_<t>.npz (made here with export_segment.py if missing)
Output: outputs/whatif/batch_runs.csv, batch_moments.csv, batch_summary.csv, batch_chart.png

Usage :  .venv\\Scripts\\python.exe scripts\\whatif_batch.py --seeds 6 --procs 4
         .venv\\Scripts\\python.exe scripts\\whatif_batch.py --report          (summary and chart of earlier runs; needs matplotlib)
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import whatif as WI

ROOT = WI.ROOT
OUT = WI.OUT
MATCHES = ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]
SCEN = ["baseline", "line_deeper", "line_higher", "press", "compact", "early_pass", "wide_attack", "striker_deeper",
        "extra_runner", "shoot_early", "patient", "quick_play", "att_as_Argentina", "att_as_France", "att_as_Croatia",
        "team_press", "low_block", "man_marking", "defence_wider", "keeper_off_line", "double_team",
        "overload_ball_side", "fullbacks_overlap", "switch_play", "long_ball"]
METRICS = ["goal", "shots", "xg", "max_xt", "max_threat", "furthest_m"]


def segments():
    out = {}
    for m in MATCHES:
        have = sorted(f for f in glob.glob(os.path.join(ROOT, "data", "real", "processed", f"{m}_segment_*.npz"))
                      if "name_home" in np.load(f, allow_pickle=True).files and len(np.load(f, allow_pickle=True)["time_s"]) >= 1200)
        if not have:
            py = os.path.join(ROOT, ".venv-analysis", "Scripts", "python.exe")
            subprocess.run([py, os.path.join(ROOT, "scripts", "export_segment.py"), "--match", m, "--minutes", "3"], cwd=ROOT, check=True)
            return segments()
        out[m] = have[0] if m != "J03WOH" else next((f for f in have if f.endswith("_3500.npz")), have[0])
    return out


def pick_moments(seg, per_match, seconds):
    z = np.load(seg, allow_pickle=True)
    t, ball, own, live = z["time_s"], z["ball"].astype(float), z["owner"].astype(str), z["live"]
    if float(np.nanmean(z["home"][:, 0, 0])) > WI.L / 2:                  # same turn as whatif.moment
        ball = WI.flip(ball)
    cand = []
    for i in range(2 * WI.FPS + 1, len(t)):
        o = own[i]
        if o not in ("home", "away") or not live[i - WI.FPS:i + 1].all() or (own[i - WI.FPS:i + 1] != o).any():
            continue
        to_goal = WI.L - ball[i, 0] if o == "home" else ball[i, 0]
        if 6 <= to_goal < 40 and 1 < ball[i, 1] < WI.W - 1:
            cand.append((to_goal, float(t[i]), o))
    chosen = []
    for d, ts, o in sorted(cand):
        if all(abs(ts - c[1]) >= 20 for c in chosen):
            chosen.append((d, ts, o))
        if len(chosen) == per_match:
            break
    return [{"segment": os.path.relpath(seg, ROOT), "start_s": ts, "attack": o, "ball_to_goal_m": round(d, 1)} for d, ts, o in sorted(chosen, key=lambda c: c[1])]


def job(j):
    mid, seg, start_s, scen, seed, seconds = j
    r = WI._job((seg, start_s, scen, 3.0, seed, seconds, None))
    r.update({"moment": mid})
    return r


def summary(df):
    df = df[df["error"].isna()] if "error" in df else df
    base = df[df["scenario"] == "baseline"].set_index(["moment", "seed"])
    rows = []
    for scen, g in df.groupby("scenario", sort=False):
        if scen == "baseline":
            continue
        g = g.set_index(["moment", "seed"])
        common = g.index.intersection(base.index)
        r = {"scenario": scen, "what_changed": WI.describe(scen, 3.0), "moments": g.loc[common].index.get_level_values(0).nunique(),
             "runs": len(common)}
        for k in METRICS:
            d = (g.loc[common, k] - base.loc[common, k]).groupby(level=0).mean()   # per moment: mean over seeds
            r[f"d_{k}"] = d.mean()
            r[f"d_{k}_ci95"] = WI.ci(d)
            r[f"{k}_helped_in"] = f"{int((d > 0).sum())}/{len(d)}"
        r["made_by"] = "defence" if scen in WI.DEFENCE else "attack"
        d, e = r["d_max_threat"], r["d_max_threat_ci95"]
        good = d > 0 if r["made_by"] == "attack" else d < 0             # good for the team that made the change
        r["verdict"] = ("works" if good else "backfires") if abs(d) > e else "no clear effect"
        rows.append(r)
    sm = pd.DataFrame(rows)
    return sm.sort_values(["made_by", "d_max_threat"], ascending=[True, False])


def chart(sm, path, n_moments):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(26, 11), sharey=True)
    y = np.arange(len(sm))
    for a, (k, title) in zip(ax, (("max_threat", "Max threat (chance of a shot in 10 s)"), ("xg", "xG per run"),
                                  ("goal", "Goals per run"), ("furthest_m", "Furthest the ball got (m)"))):
        d, e = sm[f"d_{k}"].to_numpy(), sm[f"d_{k}_ci95"].to_numpy()
        sig = np.abs(d) > np.nan_to_num(e, nan=np.inf)
        good = np.where(sm["made_by"] == "attack", d > 0, d < 0)
        col = np.where(~sig, "#b5b3ab", np.where(good, "#1baf7a", "#e0483b"))
        a.barh(y, d, xerr=e, color=col, capsize=3)
        a.axvline(0, color="k", lw=0.8)
        a.set_title(f"{title}\nchange vs no change, 95% interval")
        for i, (v, h) in enumerate(zip(d, sm[f"{k}_helped_in"])):
            a.text(0, i - 0.32, f" {v:+.3f}  (up in {h})", fontsize=7, color="#333")
    ax[0].set_yticks(y, [f"[{m[:3]}] {w}" for m, w in zip(sm["made_by"], sm["what_changed"])])
    ax[0].invert_yaxis()
    fig.suptitle(f"What-if on {n_moments} real attacking moments (7 Bundesliga matches), attacking team's numbers. "
                 f"[def] = change made by the defending team, [att] = by the attacking team.\n"
                 f"green = works for the team that made the change, red = backfires, grey = no clear effect (95% interval)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-match", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--scenarios", nargs="+", default=SCEN)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    runs_p = os.path.join(OUT, "batch_runs.csv")
    if not a.report:
        moments = []
        for m, seg in segments().items():
            for k, mo in enumerate(pick_moments(seg, a.per_match, a.seconds)):
                moments.append({"moment": f"{m}_{int(mo['start_s'])}", "match": m, **mo})
        mdf = pd.DataFrame(moments)
        mdf.to_csv(os.path.join(OUT, "batch_moments.csv"), index=False)
        print(f"{len(mdf)} moments:\n{mdf.to_string(index=False)}\n", flush=True)
        done = pd.read_csv(runs_p) if os.path.exists(runs_p) else pd.DataFrame(columns=["moment", "scenario", "seed"])
        have = set(zip(done["moment"], done["scenario"], done["seed"].astype(int)))
        jobs = [(r.moment, os.path.join(ROOT, r.segment), r.start_s, s, seed, a.seconds)
                for r in mdf.itertuples() for s in a.scenarios for seed in range(a.seeds) if (r.moment, s, seed) not in have]
        print(f"{len(jobs)} runs to play ({len(have)} already done), {a.procs} at a time", flush=True)
        t0, rows = time.time(), []
        with Pool(a.procs, initializer=WI._init, initargs=("pff",)) as pool:
            for k, r in enumerate(pool.imap_unordered(job, jobs, chunksize=2)):
                rows.append(r)
                if (k + 1) % 50 == 0 or k + 1 == len(jobs):
                    pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(runs_p, index=False)   # saved as it goes
                    el = (time.time() - t0) / 60
                    print(f"  {k + 1}/{len(jobs)} runs ({el:.1f} min, about {el / (k + 1) * (len(jobs) - k - 1):.0f} min left)", flush=True)
    df = pd.read_csv(runs_p)
    if "error" in df and df["error"].notna().any():
        print(f"{int(df['error'].notna().sum())} runs failed, e.g. {df.loc[df['error'].notna(), 'error'].iloc[0]}")
    sm = summary(df)
    sm.round(4).to_csv(os.path.join(OUT, "batch_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    cols = ["made_by", "what_changed", "verdict", "moments", "runs", "d_max_threat", "d_max_threat_ci95", "max_threat_helped_in", "d_xg", "d_xg_ci95",
            "d_goal", "d_goal_ci95", "d_furthest_m", "d_furthest_m_ci95"]
    print(sm[cols].round(3).to_string(index=False))
    try:
        chart(sm, os.path.join(OUT, "batch_chart.png"), df["moment"].nunique())
        print("wrote outputs/whatif/batch_chart.png")
    except ImportError:
        print("chart: run with --report in .venv-analysis (matplotlib)")


if __name__ == "__main__":
    main()
