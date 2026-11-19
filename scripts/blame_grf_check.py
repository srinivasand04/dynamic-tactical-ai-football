"""Who is to blame, checked in GRF: does fixing the MOST-BLAMED defender cut the danger more than fixing a random one?

who_is_to_blame.py gave every defender in 999 real dangerous moments a blame (how much threat his best reachable spot
would remove, computed on the still picture). Here the moments are played out in GRF (team models, 12 s, same seeds):
    baseline     the real moment, no change
    fix_blamed   only the most-blamed defender starts at his best reachable spot
    fix_random   only one other outfield defender of the same team (random, fixed per moment) starts at HIS best spot
Each change is scored against 'no change' with the same seed: threat of the attacking team from 1 s on while it has the
ball (coach_board.threat_from), shots and xG. If blame is real, fix_blamed should cut threat more than fix_random.
Also: does the still-picture blame predict the GRF effect (rank correlation over moments)?

Moments: the top --n moments by blame (>= 1.5 %-points, and at least 1 point more than the next defender: a clear
case), at most --per-match per match, at least 20 s apart. GRF needs
the 2 s before each moment: a 3-minute stretch is cut where none exists (export_segment.py, new files in data/).
Resumable: runs already in outputs/blame/grf_runs.csv are skipped.

Output: outputs/blame/grf_moments.csv, grf_runs.csv, grf_summary.csv, grf_chart.png (new files only)
Usage : .venv\\Scripts\\python.exe scripts\\blame_grf_check.py --seeds 8 --procs 4          (about 20 min)
        .venv-analysis\\Scripts\\python.exe scripts\\blame_grf_check.py --report             (chart)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "blame")
L, W = 105.0, 68.0
SCEN = ["baseline", "fix_blamed", "fix_random"]
TAG = ""                                       # "_hold" with --hold: separate files, the first test stays as it is


# ---------------------------------------------------------------- picking the moments (no GRF needed)

def find_segment(match, t):
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "real", "processed", f"{match}_segment_*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "name_home" in z.files and z["time_s"][0] + 2.2 <= t <= z["time_s"][-1] - 12.5:
            return f
    return None


def ensure_segment(match, t):
    seg = find_segment(match, t)
    py = os.path.join(ROOT, ".venv-analysis", "Scripts", "python.exe")
    for back in (20, 10, 5, 3):                                     # a start where both teams have 11 players
        if seg:
            break
        subprocess.run([py, os.path.join(ROOT, "scripts", "export_segment.py"), "--match", match, "--start-s", str(int(t - back)),
                        "--minutes", "3"], cwd=ROOT, capture_output=True)
        seg = find_segment(match, t)
    return seg


def pick(n, per_match, seed=0):
    """Moments, the blamed and a random defender of each, with their offsets in the real (data) frame."""
    import settings
    from football_scorecard import SUFFIX
    df = pd.read_csv(os.path.join(OUT, "blame_moments.csv")).replace([np.inf, -np.inf], np.nan).dropna(subset=["blame_best"])
    top = df.loc[df.groupby(["match", "ks"])["blame_best"].idxmax()].copy()
    second = df.groupby(["match", "ks"])["blame_best"].apply(lambda s: s.nlargest(2).iat[-1] if len(s) > 1 else 0.0)
    top["gap"] = top["blame_best"].to_numpy() - second.loc[list(zip(top.match, top.ks))].to_numpy()
    # a clear case: one defender stands out (not a ball on the goal line where anyone in the cone would do)
    top = top[(top.blame_best >= 1.5) & (top.gap >= 1.0)].sort_values("blame_best", ascending=False)
    rng = np.random.default_rng(seed)
    rows, used = [], {}
    for r in top.itertuples():
        if len(rows) >= n:
            break
        if used.get(r.match, 0) >= per_match or any(m == r.match and abs(t - r.time_s) < 20 for m, t, *_ in [(x["match"], x["time_s"]) for x in rows]):
            continue
        others = df[(df.match == r.match) & (df.ks == r.ks) & (df.slot != r.slot) & (df.role != "keeper")]
        if others.empty:
            continue
        o = others.iloc[int(rng.integers(len(others)))]
        seg = ensure_segment(r.match, r.time_s)
        if seg is None:
            print(f"  skip {r.match} {r.time_s} s: no stretch with 11 v 11 before it", flush=True)
            continue
        g = np.load(settings.base(r.match) + f"_graphs{SUFFIX}.npz")
        pos = g["xy"][int(r.ks)]
        a0 = 0 if r.defending == "away" else 11                      # attacking team's slots in the graph
        flip = np.isfinite(pos[a0, 0]) and pos[a0, 0] > L / 2       # who_is_to_blame turned the pitch so the attack goes to x = 105
        d0 = 11 if r.defending == "away" else 0

        def offset(row):
            best = np.array([row.best_x, row.best_y]) if hasattr(row, "best_x") else np.array([row["best_x"], row["best_y"]])
            if flip:
                best = np.array([L, W]) - best
            return best - pos[d0 + int(row.slot if hasattr(row, "slot") else row["slot"])]
        db, dr = offset(r), offset(o)
        if max(np.hypot(*db), np.hypot(*dr)) > 12 or not np.isfinite([*db, *dr]).all():     # 7 m/s x 1.5 s + slack
            print(f"  skip {r.match} {r.time_s} s: offset too large ({np.hypot(*db):.0f} m / {np.hypot(*dr):.0f} m)", flush=True)
            continue
        rows.append({"moment": f"{r.match}_{int(r.ks)}", "match": r.match, "time_s": float(r.time_s), "segment": os.path.relpath(seg, ROOT),
                     "side": r.defending, "threat": r.threat,
                     "blamed": r.name, "blamed_shirt": int(r.shirt), "blamed_role": r.role, "blame_blamed": r.blame_best,
                     "blamed_dx": round(float(db[0]), 2), "blamed_dy": round(float(db[1]), 2),
                     "random": o["name"], "random_shirt": int(o["shirt"]), "random_role": o["role"], "blame_random": o["blame_best"],
                     "random_dx": round(float(dr[0]), 2), "random_dy": round(float(dr[1]), 2)})
        used[r.match] = used.get(r.match, 0) + 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- GRF (patched what-if engine)

HOLD = False


def _patch():
    import coach_board  # noqa: F401   (adds threat_play / on_ball to every run, keeps other scenarios as they are)
    import whatif as WI
    inner = WI.moment

    def moment(seg, start_s, scen, shift, move=None):
        if scen in ("fix_blamed", "fix_random"):
            pos, ball, vel, att, mirrored, i = inner(seg, start_s, "baseline", shift, None)
            k, d = move
            d = -np.asarray(d) if mirrored else np.asarray(d)          # GRF frame: home on the left
            pos[k] = np.clip(pos[k] + d, [0.5, 0.5], [L - 0.5, W - 0.5])   # all 3 frames: he arrives with his real speed
            return pos, ball, vel, att, mirrored, i
        return inner(seg, start_s, scen, shift, move)
    WI.moment = moment
    return WI


def jobs_for(mo, seeds, seconds, done):
    import whatif as WI
    out = []
    for r in mo.itertuples():
        z = np.load(os.path.join(ROOT, r.segment), allow_pickle=True)
        kb, _ = WI.find_player(z, f"{r.side}:{r.blamed_shirt}")
        kr, _ = WI.find_player(z, f"{r.side}:{r.random_shirt}")
        for s in SCEN:
            mv = None if s == "baseline" else ((kb, (r.blamed_dx, r.blamed_dy)) if s == "fix_blamed" else (kr, (r.random_dx, r.random_dy)))
            for seed in range(seeds):
                if (r.moment, s, seed) not in done:
                    out.append((r.moment, os.path.join(ROOT, r.segment), r.time_s, s, 3.0, seed, seconds, mv))
    return out


def job(j):
    import whatif as WI
    import coach_board as CB
    moment_id, *rest = j
    seg, start_s, scen, shift, seed, seconds, mv = rest
    CB._B["delta"], CB._B["hold"] = {}, False
    if HOLD and mv is not None:                     # the fixed player keeps his better spot for the whole run
        mirrored = WI.moment(seg, start_s, "baseline", shift)[4]
        d = -np.asarray(mv[1]) if mirrored else np.asarray(mv[1])
        CB._B["delta"], CB._B["hold"] = {str(mv[0]): d.tolist()}, True
    r = WI._job(tuple(rest))
    r["moment"] = moment_id
    return r


def init(dataset, hold):
    global HOLD
    HOLD = hold
    _patch()
    if hold:
        import coach_board as CB
        CB.HOLD_SCENS.update({"fix_blamed", "fix_random"})
    import whatif as WI
    WI._init(dataset)


# ---------------------------------------------------------------- report

def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def report(seeds_note=""):
    runs = pd.read_csv(os.path.join(OUT, f"grf_runs{TAG}.csv"))
    runs = runs[runs["error"].isna()] if "error" in runs else runs
    mo = pd.read_csv(os.path.join(OUT, f"grf_moments{TAG}.csv")).set_index("moment")
    per = runs.groupby(["moment", "scenario"])[["threat_play", "shots", "xg", "on_ball"]].mean()
    rows = []
    for m in per.index.get_level_values(0).unique():
        p = per.loc[m]
        if not set(SCEN) <= set(p.index):
            continue
        b = p.loc["baseline"]
        rows.append({"moment": m, "blame_blamed": mo.loc[m, "blame_blamed"], "blame_random": mo.loc[m, "blame_random"],
                     **{f"d_{k}_{s[4:]}": p.loc[s, k] - b[k] for s in SCEN[1:] for k in ("threat_play", "shots", "xg")}})
    d = pd.DataFrame(rows)
    d.round(4).to_csv(os.path.join(OUT, f"grf_per_moment{TAG}.csv"), index=False)
    out = []
    for k, lab in (("threat_play", "threat from 1 s (pp)"), ("shots", "shots per run"), ("xg", "xG per run")):
        sc = 100 if k == "threat_play" else 1
        a, r_ = d[f"d_{k}_blamed"] * sc, d[f"d_{k}_random"] * sc
        diff = a - r_
        out.append({"measure": lab, "fix_blamed": a.mean(), "fix_blamed_ci95": ci(a), "fix_random": r_.mean(), "fix_random_ci95": ci(r_),
                    "blamed_minus_random": diff.mean(), "ci95": ci(diff), "blamed_better_in": f"{int((diff < 0).sum())}/{len(diff)}"})
    sm = pd.DataFrame(out)
    both = pd.concat([d[["blame_blamed", "d_threat_play_blamed"]].set_axis(["blame", "d"], axis=1),
                      d[["blame_random", "d_threat_play_random"]].set_axis(["blame", "d"], axis=1)])
    rho = both.corr("spearman").iat[0, 1]
    sm.round(3).to_csv(os.path.join(OUT, f"grf_summary{TAG}.csv"), index=False)
    pd.set_option("display.width", 220)
    print(f"\n{len(d)} moments played out in GRF{seeds_note}; change vs 'no change' (same seeds), mean over moments +- 95%:\n")
    print(sm.round(3).to_string(index=False))
    print(f"\nstill-picture blame vs GRF threat change (rank corr., both defenders of every moment): {rho:.2f}  (negative = blame predicts the cut)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        r0 = sm.iloc[0]
        ax[0].bar(["fix most-blamed", "fix random defender"], [r0.fix_blamed, r0.fix_random], yerr=[r0.fix_blamed_ci95, r0.fix_random_ci95],
                  color=["#e0483b", "#b5b3ab"], capsize=6)
        ax[0].axhline(0, c="k", lw=0.8)
        ax[0].set_ylabel("change in threat vs no change (%-points), GRF")
        ax[0].set_title(f"Fixing one defender, {len(d)} real moments (blamed better in {r0.blamed_better_in})")
        ax[1].scatter(d.blame_blamed, d.d_threat_play_blamed * 100, c="#e0483b", label="most-blamed")
        ax[1].scatter(d.blame_random, d.d_threat_play_random * 100, c="#8a8a8a", label="random defender")
        ax[1].axhline(0, c="k", lw=0.8)
        ax[1].set_xlabel("blame from the still picture (%-points)")
        ax[1].set_ylabel("GRF change in threat (%-points)")
        ax[1].set_title(f"Does blame predict the GRF effect? rank corr. {rho:.2f}")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"grf_chart{TAG}.png"), dpi=110)
        print("wrote outputs/blame/grf_summary.csv, grf_per_moment.csv, grf_chart.png")
    except ImportError:
        print("chart: run --report in .venv-analysis (matplotlib)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--per-match", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--dataset", default="pff")
    ap.add_argument("--only", help="play only this moment id (smoke test)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--hold", action="store_true", help="the fixed player keeps his offset for the whole run (files *_hold)")
    a = ap.parse_args()
    global TAG
    TAG = "_hold" if a.hold else ""
    if a.report:
        report(" (hold)" if a.hold else "")
        return
    mp = os.path.join(OUT, f"grf_moments{TAG}.csv")
    if os.path.exists(mp):
        mo = pd.read_csv(mp)
    else:
        print("picking moments and cutting GRF stretches...", flush=True)
        mo = pick(a.n, a.per_match)
        mo.to_csv(mp, index=False)
    if a.only:
        mo = mo[mo.moment == a.only]
    print(f"{len(mo)} moments, {len(SCEN)} scenarios, {a.seeds} seeds", flush=True)
    _patch()
    runs_p = os.path.join(OUT, f"grf_runs{TAG}.csv")
    done_df = pd.read_csv(runs_p) if os.path.exists(runs_p) else pd.DataFrame(columns=["moment", "scenario", "seed"])
    done = set(zip(done_df["moment"], done_df["scenario"], done_df["seed"].astype(int)))
    jobs = jobs_for(mo, a.seeds, a.seconds, done)
    print(f"{len(jobs)} runs to play ({len(done)} already done), {a.procs} at a time", flush=True)
    t0, rows = time.time(), []
    with Pool(a.procs, initializer=init, initargs=(a.dataset, a.hold)) as pool:
        for k, r in enumerate(pool.imap_unordered(job, jobs, chunksize=2)):
            rows.append(r)
            if (k + 1) % 24 == 0 or k + 1 == len(jobs):
                pd.concat([done_df, pd.DataFrame(rows)], ignore_index=True).to_csv(runs_p, index=False)
                el = (time.time() - t0) / 60
                print(f"  {k + 1}/{len(jobs)} runs ({el:.1f} min, about {el / (k + 1) * (len(jobs) - k - 1):.0f} min left)", flush=True)
    report(f", {a.seeds} seeds")


if __name__ == "__main__":
    main()
