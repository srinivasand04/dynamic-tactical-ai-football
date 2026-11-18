"""Weak spot 3: what-if on MORE moments and with PAIRS of changes.

The first batch (whatif_batch.py) used 3 moments per match (20 in all) and one change at a time. Here:
  * more moments: --per-match 6 from the same 3-minute segments (up to 42), so the top 5 can be re-checked;
  * pairs: two defensive changes at once (whatif.py takes "a+b"), e.g. double_team+man_marking. For each pair the summary
    also gives the sum of the two single effects on the same moments: pair better than the sum = the changes help each
    other, worse = they get in each other's way.
Every run has a 'no change' twin with the same seed, as before. The first batch's files are not touched.

Output: outputs/whatif/pairs_runs.csv (saved as it goes; a stopped run resumes), pairs_moments.csv, pairs_summary.csv, pairs_chart.png
Usage : .venv\\Scripts\\python.exe scripts\\whatif_pairs.py --seeds 10 --procs 4          (GRF, a few hours)
        .venv-analysis\\Scripts\\python.exe scripts\\whatif_pairs.py --report             (summary and chart only)
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import whatif as WI  # noqa: E402
import whatif_batch as B  # noqa: E402

OUT = WI.OUT
SINGLES = ["double_team", "team_press", "press", "man_marking", "keeper_off_line"]
PAIRS = ["double_team+man_marking", "team_press+man_marking", "double_team+keeper_off_line", "team_press+keeper_off_line",
         "press+man_marking", "man_marking+keeper_off_line"]
SCEN = ["baseline"] + SINGLES + PAIRS

_describe = WI.describe
WI.describe = lambda s, shift: " + ".join(_describe(p, shift) for p in s.split("+"))     # names for pairs in B.summary
WI.DEFENCE = tuple(WI.DEFENCE) + tuple(PAIRS)


def synergy(df, sm):
    """Pair effect vs the sum of its two single effects, per moment (mean over seeds), on max threat."""
    df = df[df["error"].isna()] if "error" in df else df
    per = df.groupby(["moment", "scenario"])["max_threat"].mean().unstack()
    rows = []
    for p in PAIRS:
        a, b = p.split("+")
        if not {p, a, b, "baseline"} <= set(per.columns):
            continue
        d = per[[p, a, b, "baseline"]].dropna()
        pair = d[p] - d["baseline"]
        both = (d[a] - d["baseline"]) + (d[b] - d["baseline"])
        rows.append({"scenario": p, "pair_effect": pair.mean(), "sum_of_singles": both.mean(),
                     "pair_minus_sum": (pair - both).mean(), "pair_minus_sum_ci95": WI.ci(pair - both)})
    return sm.merge(pd.DataFrame(rows), on="scenario", how="left") if rows else sm


def chart(sm, path, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sm = sm.sort_values("d_max_threat")
    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(sm))
    d, e = sm["d_max_threat"].to_numpy(), sm["d_max_threat_ci95"].to_numpy()
    col = np.where(np.abs(d) <= e, "#b5b3ab", np.where(d < 0, "#1baf7a", "#e0483b"))
    hatch = ["//" if "+" in s else "" for s in sm["scenario"]]
    for i in range(len(sm)):
        ax.barh(y[i], d[i], xerr=e[i], color=col[i], hatch=hatch[i], capsize=3, edgecolor="#555")
    ax.set_yticks(y, sm["what_changed"])
    ax.axvline(0, c="k", lw=0.8)
    ax.set_xlabel("change in max threat vs no change (negative = good for the defence), 95% interval")
    ax.set_title(f"Defensive changes on {n} real moments: single (plain) and pairs (hatched)")
    fig.tight_layout()
    fig.savefig(path, dpi=105)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-match", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--scenarios", nargs="+", default=SCEN)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    runs_p = os.path.join(OUT, "pairs_runs.csv")
    if not a.report:
        moments = []
        for m, seg in B.segments().items():
            for mo in B.pick_moments(seg, a.per_match, a.seconds):
                moments.append({"moment": f"{m}_{int(mo['start_s'])}", "match": m, **mo})
        mdf = pd.DataFrame(moments)
        mdf.to_csv(os.path.join(OUT, "pairs_moments.csv"), index=False)
        print(f"{len(mdf)} moments, {len(a.scenarios)} scenarios, {a.seeds} seeds", flush=True)
        done = pd.read_csv(runs_p) if os.path.exists(runs_p) else pd.DataFrame(columns=["moment", "scenario", "seed"])
        have = set(zip(done["moment"], done["scenario"], done["seed"].astype(int)))
        jobs = [(r.moment, os.path.join(WI.ROOT, r.segment), r.start_s, s, seed, a.seconds)
                for r in mdf.itertuples() for s in a.scenarios for seed in range(a.seeds) if (r.moment, s, seed) not in have]
        print(f"{len(jobs)} runs to play ({len(have)} already done), {a.procs} at a time", flush=True)
        t0, rows = time.time(), []
        with Pool(a.procs, initializer=WI._init, initargs=("pff",)) as pool:
            for k, r in enumerate(pool.imap_unordered(B.job, jobs, chunksize=2)):
                rows.append(r)
                if (k + 1) % 50 == 0 or k + 1 == len(jobs):
                    pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(runs_p, index=False)
                    el = (time.time() - t0) / 60
                    print(f"  {k + 1}/{len(jobs)} runs ({el:.1f} min, about {el / (k + 1) * (len(jobs) - k - 1):.0f} min left)", flush=True)
    df = pd.read_csv(runs_p)
    sm = synergy(df, B.summary(df))
    # all changes here are defensive: threat going up in a moment means the change made it worse there
    sm = sm.rename(columns={"max_threat_helped_in": "made_worse_in"})
    sm.round(4).to_csv(os.path.join(OUT, "pairs_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    cols = ["what_changed", "verdict", "moments", "runs", "d_max_threat", "d_max_threat_ci95", "made_worse_in",
            "pair_minus_sum", "pair_minus_sum_ci95"]
    print(sm[[c for c in cols if c in sm]].round(4).to_string(index=False))
    try:
        chart(sm, os.path.join(OUT, "pairs_chart.png"), df["moment"].nunique())
        print("wrote outputs/whatif/pairs_summary.csv, pairs_chart.png")
    except ImportError:
        print("chart: run with --report in .venv-analysis (matplotlib)")


if __name__ == "__main__":
    main()
