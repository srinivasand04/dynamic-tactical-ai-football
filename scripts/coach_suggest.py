"""Weak spot 7: let the system SUGGEST coach moves instead of only testing the one the coach types in.

For one real moment, every outfield player of the defending team is moved, one at a time, 4 m in each of four directions
(forward, back, left, right; his team's view). GRF plays each move from the real moment with the team models, with the same
seeds as the 'no change' twin (as in whatif.py). Each move is scored by the change in the attacking team's max threat
(chance of a shot within 10 s, value_models.py) over the next 12 s, and the best 3 are listed with a 95% interval.
A move is only suggested when its interval stays below zero (a real reduction, not luck).

Output: outputs/whatif/coach_suggest_<match>_<start>.csv (runs) and _top.csv (ranked). Earlier what-if files are not touched.
Usage : .venv\\Scripts\\python.exe scripts\\coach_suggest.py                         (J03WOH, 3610 s: the coach moment; ~40 x 6 runs)
        .venv\\Scripts\\python.exe scripts\\coach_suggest.py --start-s 3610 --seeds 10 --step 4 --procs 4
then show the best one in the viewer:  .venv\\Scripts\\python.exe scripts\\viewer_mode.py coach --move NAME FORWARD LEFT
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


def job(j):
    key, seg, start_s, seed, seconds, move = j
    r = WI._job((seg, start_s, "baseline" if move is None else "move", 3.0, seed, seconds, move))
    r["key"] = key
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="J03WOH")
    ap.add_argument("--segment", default="data/real/processed/J03WOH_segment_3500.npz")
    ap.add_argument("--start-s", type=float, default=3610)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--step", type=float, default=4.0, help="metres each player is moved")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    seg = os.path.join(WI.ROOT, a.segment)
    z = np.load(seg, allow_pickle=True)
    _, _, _, att, _, _ = WI.moment(seg, a.start_s, "baseline", 3.0)
    d0 = 11 if att == "home" else 0
    side = "away" if att == "home" else "home"
    names = [str(n) for n in z[f"name_{side}"]]
    dirs = {"forward": (a.step, 0), "back": (-a.step, 0), "left": (0, a.step), "right": (0, -a.step)}
    cands = [(f"{names[k]}|{dname}", (d0 + k, names[k], f, l)) for k in range(1, 11) for dname, (f, l) in dirs.items()]
    tag = f"{a.match}_{int(a.start_s)}"
    runs_p = os.path.join(WI.OUT, f"coach_suggest_{tag}.csv")
    done = pd.read_csv(runs_p) if os.path.exists(runs_p) else pd.DataFrame(columns=["key", "seed"])
    have = set(zip(done["key"], done["seed"].astype(int)))
    jobs = [(key, seg, a.start_s, s, a.seconds, mv) for key, mv in [("no change", None)] + cands for s in range(a.seeds)
            if (key, s) not in have]
    print(f"defending team: {side}; {len(cands)} candidate moves x {a.seeds} seeds; {len(jobs)} runs to play", flush=True)
    if jobs:
        rows, t0 = [], time.time()
        with Pool(a.procs, initializer=WI._init, initargs=("pff",)) as pool:
            for k, r in enumerate(pool.imap_unordered(job, jobs, chunksize=2)):
                rows.append(r)
                if (k + 1) % 20 == 0 or k + 1 == len(jobs):
                    pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(runs_p, index=False)
                    el = (time.time() - t0) / 60
                    print(f"  {k + 1}/{len(jobs)} runs ({el:.1f} min, about {el / (k + 1) * (len(jobs) - k - 1):.0f} min left)", flush=True)
    df = pd.read_csv(runs_p)
    df = df[df["error"].isna()] if "error" in df else df
    base = df[df["key"] == "no change"].set_index("seed")["max_threat"]
    out = []
    for key, g in df[df["key"] != "no change"].groupby("key"):
        d = (g.set_index("seed")["max_threat"] - base).dropna()
        name, dname = key.split("|")
        f, l = dirs[dname]
        out.append({"player": name, "move": f"{a.step:g} m {dname}", "forward_m": f, "left_m": l, "seeds": len(d),
                    "d_max_threat": d.mean(), "ci95": WI.ci(d), "helped_in": f"{int((d < 0).sum())}/{len(d)}",
                    "suggest": bool(d.mean() + WI.ci(d) < 0)})
    top = pd.DataFrame(out).sort_values("d_max_threat").reset_index(drop=True)
    top.round(4).to_csv(os.path.join(WI.OUT, f"coach_suggest_{tag}_top.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"\n'no change' max threat: {base.mean():.3f}.  Best moves for the defending team (negative = less danger):")
    print(top.head(10).round(4).to_string(index=False))
    best = top[top["suggest"]].head(3)
    if len(best):
        print("\nSuggested (95% interval below zero):")
        for r in best.itertuples():
            print(f"  move {r.player} {r.move}: threat {r.d_max_threat:+.3f}  ->  viewer: "
                  f".venv\\Scripts\\python.exe scripts\\viewer_mode.py coach --move \"{r.player}\" {r.forward_m:g} {r.left_m:g}")
    else:
        print("\nNo move gives a clear reduction at this moment (all 95% intervals cross zero).")


if __name__ == "__main__":
    main()
