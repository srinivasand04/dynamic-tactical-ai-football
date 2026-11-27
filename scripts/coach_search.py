"""Search for coach runs that score: many real moments x attacking players x target spots, a few seeds each.

For every real moment where a team has the ball with time on it (carrier 7 m+ from the nearest opponent, ball slow, in the
middle third or higher), every attacking outfield player ahead of the ball gets two candidate instructions (coach_run.py):
    behind   run 8 m beyond the deepest defender, in his own channel (kept inside the width of the box)
    box      run to the penalty spot area, between his channel and the middle
Each candidate is played with the same seeds as its moment's 'no change' runs (shared), so goals and threat are paired.
Output: outputs/coach_run/search_runs.csv (every run) and search_top.csv (candidates ranked by extra goals, then threat).
Only a screen: the chosen scenarios are re-run with more seeds by coach_run.py, which gives the numbers that count.

Usage : .venv\\Scripts\\python.exe scripts\\coach_search.py [--seeds 8] [--procs 4] [--matches J03WOH J03WMX ...]
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

os.environ.setdefault("GRF_PACE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import whatif as WI  # noqa: E402
import coach_run as C  # noqa: E402

L, W, FPS = WI.L, WI.W, WI.FPS
OUT = C.OUT


def moments(matches, gap_s=20.0):
    """(segment, time) of real moments with a free carrier; at most one per gap_s seconds per match."""
    out = []
    for m in matches:
        last = -1e9
        for f in sorted(glob.glob(os.path.join(C.ROOT, "data", "real", "processed", f"{m}_segment_*.npz")),
                        key=lambda p: int(p.split("_")[-1][:-4])):
            z = np.load(f, allow_pickle=True)
            if "name_home" not in z.files:
                continue
            t = z["time_s"]
            H, A, B = z["home"].astype(float), z["away"].astype(float), z["ball"].astype(float)
            for i in range(25, len(t) - 125, 5):
                if t[i] - last < gap_s:
                    continue
                o = str(z["owner"][i])
                if o not in ("home", "away") or not z["live"][i]:
                    continue
                P, O = (H, A) if o == "home" else (A, H)
                sp = np.hypot(*(B[i] - B[i - 1])) * 10
                dc, do = np.hypot(*(P[i] - B[i]).T), np.hypot(*(O[i] - B[i]).T)
                gk = np.nanmean(P[i - 20:i, 0, 0])
                bx = B[i, 0] if gk < L / 2 else L - B[i, 0]                      # ball in the attacking frame
                if np.nanmin(dc) > 1.5 or np.nanmin(do) < 7 or sp > 4 or not (40 < bx < 80):
                    continue
                j = i
                while j < len(t) - 1 and str(z["owner"][j]) in (o, "None", "nan", ""):
                    j += 1
                if (j - i) / 10 < 2:
                    continue
                out.append((f, float(t[i])))
                last = t[i]
    return out


def candidates(seg, t):
    pos, ball, _, att, mirrored, i = WI.moment(seg, t, "baseline", 3.0)
    z = np.load(seg, allow_pickle=True)
    own = list(range(11)) if att == "home" else list(range(11, 22))
    opp = list(range(11, 22)) if att == "home" else list(range(11))
    tf = (lambda p: p) if att == "home" else WI.flip                           # GRF frame -> attacking frame
    P = {k: tf(pos[k, -1]) for k in range(22)}
    b = tf(ball[-1])
    line = max(P[k][0] for k in opp[1:])                                       # deepest outfield defender
    carrier = min(own, key=lambda k: np.hypot(*(P[k] - b)))
    out = []
    for k in own[1:]:
        if k == carrier or P[k][0] < max(b[0] - 5, 45) or P[k][0] > line + 1:
            continue
        x, y = P[k]
        for kind, to in (("behind", (min(line + 8, 99.0), float(np.clip(y, 20, 48)))),
                         ("box", (94.0, float(np.clip((y + 34) / 2, 26, 42))))):
            d = np.hypot(to[0] - x, to[1] - y)
            if 10 <= d <= 32:
                name = str(z["name_home" if k < 11 else "name_away"][k % 11])
                out.append({"seg": seg, "t": t, "slot": k, "who": name, "kind": kind, "to_x": round(to[0], 1), "to_y": round(to[1], 1),
                            "dist": round(d, 1), "att": att})
    return out


def _init(dataset):
    C._init(dataset, None, (0.0, 0.0), True)


def _job(j):
    cid, seg, t, slot, to, seed, seconds = j
    C._R.update(slot=slot, to=to, **{"pass": True})
    try:
        r = C.play(seg, t, "baseline" if slot is None else "run", seed, seconds, WI._M["models"])[0]
    except Exception as e:
        r = {"error": repr(e)}
    r.update(cid=cid, seed=seed)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="+", default=["J03WOH", "J03WMX", "J03WPY", "J03WQQ", "J03WR9", "J03WOY"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--max-moments", type=int, default=24)
    ap.add_argument("--dataset", default="pff")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ms = moments(a.matches)
    rng = np.random.default_rng(0)
    if len(ms) > a.max_moments:
        ms = [ms[i] for i in sorted(rng.choice(len(ms), a.max_moments, replace=False))]
    cands = [c for seg, t in ms for c in candidates(seg, t)]
    cands = [dict(c, cid=f"c{n}") for n, c in enumerate(cands)]
    print(f"{len(ms)} moments, {len(cands)} candidate runs; {len(ms) * a.seeds + len(cands) * a.seeds} GRF runs", flush=True)
    jobs = [(f"b{n}", seg, t, None, None, s, a.seconds) for n, (seg, t) in enumerate(ms) for s in range(a.seeds)]
    mid = {(seg, t): f"b{n}" for n, (seg, t) in enumerate(ms)}
    jobs += [(c["cid"], c["seg"], c["t"], c["slot"], (c["to_x"], c["to_y"]), s, a.seconds) for c in cands for s in range(a.seeds)]
    t0 = time.time()
    rows = []
    with Pool(a.procs, initializer=_init, initargs=(a.dataset,)) as pool:
        for n, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=2), 1):
            rows.append(r)
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)} runs ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "search_runs.csv"), index=False)
    top = []
    for c in cands:
        b = R[R.cid == mid[(c["seg"], c["t"])]].set_index("seed")
        r = R[R.cid == c["cid"]].set_index("seed")
        s = b.index.intersection(r.index)
        if not len(s):
            continue
        top.append({**{k: c[k] for k in ("cid", "seg", "t", "who", "kind", "to_x", "to_y", "dist", "att")},
                    "goals_base": int(b.loc[s, "goal"].sum()), "goals_run": int(r.loc[s, "goal"].sum()),
                    "threat_base": round(float(b.loc[s, "max_threat"].mean()), 3), "threat_run": round(float(r.loc[s, "max_threat"].mean()), 3),
                    "touched": round(float(r.loc[s, "touched"].mean()), 2), "seeds": len(s)})
    T = pd.DataFrame(top)
    T["d_goals"] = T.goals_run - T.goals_base
    T["d_threat"] = T.threat_run - T.threat_base
    T = T.sort_values(["d_goals", "d_threat"], ascending=False)
    T["seg"] = T.seg.map(os.path.basename)
    T.to_csv(os.path.join(OUT, "search_top.csv"), index=False)
    print(T.head(15).to_string(index=False))
    print(f"[done in {time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
