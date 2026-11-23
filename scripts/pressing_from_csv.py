"""Pressing Intensity (unravelsports) for matches whose raw DFL XML is no longer on disk.

Uses unravelsports' own formulas (time_to_intercept, probability_to_intercept) on our stored tracking CSV
(data/real/raw/<match>_tracking.csv.gz), with the same settings as scripts/unravel_trial.py:
  5 frames/s, alive frames only, Savitzky-Golay velocities (players 7/1, ball 3/1), reaction 0.7 s, threshold 1.5 s,
  sigma 0.45, max speed 12 m/s, method "teams", ball_method "max" (ball merged with carrier = closest owning player < 25 m),
  team score per frame = mean over columns of the max over rows of the transposed ("pressing") PTI matrix,
  exactly as team_pressing() in unravel_trial.py.
--check <match> compares against the package's own output (outputs/unravel/pressing_<match>.csv) for that match.
Writes outputs/unravel/pressing_<match>.csv (same columns as unravel_trial.py), never overwrites the package outputs
unless --force.
Usage : .venv-unravel\\Scripts\\python.exe scripts\\pressing_from_csv.py J03WMX J03WN1 J03WOH J03WPY
        .venv-unravel\\Scripts\\python.exe scripts\\pressing_from_csv.py --check J03WOY
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from unravel.soccer.models.utils import probability_to_intercept, time_to_intercept

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "real", "raw")
OUT = os.path.join(ROOT, "outputs", "unravel")
FPS, STEP, HORIZON = 25, 5, 10.0


def smooth_vel(pos, dt, win):
    """pos: (T, ...) positions on the 5/s grid -> velocities, savgol-smoothed along time (NaNs left NaN)."""
    v = np.full_like(pos, np.nan)
    v[1:] = (pos[1:] - pos[:-1]) / dt
    v[0] = v[1] if len(v) > 1 else 0
    out = v.copy()
    if len(v) >= win:
        flat = v.reshape(len(v), -1)
        o = out.reshape(len(v), -1)
        for j in range(flat.shape[1]):
            col = flat[:, j]
            ok = ~np.isnan(col)
            if ok.sum() >= win:
                o[ok, j] = savgol_filter(col[ok], window_length=win, polyorder=1)
    return out


def match_press(mid):
    meta = json.load(open(os.path.join(RAW, f"{mid}_meta.json"), encoding="utf-8"))
    L, W = meta["pitch_length_m"], meta["pitch_width_m"]
    pl_ = pd.read_csv(os.path.join(RAW, f"{mid}_players.csv"))
    team_of = dict(zip(pl_.player_id, pl_.team_id))
    tr = pd.read_csv(os.path.join(RAW, f"{mid}_tracking.csv.gz"))
    pids = [c[:-2] for c in tr.columns if c.endswith("_x") and c.startswith("DFL-OBJ")]
    rows = []
    for per, g in tr.groupby("period_id"):
        g = g.iloc[::STEP]                                   # 5 frames/s, like sample_rate=5/25
        alive = (g.ball_state == "alive").to_numpy()
        P = np.stack([np.stack([g[f"{p}_x"].to_numpy() * L - L / 2, g[f"{p}_y"].to_numpy() * W - W / 2], -1) for p in pids], 1)
        B = np.stack([g.ball_x.to_numpy() * L - L / 2, g.ball_y.to_numpy() * W - W / 2, g.ball_z.fillna(0).to_numpy()], -1)
        VP, VB = smooth_vel(P, STEP / FPS, 7), smooth_vel(B, STEP / FPS, 3)
        own = g.ball_owning_team_id.to_numpy()
        fid, ts = g.frame_id.to_numpy(), pd.to_timedelta(g.timestamp).dt.total_seconds().to_numpy()
        teams = np.array([team_of.get(p) for p in pids])
        # orientation as in the package ("ball_owning"): the attacking team plays towards +x. A team attacks +x in this
        # period when its players stand on average left of the other team's.
        tids = [t for t in set(teams) if t is not None]
        mx = {t: np.nanmean(P[:, teams == t, 0]) for t in tids}
        attacks_plus = {t: mx[t] < np.mean([mx[o] for o in tids if o != t]) for t in tids}
        for k in range(len(g)):
            if not alive[k] or not isinstance(own[k], str):
                continue
            on = ~np.isnan(P[k, :, 0]) & ~np.isnan(VP[k, :, 0])
            att, dfn = np.where(on & (teams == own[k]))[0], np.where(on & (teams != own[k]) & (teams != None))[0]
            if len(att) == 0 or len(dfn) == 0 or np.isnan(B[k, 0]):
                continue
            dist = np.linalg.norm(P[k, att] - B[k, :2], axis=1)
            if dist.min() >= 25.0:
                continue                                      # no ball carrier -> package drops the frame
            carrier = int(np.argmin(dist))
            z = lambda a: np.concatenate([a, np.zeros((len(a), 1))], 1)
            p1 = np.vstack([z(P[k, att]), B[k][None]]); v1 = np.vstack([z(VP[k, att]), np.nan_to_num(VB[k])[None]])
            p2, v2 = z(P[k, dfn]), z(VP[k, dfn])
            tti = time_to_intercept(p1=p1, p2=p2, v1=v1, v2=v2, reaction_time=0.7, max_object_speed=12.0)
            bi = len(att)
            tti[:, carrier] = np.minimum(tti[:, carrier], tti[:, bi]); tti = np.delete(tti, bi, axis=1)
            pti = probability_to_intercept(tti, tti_sigma=0.45, tti_time_threshold=1.5).T
            sgn = 1.0 if attacks_plus.get(own[k], True) else -1.0
            rows.append((per, int(fid[k]), float(np.mean(np.max(pti, axis=0))), own[k], sgn * B[k, 0], sgn * B[k, 1], ts[k]))
    P = pd.DataFrame(rows, columns=["period_id", "frame_id", "press", "att", "ball_x", "ball_y", "t"])
    ev = pd.read_csv(os.path.join(RAW, f"{mid}_events.csv.gz"))
    ev = ev[ev.event_type == "SHOT"].assign(ts=lambda e: pd.to_timedelta(e.timestamp).dt.total_seconds())
    P["shot_next_10s"] = [((ev.period_id == p) & (ev.team_id == a) & (ev.ts > t) & (ev.ts <= t + HORIZON)).any()
                          for p, a, t in zip(P.period_id, P.att, P.t)]
    P["match"] = mid
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matches", nargs="*")
    ap.add_argument("--check", help="match that also has the package output; prints the agreement")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.check:
        mine = match_press(a.check)
        ref = pd.read_csv(os.path.join(OUT, f"pressing_{a.check}.csv"))
        j = mine.merge(ref, on=["period_id", "frame_id"], suffixes=("_csv", "_pkg"))
        r = np.corrcoef(j.press_csv, j.press_pkg)[0, 1]
        print(f"{a.check}: {len(mine):,} frames here, {len(ref):,} in package output, {len(j):,} shared")
        print(f"  correlation {r:.3f}   mean {j.press_csv.mean():.3f} vs {j.press_pkg.mean():.3f}   "
              f"mean abs diff {np.abs(j.press_csv - j.press_pkg).mean():.3f}   same attacking team {np.mean(j.att_csv == j.att_pkg):.1%}")
        print(f"  ball_x mean abs diff {np.abs(j.ball_x_csv - j.ball_x_pkg).mean():.2f} m   same zone sign {np.mean(np.sign(j.ball_x_csv) == np.sign(j.ball_x_pkg)):.1%}")
        mine.to_csv(os.path.join(OUT, f"check_pressing_{a.check}_from_csv.csv"), index=False)
        return
    for mid in a.matches:
        dst = os.path.join(OUT, f"pressing_{mid}.csv")
        if os.path.exists(dst) and not a.force:
            print(f"{mid}: {dst} exists, skipped (use --force)"); continue
        P = match_press(mid)
        P.to_csv(dst, index=False)
        print(f"{mid}: {len(P):,} frames, mean press {P.press.mean():.3f}, shots-next-10s {P.shot_next_10s.mean():.2%}", flush=True)


if __name__ == "__main__":
    main()
