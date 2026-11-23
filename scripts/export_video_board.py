"""Video to board: a broadcast clip tracked by RF-DETR SoccerNet (track_eval.py --detector rfdetr_sn) as a viewer file.

Each tracked person becomes a player slot (team 0 -> slots 0-10, team 1 -> 11-21; a track keeps its slot while it lives, the
goalkeeper takes slot 0 / 11; referees are left out), the ball comes from the detector's ball boxes (held for gaps < 1 s).
Pitch positions come from our camera calibration (pitch_calib.py), centred metres -> viewer metres (0..105, 0..68).
Only the players the camera sees are on the board, so it is a partial picture of the team.
Shirt numbers are unknown from video: the label is the track number.

Input : outputs/vision/tracks_sn/<clip>_positions.csv, <clip>_ball.csv
Output: viewer/data/video_<clip>.json + its entry in viewer/data/index.json
--fill: off-camera players are filled in by centroid voting (impute_offscreen.py, open method offscreen-impute) and
        drawn as see-through ghosts (field "imp"); written as video_<clip>_filled.json, the plain files are kept.
Usage : python scripts\\export_video_board.py SNGS-116 SNGS-117 SNGS-118
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_viewer import write  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DM = 10.0


def fill_offscreen(t, px, py, max_gap=20.0):
    """Fill empty slots of players seen before (within max_gap s) with the centroid-voting guess. Returns the imp flags."""
    from impute_offscreen import OffscreenImputer
    imp = np.zeros(px.shape, int)
    oi = OffscreenImputer()
    for fi in range(len(t)):
        vis = {tm: {s: (px[fi, s] / DM, py[fi, s] / DM) for s in range(11 * tm, 11 * tm + 11) if px[fi, s] >= 0} for tm in (0, 1)}
        oi.update(float(t[fi]), vis)
        for tm in (0, 1):
            for s in range(11 * tm, 11 * tm + 11):
                ls = oi.last_seen.get((tm, s))
                if px[fi, s] >= 0 or ls is None or t[fi] - ls[1] > max_gap:
                    continue
                g = oi.guess(tm, s, float(t[fi]), vis[tm])
                if g is not None:
                    px[fi, s] = int(round(np.clip(g[0], 0, 105) * DM)); py[fi, s] = int(round(np.clip(g[1], 0, 68) * DM)); imp[fi, s] = 1
    return imp


def export(clip, src="outputs/vision/tracks_sn", dry=False, fill=False):
    os.chdir(ROOT)
    P = pd.read_csv(f"{src}/{clip}_positions.csv")
    Bf = f"{src}/{clip}_ball.csv"
    B = pd.read_csv(Bf) if os.path.exists(Bf) else pd.DataFrame(columns=["frame", "time_s", "x_m", "y_m"])
    P = P[P["team"].astype(str).isin(["0", "1", "0.0", "1.0"])].copy()
    P["team"] = P["team"].astype(float).astype(int)
    frames = sorted(P["frame"].unique())
    slot_of, free = {}, {0: list(range(1, 11)), 1: list(range(12, 22))}
    keeper_slot = {0: 0, 1: 11}
    last_seen = {}
    px = np.full((len(frames), 22), -1, int)
    py = np.full((len(frames), 22), -1, int)
    ids = np.full((len(frames), 22), -1, int)
    for fi, f in enumerate(frames):
        g = P[P["frame"] == f]
        live = set(g["track_id"])
        for tid in [t for t in slot_of if last_seen.get(t, -99) < fi - 12]:       # a track unseen for ~1 s frees its slot
            s = slot_of.pop(tid)
            if s in (0, 11):
                keeper_slot[0 if s == 0 else 1] = s
            else:
                free[0 if s < 11 else 1].append(s)
        for r in g.itertuples():
            if r.track_id not in slot_of:
                if getattr(r, "role", "player") == "goalkeeper" and keeper_slot[r.team] is not None:
                    slot_of[r.track_id] = keeper_slot[r.team]; keeper_slot[r.team] = None
                elif free[r.team]:
                    slot_of[r.track_id] = free[r.team].pop(0)
                else:
                    continue
            last_seen[r.track_id] = fi
            s = slot_of[r.track_id]
            ids[fi, s] = r.track_id
            if np.isfinite(r.x_m) and np.isfinite(r.y_m) and abs(r.x_m) < 60 and abs(r.y_m) < 40:
                px[fi, s] = int(round((r.x_m + 52.5) * DM)); py[fi, s] = int(round((r.y_m + 34.0) * DM))
        del live
    t = np.round(np.array([P.loc[P["frame"] == f, "time_s"].iloc[0] for f in frames]), 2)
    bx = np.full(len(frames), -1, int); by = np.full(len(frames), -1, int)
    if len(B):
        B = B.sort_values("frame")
        for fi, tt in enumerate(t):
            k = np.searchsorted(B["time_s"].to_numpy(), tt, side="right") - 1
            if k >= 0 and tt - B["time_s"].iloc[k] < 1.0 and np.isfinite(B["x_m"].iloc[k]):
                bx[fi] = int(round((B["x_m"].iloc[k] + 52.5) * DM)); by[fi] = int(round((B["y_m"].iloc[k] + 34.0) * DM))
    for a in (bx, by):                                               # no ball seen yet: put it at the first seen spot
        ok = np.where(a >= 0)[0]
        if len(ok):
            a[:ok[0]] = a[ok[0]]
            for i in range(ok[0] + 1, len(a)):
                if a[i] < 0:
                    a[i] = a[i - 1]
        else:
            a[:] = int(52.5 * DM) if a is bx else int(34 * DM)
    seen_raw = (px >= 0).sum(1)
    imp = fill_offscreen(t, px, py) if fill else None
    rosters, prev = [], None
    for fi in range(len(frames)):
        r = [int(x) for x in ids[fi]]
        if r != prev:
            rosters.append([fi, r]); prev = r
    F = len(frames)
    out = {"match": clip, "dataset": "video", "home": "Team A (video)", "away": "Team B (video)", "final_score": "",
           "fps": round(1 / np.median(np.diff(t)), 2) if F > 1 else 12.5, "t": t.tolist(),
           "px": px.ravel().tolist(), "py": py.ravel().tolist(), "bx": bx.tolist(), "by": by.tolist(), "bz": [0] * F,
           "live": [1] * F, "sh": [0] * F, "sa": [0] * F, "rosters": rosters, "names": {"home": {}, "away": {}},
           "clock": [[float(t[0]), 0.0]], "layers": []}
    if fill:
        out["imp"] = imp.ravel().tolist()
    seen = (px >= 0).sum(1)
    if dry:
        print(clip, F, "frames, players on board per frame", round(seen_raw.mean(), 1), "->", round(seen.mean(), 1), "ball frames", int((bx >= 0).sum()), "roster changes", len(rosters))
        return out
    if fill:
        write(out, f"video_{clip}_filled.json", f"Video to board, filled  |  SoccerNet clip {clip} (RF-DETR SoccerNet tracking; {seen_raw.mean():.0f} "
                                                f"players in camera on average, off-camera players guessed by centroid voting -> {seen.mean():.0f}; guesses are ghosts)")
        return
    write(out, f"video_{clip}.json", f"Video to board  |  SoccerNet clip {clip} (broadcast video, RF-DETR SoccerNet tracking; "
                                     f"only players in camera, on average {seen.mean():.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--src", default="outputs/vision/tracks_sn")
    ap.add_argument("--dry", action="store_true", help="check only, write nothing")
    ap.add_argument("--fill", action="store_true", help="fill off-camera players (centroid voting), write video_<clip>_filled.json")
    a = ap.parse_args()
    for c in a.clips:
        export(c, a.src, a.dry, a.fill)


if __name__ == "__main__":
    main()
