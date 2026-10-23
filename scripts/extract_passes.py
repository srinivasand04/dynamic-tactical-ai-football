"""Step 1 of the pass work: take the ball owner and the receiver from the real match EVENTS, not from a guess.

For every pass in the event file we find the tracking frame where the pass is played, and keep a snapshot of
the pitch at that moment: every player's position and velocity, who passed, who received, and whether it arrived.

How the moment is found
    the event clock gives a time; the tracking frame is found by looking up to 1 s around that time for the frame
    where the passer is closest to the ball (the clocks of the two files are not perfectly in step).
    If the passer is never within 3 m of the ball in that window, the pass is not used (the moment is uncertain).

What is checked and printed per match
    passes found, with a known receiver, used (passer within 3 m of the ball)
    guess agrees   : how often the old guess (nearest player of the owning team within 4 m) is the real passer
    team agrees    : how often the tracking's 'ball owning team' is the passer's team

Input : data/real/raw/<m>_events.csv.gz, data/real/processed/<m>.csv.gz
Output: data/real/processed/<m>_passes.npz  (one snapshot per pass; slots as in the graphs: 0-10 home, 11-21 away, 22 ball)

Usage :  python scripts\\extract_passes.py              (every real match in settings.json)
         python scripts\\extract_passes.py J03WMX
"""
import argparse

import numpy as np
import pandas as pd

import settings

L, W, FPS, BALL = 105.0, 68.0, 10, 22
WINDOW_S, MAX_SYNC_M, GUESS_M = 1.0, 3.0, 4.0


def load_tracking(m, frames):
    """Positions, velocities and DFL ids for the given frames, as arrays indexed [frame row, slot]."""
    d = pd.read_csv(settings.csv(m), usecols=["frame", "time_s", "team", "player_id", "x_m", "y_m", "vx_ms", "vy_ms",
                                               "ball_x_m", "ball_y_m", "ball_owner_team", "dfl_id"])
    d = d[d["frame"].isin(frames)]
    row = {f: i for i, f in enumerate(frames)}
    n = len(frames)
    xy, vel = np.full((n, 23, 2), np.nan), np.full((n, 23, 2), np.nan)
    ids = np.full((n, 23), "", dtype=object)
    r = d["frame"].map(row).to_numpy()
    s = (d["player_id"] + np.where(d["team"] == "away", 11, 0)).to_numpy()
    xy[r, s] = d[["x_m", "y_m"]].to_numpy()
    vel[r, s] = d[["vx_ms", "vy_ms"]].to_numpy()
    ids[r, s] = d["dfl_id"].to_numpy()
    f = d.drop_duplicates("frame")
    fr = f["frame"].map(row).to_numpy()
    xy[fr, BALL] = f[["ball_x_m", "ball_y_m"]].to_numpy()
    owner = np.full(n, "none", dtype=object)
    owner[fr] = f["ball_owner_team"].to_numpy()
    return xy, np.nan_to_num(vel), ids, owner


def to_metres(x, y, near):
    """Event coordinates are 0..1; the side the team attacks is not fixed, so try both directions and keep the one
    that puts the pass start next to the passer in the tracking data."""
    a = np.array([x * L, (1 - y) * W])
    b = np.array([(1 - x) * L, y * W])
    return (a, False) if np.linalg.norm(a - near) <= np.linalg.norm(b - near) else (b, True)


def guess_owner(xy, slot_team):
    """The old guess (bo_graph.py): the player of the owning team nearest the ball, if within 4 m."""
    mates = [i for i in range(22) if (i < 11) == (slot_team == 0) and not np.isnan(xy[i, 0])]
    if not mates or np.isnan(xy[BALL, 0]):
        return -1
    d = np.linalg.norm(xy[mates] - xy[BALL], axis=1)
    return mates[int(np.argmin(d))] if d.min() <= GUESS_M else -1


def extract(m):
    e = pd.read_csv(f"data/real/raw/{m}_events.csv.gz")
    p = e[e["event_type"] == "PASS"].copy()
    d0 = pd.read_csv(settings.csv(m), usecols=["frame", "time_s", "period"]).drop_duplicates("frame")
    period2_start = float(d0.loc[d0["period"] == 2, "time_s"].min())
    p["time_s"] = pd.to_timedelta(p["timestamp"]).dt.total_seconds() + np.where(p["period_id"] == 2, period2_start, 0.0)
    last = int(d0["frame"].max())
    centre = np.clip(np.round(p["time_s"].to_numpy() * FPS).astype(int), 0, last)
    k = int(WINDOW_S * FPS)
    frames = np.unique(np.concatenate([np.clip(centre + j, 0, last) for j in range(-k, k + 1)]))
    xy_all, vel_all, ids_all, owner_all = load_tracking(m, frames)
    row = {f: i for i, f in enumerate(frames)}

    out = {key: [] for key in ["frame", "time_s", "xy", "vel", "passer", "receiver", "complete", "set_piece",
                               "sync_m", "guess", "track_owner", "end_xy", "start_xy"]}
    n_known = 0
    for c, ev in zip(centre, p.itertuples()):
        n_known += isinstance(ev.receiver_player_id, str)
        best = None
        for f in range(max(c - k, 0), min(c + k, last) + 1):
            i = row[f]
            where = np.flatnonzero(ids_all[i] == ev.player_id)
            if len(where) == 0 or np.isnan(xy_all[i, BALL, 0]):
                continue
            dist = float(np.linalg.norm(xy_all[i, where[0]] - xy_all[i, BALL]))
            if best is None or dist < best[0]:
                best = (dist, f, int(where[0]))
        if best is None or best[0] > MAX_SYNC_M:
            continue
        dist, f, passer = best
        i = row[f]
        rec = np.flatnonzero(ids_all[i] == ev.receiver_player_id) if isinstance(ev.receiver_player_id, str) else []
        start, flipped = to_metres(ev.coordinates_x, ev.coordinates_y, xy_all[i, passer])
        end = (np.array([(1 - ev.end_coordinates_x) * L, ev.end_coordinates_y * W]) if flipped
               else np.array([ev.end_coordinates_x * L, (1 - ev.end_coordinates_y) * W]))
        out["frame"].append(f)
        out["time_s"].append(f / FPS)
        out["xy"].append(xy_all[i])
        out["vel"].append(vel_all[i])
        out["passer"].append(passer)
        out["receiver"].append(int(rec[0]) if len(rec) else -1)
        out["complete"].append(ev.result == "COMPLETE")
        out["set_piece"].append(isinstance(ev.set_piece_type, str))
        out["sync_m"].append(dist)
        out["guess"].append(guess_owner(xy_all[i], 0 if passer < 11 else 1))
        out["track_owner"].append(owner_all[i])
        out["start_xy"].append(start)
        out["end_xy"].append(end)

    arr = {key: np.array(v) for key, v in out.items()}
    arr["present"] = ~np.isnan(arr["xy"][..., 0])
    arr["team"] = np.array([0] * 11 + [1] * 11 + [2])
    np.savez_compressed(settings.base(m) + "_passes.npz", **arr)

    side = np.where(arr["passer"] < 11, "home", "away")
    open_play = ~arr["set_piece"]
    return {"match": m, "passes": len(p), "receiver_known": n_known, "used": len(arr["passer"]),
            "open_play": int(open_play.sum()), "complete_%": round(100 * arr["complete"].mean(), 1),
            "sync_m": round(float(np.median(arr["sync_m"])), 2),
            "guess_agrees_%": round(100 * float((arr["guess"] == arr["passer"])[open_play].mean()), 1),
            "guess_none_%": round(100 * float((arr["guess"] == -1)[open_play].mean()), 1),
            "team_agrees_%": round(100 * float((arr["track_owner"] == side)[open_play].mean()), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matches", nargs="*", default=settings.available(settings.REAL))
    a = ap.parse_args()
    rows = []
    for m in a.matches:
        rows.append(extract(m))
        print(f"  {m} done", flush=True)
    t = pd.DataFrame(rows).set_index("match")
    print("\nPASSES FROM THE EVENT DATA\n")
    print(t.to_string())
    print("\n  used           = passes where the passer is within 3 m of the ball at the matched frame")
    print("  sync_m         = median distance passer - ball at that frame (small = the two clocks line up)")
    print("  guess_agrees_% = open-play passes where the OLD guess (nearest player within 4 m) is the real passer")
    print("  guess_none_%   = open-play passes where the old guess found nobody (so no graph would have been drawn)")
    print("  team_agrees_%  = open-play passes where the tracking's 'ball owning team' is the passer's team")
    print("\nsaved data/real/processed/<match>_passes.npz")


if __name__ == "__main__":
    main()
