"""Cut a continuous stretch of a real match for GRF (stage 1, analysis environment).

Keeps every frame (10 per second) of the chosen stretch for the 22 players on the pitch at its start and for the ball,
in metres, mirrored so the home team defends the left goal as in GRF. A player who leaves the pitch inside the stretch
(substitution, red card) stays where he was last seen.

Input : data/real/processed/<match>.csv.gz
Output: data/real/processed/<match>_segment_<start_s>.npz

Usage :  python scripts\\export_segment.py --match J03WOH --minutes 3            (picks the busiest 3 minutes of open play)
         python scripts\\export_segment.py --match J03WOH --start-s 600 --minutes 2
"""
import argparse
import os
import numpy as np
import pandas as pd

L, W, FPS = 105.0, 68.0, 10
ROLE_ORDER = ["GK", "CB", "LB", "RB", "DM", "CM", "LM", "RM", "AM", "CF"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="J03WOH")
    ap.add_argument("--dir", default="data/real/processed")
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--start-s", type=float, default=None, help="start time in seconds; default = busiest stretch of open play")
    a = ap.parse_args()

    d = pd.read_csv(f"{a.dir}/{a.match}.csv.gz")
    frames = np.sort(d["frame"].unique())
    F, n = len(frames), int(a.minutes * 60 * FPS)
    fr = d.drop_duplicates("frame").set_index("frame").loc[frames]
    live = (fr["game_mode"].to_numpy() == "normal").astype(float)
    period = fr["period"].to_numpy()
    if a.start_s is None:                                              # most open play inside one half
        share = np.convolve(live, np.ones(n) / n, mode="valid")
        share[period[: len(share)] != period[n - 1: n - 1 + len(share)]] = -1
        i0 = int(np.argmax(share))
    else:
        i0 = int(np.searchsorted(fr["time_s"].to_numpy(), a.start_s))
    i1 = min(i0 + n, F)
    assert period[i0] == period[i1 - 1], "stretch crosses half time; pick another start"

    pids = sorted(d["dfl_id"].unique())
    fi = np.searchsorted(frames, d["frame"].to_numpy())
    pi = pd.Index(pids).get_indexer(d["dfl_id"])
    P = np.full((F, len(pids), 2), np.nan, dtype=np.float32)
    P[fi, pi, 0], P[fi, pi, 1] = d["x_m"].to_numpy(), d["y_m"].to_numpy()
    who = d.drop_duplicates("dfl_id").set_index("dfl_id").loc[pids]
    team, role = who["team"].to_numpy(), who["role"].to_numpy()

    names = {}
    pfile = f"data/real/raw/{a.match}_players.csv"                  # shirt numbers come from the data; names from the DFL player list
    if os.path.exists(pfile):
        names = pd.read_csv(pfile).set_index("player_id")["name"].to_dict()
    out = {}
    for t in ("home", "away"):
        ids = np.where((team == t) & ~np.isnan(P[i0, :, 0]))[0]
        assert len(ids) == 11, f"{t} has {len(ids)} players at the start"
        ids = sorted(ids, key=lambda k: (ROLE_ORDER.index(role[k]) if role[k] in ROLE_ORDER else 99, pids[k]))
        seg = pd.DataFrame(P[i0:i1][:, ids].reshape(i1 - i0, -1)).ffill().bfill().to_numpy().reshape(i1 - i0, 11, 2)
        out[t], out["role_" + t] = seg, role[ids]
        out["jersey_" + t] = who["jersey"].to_numpy()[ids].astype(int)
        out["name_" + t] = np.array([names.get(pids[k], "") for k in ids])
    flip = bool(out["home"][0, 0, 0] > L / 2)
    ball = fr[["ball_x_m", "ball_y_m"]].to_numpy(dtype=np.float32)[i0:i1]
    if flip:
        for k in ("home", "away", "ball"):
            arr = out[k] if k != "ball" else ball
            arr[..., 0], arr[..., 1] = L - arr[..., 0], W - arr[..., 1]
    start_s = float(fr["time_s"].to_numpy()[i0])
    path = f"{a.dir}/{a.match}_segment_{int(start_s)}.npz"
    np.savez_compressed(path, home=out["home"], away=out["away"], ball=ball, role_home=out["role_home"],
                        role_away=out["role_away"], time_s=fr["time_s"].to_numpy()[i0:i1], live=live[i0:i1] > 0,
                        owner=fr["ball_owner_team"].to_numpy()[i0:i1], flip=flip,
                        jersey_home=out["jersey_home"], jersey_away=out["jersey_away"],
                        name_home=out["name_home"], name_away=out["name_away"])
    print(f"{a.match}: {i1 - i0} frames ({(i1 - i0) / FPS / 60:.1f} min) from {start_s:.0f} s, "
          f"{live[i0:i1].mean() * 100:.0f}% open play, flipped={flip} -> {path}")


if __name__ == "__main__":
    main()
