"""Export real Key States as snapshots GRF can start from (stage 1: runs in the analysis environment).

For every Key State of a real match that has exactly 11 players per team on the pitch and normal play, store
  - the 22 player positions and the ball (metres, 105 x 68 pitch), players ordered GK first, then by role
  - each player's velocity at that moment (m/s)
  - the real positions for the next --horizon frames (10 frames/s = one GRF step), used to score a GRF rollout
The pitch is flipped when needed so the home team always defends the left goal, as in GRF.

Input : data/real/processed/<match>.csv.gz and <match>_adaptive_ks.csv
Output: data/real/processed/<match>_ks_states.npz

Usage :  python scripts\\export_ks_states.py --match J03WOH --horizon 100
"""
import argparse
import numpy as np
import pandas as pd

L, W, FPS = 105.0, 68.0, 10
ROLE_ORDER = ["GK", "CB", "LB", "RB", "DM", "CM", "LM", "RM", "AM", "CF"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="J03WOH")
    ap.add_argument("--dir", default="data/real/processed")
    ap.add_argument("--horizon", type=int, default=100, help="frames kept after each Key State (10 = 1 s)")
    ap.add_argument("--all-modes", action="store_true", help="also keep kickoffs, throw-ins and other stoppages")
    a = ap.parse_args()

    d = pd.read_csv(f"{a.dir}/{a.match}.csv.gz")
    frames = np.sort(d["frame"].unique())
    F = len(frames)
    pids = sorted(d["dfl_id"].unique())
    fi = np.searchsorted(frames, d["frame"].to_numpy())
    pi = pd.Index(pids).get_indexer(d["dfl_id"])
    P = np.full((F, len(pids), 2), np.nan, dtype=np.float32)
    P[fi, pi, 0] = d["x_m"].to_numpy()
    P[fi, pi, 1] = d["y_m"].to_numpy()
    who = d.drop_duplicates("dfl_id").set_index("dfl_id").loc[pids]
    team, role = who["team"].to_numpy(), who["role"].to_numpy()
    fr = d.drop_duplicates("frame").set_index("frame").loc[frames]
    ball = fr[["ball_x_m", "ball_y_m"]].to_numpy(dtype=np.float32)
    mode, owner = fr["game_mode"].to_numpy(), fr["ball_owner_team"].to_numpy()

    ks = pd.read_csv(f"{a.dir}/{a.match}_adaptive_ks.csv")
    H = a.horizon
    keep = {k: [] for k in ["ks_id", "frame", "time_s", "flip", "owner", "ball", "home", "away",
                            "vel_home", "vel_away", "role_home", "role_away"]}
    skipped = {"edge": 0, "gap": 0, "mode": 0, "not11": 0}
    for _, r in ks.iterrows():
        i = int(np.searchsorted(frames, r["frame"]))
        if i < 1 or i + H >= F:
            skipped["edge"] += 1
            continue
        if frames[i + H] - frames[i] != H or frames[i] - frames[i - 1] != 1:
            skipped["gap"] += 1
            continue
        if mode[i] != "normal" and not a.all_modes:
            skipped["mode"] += 1
            continue
        order = {}
        for t in ("home", "away"):
            ids = np.where((team == t) & ~np.isnan(P[i, :, 0]))[0]
            if len(ids) != 11:
                break
            ids = sorted(ids, key=lambda k: (ROLE_ORDER.index(role[k]) if role[k] in ROLE_ORDER else 99, pids[k]))
            if role[ids[0]] != "GK" or any(role[k] not in ROLE_ORDER for k in ids):
                break
            order[t] = ids
        if len(order) != 2:
            skipped["not11"] += 1
            continue
        flip = bool(P[i, order["home"][0], 0] > L / 2)               # home keeper on the right -> mirror the pitch
        def fix(arr):
            arr = arr.copy()
            if flip:
                arr[..., 0], arr[..., 1] = L - arr[..., 0], W - arr[..., 1]
            return arr
        for t in ("home", "away"):
            fut = np.stack([P[i + h, order[t]] for h in range(H + 1)], axis=1)       # (11, H+1, 2)
            vel = np.nan_to_num((P[i, order[t]] - P[i - 1, order[t]]) * FPS)         # (11, 2) m/s
            keep[t].append(fix(fut))
            keep["vel_" + t].append(-vel if flip else vel)
            keep["role_" + t].append(role[order[t]])
        keep["ball"].append(fix(ball[i:i + H + 1]))
        keep["ks_id"].append(int(r["ks_id"]))
        keep["frame"].append(int(r["frame"]))
        keep["time_s"].append(float(r["time_s"]))
        keep["flip"].append(flip)
        keep["owner"].append(owner[i])

    out = f"{a.dir}/{a.match}_ks_states.npz"
    np.savez_compressed(out, horizon=H, **{k: np.array(v) for k, v in keep.items()})
    print(f"{a.match}: {len(ks)} Key States, kept {len(keep['ks_id'])}, skipped {skipped} -> {out}")


if __name__ == "__main__":
    main()
