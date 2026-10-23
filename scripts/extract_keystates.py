"""Adaptive Key-State extraction from player + ball geometry only (no football events).

Rule: start with the first frame as Key State K1. Scan forward; the first frame whose change
from the LAST selected Key State reaches theta becomes the next Key State.

    D_KS = alpha * D_absolute + beta * D_relative + gamma * D_ball          (all in metres)
    D_absolute = mean distance each player moved since the last Key State
    D_relative = mean change of the player-to-player distances (the arrangement of the players)
    D_ball     = distance the ball moved

Players are identified by a stable id (dfl_id for the real match, team + player_id for the simulator),
so a substitution does not look like a jump: only players present in both frames are compared.

Input : common-format CSV (data/real/processed/J03WMX.csv.gz or data/simulated/match_001.csv)
Output: one row per Key State

Usage :  python scripts\\extract_keystates.py --csv data\\simulated\\match_001.csv --theta 3
         python scripts\\extract_keystates.py --csv data\\real\\processed\\J03WMX.csv.gz --sweep 1 2 3 5 8
"""
import argparse
import numpy as np
import pandas as pd
import settings


def load(path):
    d = pd.read_csv(path)
    d["pid"] = d["dfl_id"] if "dfl_id" in d.columns else d["team"] + "_" + d["player_id"].astype(str)
    frames = np.sort(d["frame"].unique())
    pids = sorted(d["pid"].unique())
    fi = pd.Series(np.arange(len(frames)), index=frames)[d["frame"]].to_numpy()
    pi = pd.Series(np.arange(len(pids)), index=pids)[d["pid"]].to_numpy()
    P = np.full((len(frames), len(pids), 2), np.nan)          # (frames, players, xy), NaN = not on the pitch
    P[fi, pi, 0] = d["x_m"].to_numpy()
    P[fi, pi, 1] = d["y_m"].to_numpy()
    f = d.drop_duplicates("frame").set_index("frame").loc[frames]
    return frames, f["time_s"].to_numpy(), P, f[["ball_x_m", "ball_y_m"]].to_numpy()


def pair_matrix(p):
    """(N, 2) positions -> (N, N) distances, diagonal = NaN."""
    m = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(m, np.nan)
    return m


def parts(p, b, pk, bk, mk):
    """The three change measures between the current frame (p, b) and the Key State (pk, bk, mk)."""
    d_abs = np.nanmean(np.linalg.norm(p - pk, axis=1))
    d_rel = np.nanmean(np.abs(pair_matrix(p) - mk))
    d_ball = float(np.linalg.norm(b - bk))
    return d_abs, d_rel, d_ball


def extract(t, P, B, theta, alpha, beta, gamma):
    rows, k = [], 0
    pk, bk, mk = P[0], B[0], pair_matrix(P[0])
    rows.append((0, t[0], np.nan, np.nan, np.nan, np.nan, np.nan))
    for u in range(1, len(t)):
        d_abs, d_rel, d_ball = parts(P[u], B[u], pk, bk, mk)
        d = alpha * d_abs + beta * d_rel + gamma * d_ball
        if d >= theta:
            rows.append((u, t[u], t[u] - t[k], d_abs, d_rel, d_ball, d))
            k = u
            pk, bk, mk = P[u], B[u], pair_matrix(P[u])
    return pd.DataFrame(rows, columns=["frame_idx", "time_s", "gap_s", "d_abs", "d_rel", "d_ball", "d_ks"])


def summary(ks, n_frames, theta):
    g = ks["gap_s"].dropna()
    return {"theta": theta, "key_states": len(ks), "CR_%": round(100 * len(ks) / n_frames, 2),
            "gap_min_s": round(g.min(), 2), "gap_median_s": round(g.median(), 2),
            "gap_p90_s": round(g.quantile(0.9), 2), "gap_max_s": round(g.max(), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--theta", type=float, default=settings.get("key_states", "theta_m", default=3.0))       # metres of weighted change
    ap.add_argument("--alpha", type=float, default=settings.get("key_states", "alpha_players", default=0.4))       # player movement
    ap.add_argument("--beta", type=float, default=settings.get("key_states", "beta_arrangement", default=0.4))        # change of arrangement
    ap.add_argument("--gamma", type=float, default=settings.get("key_states", "gamma_ball", default=0.2))       # ball movement
    ap.add_argument("--sweep", type=float, nargs="*")         # try several thetas, print a table, write nothing
    a = ap.parse_args()
    assert abs(a.alpha + a.beta + a.gamma - 1) < 1e-9, "alpha + beta + gamma must be 1"

    frames, t, P, B = load(a.csv)
    print(f"{len(frames)} frames, {P.shape[1]} player ids, {t[-1] / 60:.1f} min")

    if a.sweep:
        table = [summary(extract(t, P, B, th, a.alpha, a.beta, a.gamma), len(frames), th) for th in a.sweep]
        print(pd.DataFrame(table).to_string(index=False))
        return

    ks = extract(t, P, B, a.theta, a.alpha, a.beta, a.gamma)
    ks.insert(0, "ks_id", np.arange(len(ks)))
    ks["frame"] = frames[ks["frame_idx"].to_numpy()]
    out = a.out or a.csv.replace(".csv.gz", "").replace(".csv", "") + "_adaptive_ks.csv"
    ks.drop(columns="frame_idx").round(3).to_csv(out, index=False)
    print(pd.DataFrame([summary(ks, len(frames), a.theta)]).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
