"""Stage A: turn every Key State into a graph, and add the time dimension between consecutive Key States.

One graph per Key State (a frame picked by extract_keystates.py):
  nodes  23 slots: 0-10 home players, 11-21 away players, 22 the ball  (a player slot is empty if nobody is there,
         e.g. a team down to 10 men)
  edges  three types, as yes/no matrices of shape (23, 23):
           team : two teammates                          (all pairs inside a team)
           opp  : two opponents closer than --opp-r metres  (marking / pressing)
           ball : the ball and its --ball-k nearest players
         or, with --mode delaunay, the Delaunay triangulation of the players (a parameter-free neighbour mesh),
         split into team / opp by whether the two players are on the same team
  node features  position, velocity, role, team, distance to the ball
Time between consecutive Key States:
  displacement of every node since the previous Key State, the time gap, and the sequence id
  (a temporal edge links slot i at one Key State to slot i at the next).

Input : common-format CSV, its <name>_adaptive_ks.csv (and <name>_ks_features.csv for sequence ids, if present)
Output: <name>_graphs.npz (<name>_graphs_delaunay.npz with --mode delaunay), and a printed table of graph statistics

Usage :  python scripts\\build_graphs.py --csv data\\real\\processed\\J03WMX.csv.gz data\\simulated\\match_001.csv
         python scripts\\build_graphs.py --csv <list> --mode delaunay
"""
import argparse
import os
import numpy as np
import pandas as pd
import settings
from scipy.spatial import Delaunay

N, BALL = 23, 22
ROLES = {"GK": 0, "CB": 1, "LB": 2, "RB": 3, "DM": 4, "CM": 5, "LM": 6, "RM": 7, "AM": 8, "CF": 9}
PITCH_L, PITCH_W = 105.0, 68.0
TEAM = np.r_[np.zeros(11), np.ones(11), 2].astype(int)       # 0 home, 1 away, 2 ball


def base_name(csv):
    return csv.replace(".csv.gz", "").replace(".csv", "")


def load_key_state_frames(csv):
    name = base_name(csv)
    ks = pd.read_csv(name + "_adaptive_ks.csv")
    cols = ["frame", "team", "player_id", "role", "x_m", "y_m", "vx_ms", "vy_ms", "ball_x_m", "ball_y_m"]
    d = pd.read_csv(csv, usecols=cols)
    ball_all = d.drop_duplicates("frame").set_index("frame")[["ball_x_m", "ball_y_m"]].sort_index()   # every frame, to get the ball's velocity
    d = d[d["frame"].isin(ks["frame"])]

    frames = ks["frame"].to_numpy()
    K = len(frames)
    xy = np.full((K, N, 2), np.nan)
    vel = np.zeros((K, N, 2))
    role = np.full((K, N), -1)

    fi = np.searchsorted(frames, d["frame"].to_numpy())
    slot = np.where(d["team"].to_numpy() == "home", d["player_id"].to_numpy(), 11 + d["player_id"].to_numpy())
    xy[fi, slot, 0], xy[fi, slot, 1] = d["x_m"].to_numpy(), d["y_m"].to_numpy()
    vel[fi, slot, 0], vel[fi, slot, 1] = np.nan_to_num(d["vx_ms"].to_numpy()), np.nan_to_num(d["vy_ms"].to_numpy())
    vel[:, :22] = np.clip(vel[:, :22], -12, 12)                   # the real tracking has rare jumps of hundreds of m/s
    role[fi, slot] = d["role"].map(ROLES).fillna(-1).to_numpy()

    b = d.drop_duplicates("frame").set_index("frame").loc[frames]
    xy[:, BALL, 0], xy[:, BALL, 1] = b["ball_x_m"].to_numpy(), b["ball_y_m"].to_numpy()

    # ball velocity: position now minus position 2 frames (0.2 s) earlier; the data has no ball velocity column
    ball_v = np.nan_to_num((ball_all.reindex(frames).to_numpy() - ball_all.reindex(frames - 2).to_numpy()) / 0.2)
    vel[:, BALL] = np.clip(ball_v, -40, 40)

    seq = np.zeros(K, int)
    feat = name + "_ks_features.csv"
    if os.path.exists(feat):
        seq = pd.read_csv(feat)["seq_id"].to_numpy()
    return ks, xy, vel, role, seq


def make_edges(xy, mode, opp_r, ball_k, team_k=0):
    K = len(xy)
    present = ~np.isnan(xy[:, :, 0])
    D = np.linalg.norm(xy[:, :, None, :] - xy[:, None, :, :], axis=3)           # (K, N, N), NaN where absent
    ply = present.copy()
    ply[:, BALL] = False
    both = ply[:, :, None] & ply[:, None, :] & ~np.eye(N, dtype=bool)[None]
    same = (TEAM[:, None] == TEAM[None, :])[None]

    if mode == "radius":
        adj_team = both & same
        if team_k and team_k > 0:                                                 # sparse: each player to its k nearest teammates
            Dt = np.where(both & same, np.nan_to_num(D, nan=1e9), 1e9)
            nearest_mates = np.argsort(Dt, axis=2)[:, :, :team_k]
            knn = np.zeros((K, N, N), bool)
            knn[np.arange(K)[:, None, None], np.arange(N)[None, :, None], nearest_mates] = True
            knn &= both & same                                                    # drop slots that pointed at nobody
            adj_team = knn | knn.transpose(0, 2, 1)                               # a link is two-way
        adj_opp = both & ~same & (np.nan_to_num(D, nan=1e9) < opp_r)
    else:                                                                         # delaunay mesh of the players
        mesh = np.zeros((K, N, N), bool)
        for k in range(K):
            idx = np.flatnonzero(ply[k])
            if len(idx) < 4:
                continue
            try:
                tri = Delaunay(xy[k, idx])
            except Exception:                                                     # degenerate geometry
                continue
            for s in tri.simplices:
                for a in range(3):
                    i, j = idx[s[a]], idx[s[(a + 1) % 3]]
                    mesh[k, i, j] = mesh[k, j, i] = True
        adj_team, adj_opp = mesh & same, mesh & ~same

    adj_ball = np.zeros((K, N, N), bool)
    Db = np.where(ply, D[:, BALL, :], np.inf)
    Db[~present[:, BALL]] = np.inf
    nearest = np.argsort(Db, axis=1)[:, :ball_k]
    for k in range(K):
        for j in nearest[k]:
            if np.isfinite(Db[k, j]):
                adj_ball[k, BALL, j] = adj_ball[k, j, BALL] = True
    return adj_team, adj_opp, adj_ball, D, present


def stats(name, ks, xy, adj_team, adj_opp, adj_ball, D, present):
    K = len(xy)
    ply = present.copy()
    ply[:, BALL] = False
    # compactness: mean distance of a player to the centre of its own team
    comp = []
    for t in (0, 1):
        idx = np.flatnonzero(TEAM == t)
        pts = xy[:, idx]
        cen = np.nanmean(pts, axis=1, keepdims=True)
        comp.append(np.nanmean(np.linalg.norm(pts - cen, axis=2), axis=1))
    comp = np.nanmean(comp, axis=0)
    # nearest opponent per player
    dn = []
    for t in (0, 1):
        a, b = np.flatnonzero(TEAM == t), np.flatnonzero(TEAM == 1 - t)
        dn.append(np.nanmin(np.where(np.isnan(D[:, a][:, :, b]), np.inf, D[:, a][:, :, b]), axis=2))
    dn = np.concatenate(dn, axis=1)
    dn[~np.isfinite(dn)] = np.nan
    # time: how much do the opponent edges change from one Key State to the next?
    inter = (adj_opp[1:] & adj_opp[:-1]).sum(axis=(1, 2))
    union = (adj_opp[1:] | adj_opp[:-1]).sum(axis=(1, 2))
    churn = 1 - np.where(union > 0, inter / np.maximum(union, 1), 1)
    move = np.linalg.norm(np.nan_to_num(xy[1:, :22] - xy[:-1, :22]), axis=2)
    both_present = present[1:, :22] & present[:-1, :22]
    disp = np.nanmean(np.where(both_present, move, np.nan), axis=1)
    gap = ks["gap_s"].to_numpy()[1:]
    return {"match": os.path.basename(name), "graphs": K,
            "opp_edges": round(adj_opp.sum(axis=(1, 2)).mean() / 2, 1),
            "team_compact_m": round(float(np.nanmean(comp)), 1),
            "nearest_opp_m": round(float(np.nanmean(dn)), 1),
            "edge_churn": round(float(np.nanmean(churn)), 2),
            "move_per_KS_m": round(float(np.nanmean(disp)), 1),
            "speed_m/s": round(float(np.nanmean(disp / gap)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=None, help="default: every match in settings.json that has Key States")
    ap.add_argument("--mode", choices=["radius", "delaunay"], default=settings.get("graph", "mode", default="radius"))
    ap.add_argument("--opp-r", type=float, default=settings.get("graph", "opp_radius_m", default=8.0))
    ap.add_argument("--ball-k", type=int, default=settings.get("graph", "ball_k", default=3))
    ap.add_argument("--team-k", type=int, default=settings.get("graph", "team_k", default=5), help="link each player to its k nearest teammates (0 = all teammates)")
    a = ap.parse_args()
    if not a.csv:
        a.csv = [settings.csv(m) for m in settings.available(settings.REAL + settings.SIM) if os.path.exists(settings.base(m) + "_adaptive_ks.csv")]

    rows = []
    for csv in a.csv:
        name = base_name(csv)
        ks, xy, vel, role, seq = load_key_state_frames(csv)
        adj_team, adj_opp, adj_ball, D, present = make_edges(xy, a.mode, a.opp_r, a.ball_k, a.team_k)

        dxy = np.zeros_like(xy)
        dxy[1:] = np.nan_to_num(xy[1:] - xy[:-1])                                  # time: displacement since the previous Key State
        dxy[1:][~(present[1:] & present[:-1])] = 0
        dist_ball = np.where(present, D[:, BALL, :], np.nan)

        out = name + ("_graphs.npz" if a.mode == "radius" else "_graphs_delaunay.npz")   # settings.graphs(m) reads the same name
        np.savez_compressed(out, frame=ks["frame"].to_numpy(), ks_id=ks["ks_id"].to_numpy(),
                            time_s=ks["time_s"].to_numpy(), gap_s=ks["gap_s"].fillna(0).to_numpy(), seq_id=seq,
                            xy=xy, vel=vel, role=role, present=present, team=TEAM, dist_ball=dist_ball, dxy_prev=dxy,
                            adj_team=adj_team, adj_opp=adj_opp, adj_ball=adj_ball)
        rows.append(stats(name, ks, xy, adj_team, adj_opp, adj_ball, D, present))
        print(f"wrote {out}  ({len(xy)} graphs, mode={a.mode})")

    print()
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
