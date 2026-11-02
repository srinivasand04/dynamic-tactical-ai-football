"""Stage A, part 2: add the PASS edges (the ball-owner graph) to every Key State graph, for the network.

For every Key State where somebody has the ball (the owning team's player nearest the ball, within 4 m, live play) the pass
options of that player are described by the models of the pass work, one row per teammate:

    pscore   4 numbers learned from real passes (fit_pass_model.py):
               the chance the passer picks him (model A), the chance the pass arrives (B), the danger of the arrival point (C,
               scaled x5), and the offside flag
    praw     the 13 plain numbers of pass_features.py that the models are made from (space, distance, goal, offside, lane, advance,
               the same at the arrival point, time margin, running); no learned model inside, so nothing to leak

Key States without a ball owner have no pass edges. The network (train_gnn.py --arch pass_score / pass_raw) adds them as a
fourth edge type between the ball owner and each teammate, in both directions, and the numbers above ride on the edge.

Note on 'pscore': the three models are fitted on all real matches (13 + 13 + 8 numbers), so an evaluation on a real match carries
a tiny amount of leakage; 'praw' has none. Compare the two.

Input : <name>_graphs.npz, the common-format CSV (who owns the ball), outputs/pass_model.json
Output: <name>_passedges.npz  (pbo, pmate, pmask, pscore, praw)
Usage :  python scripts\\build_pass_edges.py
"""
import argparse
import os

import numpy as np
import pandas as pd

import fit_pass_model as F
import pass_features as P
import settings

MAX_BALL_M, N_MATES, BALL = 4.0, P.N_MATES, 22


def softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


def build(m, models):
    w, comp, val = models
    g = np.load(settings.graphs(m))
    c = pd.read_csv(settings.csv(m), usecols=["frame", "ball_owner_team", "game_mode"]).drop_duplicates("frame").set_index("frame").loc[g["frame"]]
    owner, live = c["ball_owner_team"].to_numpy(), (c["game_mode"] == "normal").to_numpy()
    xy, vel, present, team = g["xy"], g["vel"], g["present"], g["team"]
    K = len(xy)
    pbo = np.full(K, -1, np.int64)
    pmate = np.full((K, N_MATES), -1, np.int64)
    pmask = np.zeros((K, N_MATES), bool)
    pscore = np.zeros((K, N_MATES, 4), np.float32)
    praw = np.zeros((K, N_MATES, len(P.COLUMNS)), np.float32)
    for k in range(K):
        if not live[k] or owner[k] not in ("home", "away") or np.isnan(xy[k, BALL, 0]):
            continue
        side = 0 if owner[k] == "home" else 1
        mates = [i for i in range(22) if team[i] == side and present[k, i]]
        if not mates:
            continue
        d = np.linalg.norm(xy[k, mates] - xy[k, BALL], axis=1)
        if d.min() > MAX_BALL_M:
            continue
        bo = mates[int(d.argmin())]
        cand = P.candidates(xy[k], vel[k], present[k], team, bo)
        if cand is None:
            continue
        n = min(len(cand["slots"]), N_MATES)
        I = P.to_inputs(cand["X"][:n])[None]                                   # (1, n, 13)
        s = I[..., F.idx(F.FULL)] @ w
        arrives = F.completion_prob(comp, I)[0]
        danger = F.value_at(val, cand["dest"][:n, 0], cand["dest"][:n, 1])
        pbo[k] = bo
        pmate[k, :n] = cand["slots"][:n]
        pmask[k, :n] = True
        pscore[k, :n] = np.stack([softmax(s[0]), arrives, 5 * danger, I[0, :, F.COL["offside"]]], axis=1)
        praw[k, :n] = I[0]
    np.savez_compressed(settings.base(m) + "_passedges.npz", pbo=pbo, pmate=pmate, pmask=pmask, pscore=pscore, praw=praw)
    return {"match": m, "key_states": K, "with_pass_edges_%": round(100 * float((pbo >= 0).mean()), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matches", nargs="*")
    a = ap.parse_args()
    todo = a.matches or [m for m in settings.available(settings.REAL + settings.SIM) if os.path.exists(settings.graphs(m))]
    models = F.from_json()
    rows = []
    for m in todo:
        rows.append(build(m, models))
        print(f"  {m} done", flush=True)
    t = pd.DataFrame(rows).set_index("match")
    t["group"] = [settings.variant(m) for m in t.index]
    print("\nPASS EDGES PER MATCH\n")
    print(t.to_string())
    print("\nPER GROUP\n")
    print(t.groupby("group")["with_pass_edges_%"].mean().round(1).to_string())
    print("\nsaved <base>_passedges.npz per match")


if __name__ == "__main__":
    main()
