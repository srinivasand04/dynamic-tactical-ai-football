"""Team on-ball decisions (Part A3): how each team uses the ball, learned from its real matches.

From the ball track of every match (the kick finder of grf_pass.py) we get each team's kicks: who kicked, where the ball went, who
got it. From them, per team:
  * RECEIVER CHOICE: a softmax over the ten teammates of the kicker, scored on six numbers per teammate (forward gain, distance,
    sideways distance, free space around him, how open the passing lane is, how far up the pitch he stands). Fitted per team
    (L2-regularised towards the all-teams weights) on completed passes. Checked on each team's held-out match: top-1 accuracy of the
    team's weights vs the all-teams weights vs 'nearest teammate'.
  * STYLE NUMBERS: how long a player keeps the ball before passing (median), the share of long passes (> 25 m), how far from goal
    the team shoots (80th percentile), passes per minute of possession.
These drive the ball carrier of team_agent.py in GRF.

Input : the common-format files of the dataset and its registry
Output: outputs/team_policy/<dataset>_decisions.json and a printed check
Usage :  python scripts\\team_decisions.py --dataset pff --teams Argentina France Croatia
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import settings
from grf_pass import detect_kicks

L, W = 105.0, 68.0
FEATS = ["forward_gain", "distance", "sideways", "space", "lane_open", "height_up_pitch"]


def match_arrays(m):
    """Players (22, F, 2) and ball (F, 2) on the match's own frame list, plus the frame numbers."""
    d = pd.read_csv(settings.csv(m), usecols=["frame", "team", "player_id", "x_m", "y_m", "ball_x_m", "ball_y_m"],
                    dtype={"frame": "int32", "team": "category", "player_id": "int8", "x_m": "float32", "y_m": "float32",
                           "ball_x_m": "float32", "ball_y_m": "float32"})
    frames = np.sort(d["frame"].unique())
    fi = np.searchsorted(frames, d["frame"].to_numpy())
    slot = np.where(d["team"].to_numpy() == "home", 0, 11) + d["player_id"].to_numpy()
    ok = slot < 22
    P = np.full((22, len(frames), 2), np.nan)
    P[slot[ok], fi[ok], 0], P[slot[ok], fi[ok], 1] = d["x_m"].to_numpy()[ok], d["y_m"].to_numpy()[ok]
    b = d.drop_duplicates("frame").set_index("frame").loc[frames][["ball_x_m", "ball_y_m"]].to_numpy()
    return P, b, frames


def candidate_features(P, t, kicker):
    """Six numbers for each teammate of the kicker at frame t, the kicker's team turned to attack towards x = 105."""
    own = range(0, 11) if kicker < 11 else range(11, 22)
    opp = list(range(11, 22)) if kicker < 11 else list(range(0, 11))
    flip = kicker >= 11                                         # the away team attacks towards x = 0 in the data
    pos = P[:, t].copy()
    if flip:
        pos = np.stack([L - pos[:, 0], W - pos[:, 1]], axis=1)
    k = pos[kicker]
    O = pos[opp]
    cands, X = [], []
    for j in own:
        if j == kicker or np.isnan(pos[j, 0]):
            continue
        p = pos[j]
        seg = p - k
        sl = max(np.dot(seg, seg), 1e-6)
        u = np.clip(((O - k) @ seg) / sl, 0, 1)                 # opponents' nearest points on the pass line
        lane = np.nanmin(np.linalg.norm(O - (k + u[:, None] * seg), axis=1))
        space = np.nanmin(np.linalg.norm(O - p, axis=1))
        X.append([(p[0] - k[0]) / 10, np.linalg.norm(seg) / 10, abs(seg[1]) / 10, min(space, 10) / 5, min(lane, 10) / 5, p[0] / L])
        cands.append(j)
    return cands, np.array(X)


def match_kicks(job):
    """All kicks of one match, sorted into the per-team lists (runs in a worker process)."""
    mid, home, away, teams, test = job
    part = {t: {"train": [], "test": [], "hold": [], "long": [], "shot_d": [], "kicks": 0, "poss_s": 0.0} for t in teams + ["other"]}
    P, b, frames = match_arrays(mid)
    cut = np.flatnonzero(np.diff(frames) > 1) + 1                 # detect kicks inside each unbroken stretch
    for s0, s1 in zip(np.r_[0, cut], np.r_[cut, len(frames)]):
        if s1 - s0 < 30:
            continue
        ks = detect_kicks(b[s0:s1], P[:, s0:s1])
        for i, k in enumerate(ks):
            side_team = home if k["kicker"] < 11 else away
            key = side_team if side_team in teams else "other"
            T = part[key]
            T["kicks"] += 1
            t = s0 + k["t"]
            x_goal = L if k["kicker"] < 11 else 0.0
            end = k["end_xy"]
            if (abs(end[0] - x_goal) < 3 or (end[0] > L) or (end[0] < 0)) and abs(end[1] - W / 2) < 12:
                T["shot_d"].append(float(np.hypot(b[t, 0] - x_goal, b[t, 1] - W / 2)))
                continue
            T["long"].append(k["len_m"] > 25)
            if i > 0 and ks[i - 1]["receiver"] == k["kicker"] and ks[i - 1]["same_team"]:
                T["hold"].append((k["t"] - ks[i - 1]["end"]) / 10)
                T["poss_s"] += (k["t"] - ks[i - 1]["t"]) / 10
            if k["same_team"] and k["recv_m"] < 3:
                cands, X = candidate_features(P[:, s0:s1], k["t"], k["kicker"])
                if k["receiver"] in cands and len(cands) >= 5:
                    held_out = key in test and test[key][0] == mid
                    T["test" if held_out else "train"].append((X, cands.index(k["receiver"])))
    return mid, home, away, part


def fit(samples, w0=None, l2=0.5, steps=400, lr=0.5):
    """Softmax over candidates; gradient descent; L2 towards w0 (the all-teams weights)."""
    w = np.zeros(len(FEATS)) if w0 is None else w0.copy()
    ref = np.zeros(len(FEATS)) if w0 is None else w0
    for _ in range(steps):
        g = l2 * (w - ref)
        for X, y in samples:
            s = X @ w
            p = np.exp(s - s.max())
            p /= p.sum()
            g += (X.T @ p - X[y]) / len(samples)
        w -= lr * g
    return w


def top1(samples, w):
    return float(np.mean([int(np.argmax(X @ w) == y) for X, y in samples])) if samples else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pff")
    ap.add_argument("--teams", nargs="+", default=["Argentina", "France", "Croatia"])
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    reg = settings.registry(a.dataset)
    reg = reg[reg["match_id"].isin(settings.dataset_matches(a.dataset))].sort_values("date")
    cfg_path = f"outputs/team_policy/{a.dataset}_config.json"
    test = {}
    if os.path.exists(cfg_path):                                 # the same held-out matches as the movement model
        test = {t: tuple(v) for t, v in json.load(open(cfg_path))["test_matches"].items()}
    else:
        for t in a.teams:                                        # same rule as team_policy.py: latest match, never shared
            rows = reg[(reg["home"] == t) | (reg["away"] == t)]
            for r in rows.iloc[::-1].itertuples():
                if r.match_id not in {m for m, _ in test.values()}:
                    test[t] = (r.match_id, "home" if r.home == t else "away")
                    break
    test_ids = {m for m, _ in test.values()}

    per_team = {t: {"train": [], "test": [], "hold": [], "long": [], "shot_d": [], "kicks": 0, "poss_s": 0.0} for t in a.teams + ["other"]}
    jobs = [(r.match_id, r.home, r.away, a.teams, test) for r in reg.itertuples()]
    with Pool(a.procs) as pool:                                  # matches in parallel; each returns its kicks sorted by team
        for mid, home, away, part in pool.imap_unordered(match_kicks, jobs):
            for key, T in part.items():
                for f in ("train", "test", "hold", "long", "shot_d"):
                    per_team[key][f] += T[f]
                per_team[key]["kicks"] += T["kicks"]
                per_team[key]["poss_s"] += T["poss_s"]
            print(f"  {mid} {home} v {away} done", flush=True)

    all_train = [s for T in per_team.values() for s in T["train"]]
    w_all = fit(all_train)
    out = {"features": FEATS, "all_teams": {"weights": w_all.tolist()}, "teams": {}}
    print("\nRECEIVER CHOICE: top-1 accuracy on each team's held-out match (completed passes)")
    print(f"{'team':12s} {'passes':>6s} | team weights  all-teams weights  nearest teammate  chance")
    for t in a.teams + ["other"]:
        T = per_team[t]
        w = fit(T["train"], w0=w_all) if T["train"] else w_all
        near = top1(T["test"], np.array([0, -1, 0, 0, 0, 0.0]))
        chance = float(np.mean([1 / len(X) for X, _ in T["test"]])) if T["test"] else np.nan
        if T["test"]:
            print(f"{t:12s} {len(T['test']):6d} | {top1(T['test'], w):11.2f}  {top1(T['test'], w_all):16.2f}  {near:16.2f}  {chance:6.2f}")
        out["teams"][t] = {"weights": w.tolist(), "n_train_passes": len(T["train"]),
                           "hold_s": float(np.median(T["hold"])) if T["hold"] else 1.0,
                           "long_share": float(np.mean(T["long"])) if T["long"] else 0.2,
                           "shot_dist_m": float(np.percentile(T["shot_d"], 80)) if len(T["shot_d"]) >= 3 else 22.0,
                           "passes_per_min_possession": float(60 * len(T["hold"]) / T["poss_s"]) if T["poss_s"] else 10.0,
                           "kicks": T["kicks"]}
    print("\nSTYLE NUMBERS")
    print(pd.DataFrame({t: {k: v for k, v in d.items() if k != "weights"} for t, d in out["teams"].items()}).T.round(2).to_string())
    print("\nRECEIVER WEIGHTS (" + ", ".join(FEATS) + ")")
    for t, d in out["teams"].items():
        print(f"  {t:12s} " + " ".join(f"{x:+.2f}" for x in d["weights"]))
    os.makedirs("outputs/team_policy", exist_ok=True)
    json.dump(out, open(f"outputs/team_policy/{a.dataset}_decisions.json", "w"), indent=1)
    print(f"\nwrote outputs/team_policy/{a.dataset}_decisions.json")


if __name__ == "__main__":
    main()
