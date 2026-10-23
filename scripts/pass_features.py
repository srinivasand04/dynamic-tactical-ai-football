"""Step 3 of the pass work: describe every pass option with MOVEMENT and TIMING, not only with a still picture.

For every pass (real: from the event data, extract_passes.py; simulated: from the ball-owner changes) and for every teammate
the passer could have chosen, this builds a row of numbers. Still-picture numbers first, then the new ones:

    still picture (what the old graph already knew)
        d_opp      distance from the teammate to his nearest opponent                  (m)
        d_bo       length of the pass                                                  (m)
        d_goal     distance from the teammate to the centre of the opponent goal       (m)
        offside    1 if the teammate is offside (the rule of bo_graph.py)
        lane       the old, still-picture lane score of bo_graph.py (1 = open)
        adv        how far the teammate is ahead of the passer, towards the goal       (m)
    movement (new)
        d_opp_f    the same space, but where he WILL be when the ball arrives: every player moves on with his own velocity
        d_goal_f, adv_f   goal distance and advance at the arrival point
        margin     TIME margin of the pass lane: for every opponent, the time he needs to reach a point on the pass line minus
                   the time the ball needs to get there (ball 15 m/s, opponent 6 m/s + 0.2 s reaction). Smallest value over
                   the line and the opponents. Negative = some opponent can get there first (the pass can be cut out); a slow
                   defender who is close but cannot arrive in time no longer blocks.
        run_fwd    the teammate's running speed towards the opponent goal (m/s): is he making a forward run?
        speed      his running speed (m/s)

Everything uses numbers that tracking gives: positions and velocities. The constants (ball speed 15 m/s, opponent speed
6 m/s, reaction 0.2 s, players slow down to half over the flight) are set at the top of this file.

Input : data/real/processed/<m>_passes.npz (extract_passes.py), data/simulated/.../<m>.csv
Output: <base>_passfeat.npz per match: for every pass X (10 teammates x 12 numbers), mask, dest (arrival point in the attacking
        direction), target (who really got the ball), complete, passer, frame, time_s
Usage :  python scripts\\pass_features.py                 (every real and simulated match in settings.json)
         python scripts\\pass_features.py J03WMX match_001
"""
import argparse
import os

import numpy as np
import pandas as pd

import bo_graph
import settings

L, W, FPS, N_MATES = 105.0, 68.0, 10, 10
VB, V_OPP, REACT, TAU_MAX, DAMP = 15.0, 6.0, 0.2, 3.0, 0.5
SIM_GAP_MIN_S, SIM_GAP_MAX_S = 0.2, 3.0
AIM_FRAMES, AIM_MIN_M, AIM_MAX_DEG = 5, 2.0, 20.0                          # failed pass: the ball's first 0.5 s decide who was meant

RAW = ["d_opp", "d_bo", "d_goal", "offside", "lane", "adv", "d_opp_f", "d_goal_f", "adv_f", "margin", "run_fwd", "speed"]
COLUMNS = ["open_space", "dist", "dist2", "goal", "offside", "lane", "adv",
           "open_space_f", "goal_f", "adv_f", "margin", "run_fwd", "speed"]
GROUPS = {"still picture": COLUMNS[:7], "arrival point": COLUMNS[7:10], "time margin": ["margin"], "running": ["run_fwd", "speed"]}


def to_inputs(X):
    """Raw numbers -> the inputs of the models (roughly -2..3, so the learned weights are comparable)."""
    d = {k: X[..., i] for i, k in enumerate(RAW)}
    dist = d["d_bo"] / 10
    cols = [np.minimum(d["d_opp"], 20) / 10, dist, dist ** 2, d["d_goal"] / 30, d["offside"], d["lane"], d["adv"] / 20,
            np.minimum(d["d_opp_f"], 20) / 10, d["d_goal_f"] / 30, d["adv_f"] / 20, np.clip(d["margin"], -1.5, 2.0),
            np.clip(d["run_fwd"], -4, 8) / 4, np.clip(d["speed"], 0, 9) / 6]
    return np.stack(cols, axis=-1)


def candidates(xy, vel, present, team, passer):
    """One row per teammate of the passer (slot order of the graph). None when no sensible graph exists."""
    g = bo_graph.build(xy, present, team, None, bo=passer)
    if g is None:
        return None
    mates = [e for e in g["edges"] if e["type"] == "teammate"]
    if not mates:
        return None
    side = g["side"]
    sgn = 1.0 if side == 0 else -1.0
    att = (lambda x: x) if side == 0 else (lambda x: L - x)
    goal = np.array([L if side == 0 else 0.0, W / 2])
    a = xy[passer]
    to = np.array([e["to"] for e in mates])
    opps = np.array(g["opps"])
    pos, v = xy[to], vel[to]
    d_bo = np.linalg.norm(pos - a, axis=1)
    tau = np.clip(d_bo / VB, 0.0, TAU_MAX)                                  # flight time of the ball to him
    hi = np.array([L, W])
    posf = np.clip(pos + v * (tau[:, None] * DAMP), 0, hi)                  # where he will be
    opos, ovel = xy[opps], vel[opps]
    oposf = np.clip(opos[None] + ovel[None] * (tau[:, None, None] * DAMP), 0, hi)   # where the opponents will be (n, o, 2)

    d_opp = np.linalg.norm(pos[:, None] - opos[None], axis=2).min(1)
    d_opp_f = np.linalg.norm(posf[:, None] - oposf, axis=2).min(1)
    adv = att(pos[:, 0]) - att(a[0])
    adv_f = att(posf[:, 0]) - att(a[0])

    u = np.linspace(0.1, 1.0, 10)                                           # 10 points on the pass line, towards where he will be
    q = a + u[None, :, None] * (posf - a)[:, None, :]
    t_ball = u[None, :] * np.linalg.norm(posf - a, axis=1)[:, None] / VB
    t_opp = np.linalg.norm(q[:, :, None, :] - opos[None, None], axis=3) / V_OPP + REACT
    margin = (t_opp - t_ball[:, :, None]).min(axis=(1, 2))

    X = np.stack([d_opp, d_bo, np.linalg.norm(pos - goal, axis=1), [float(e["offside"]) for e in mates],
                  [e["lane"] for e in mates], adv, d_opp_f, np.linalg.norm(posf - goal, axis=1), adv_f, margin,
                  sgn * v[:, 0], np.linalg.norm(v, axis=1)], axis=1)
    dest = np.stack([att(posf[:, 0]), posf[:, 1]], axis=1)
    return {"X": X, "dest": dest, "slots": to, "posf": posf, "a": a}


def pad(a, n=N_MATES):
    out = np.zeros((n,) + a.shape[1:], dtype=float)
    out[:len(a)] = a[:n]
    return out


def row_of(c, receiver, passer, complete, frame, time_s):
    slots = list(c["slots"])
    if receiver not in slots:
        return None
    n = min(len(slots), N_MATES)
    mask = np.zeros(N_MATES, bool)
    mask[:n] = True
    return {"X": pad(c["X"]), "dest": pad(c["dest"]), "mask": mask, "target": slots.index(receiver), "complete": bool(complete),
            "passer": passer, "frame": frame, "time_s": time_s}


def pack(rows, n_intercept=0):
    if not rows:
        raise SystemExit("no usable passes")
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["n_intercept"] = np.array(n_intercept)
    return out


def real_match(m):
    z = np.load(settings.base(m) + "_passes.npz", allow_pickle=True)
    b = pd.read_csv(settings.csv(m), usecols=["frame", "ball_x_m", "ball_y_m"]).drop_duplicates("frame").set_index("frame")
    ball = b.reindex(range(int(b.index.max()) + 1)).to_numpy()              # ball position of every frame
    rows = []
    for i in range(len(z["passer"])):
        p, r = int(z["passer"][i]), int(z["receiver"][i])
        if z["set_piece"][i] or r == p:
            continue
        known = r >= 0 and z["team"][r] == z["team"][p] and z["present"][i, r]
        if not known and z["complete"][i]:
            continue
        c = candidates(z["xy"][i], z["vel"][i], z["present"][i], z["team"], p)
        if c is None:
            continue
        inferred = False
        if not known:                                                       # failed pass: the events name no receiver and no end point
            f0 = int(z["frame"][i])
            if f0 + AIM_FRAMES >= len(ball):
                continue
            moved = ball[f0 + AIM_FRAMES] - ball[f0]                        # where the ball went in the first half second
            if np.isnan(moved).any() or np.linalg.norm(moved) < AIM_MIN_M:
                continue
            to_mate = c["posf"] - c["a"]
            cos = (to_mate @ moved) / (np.linalg.norm(to_mate, axis=1) * np.linalg.norm(moved) + 1e-9)
            if np.degrees(np.arccos(np.clip(cos.max(), -1, 1))) > AIM_MAX_DEG:
                continue                                                    # the ball did not head for any teammate
            r, inferred = int(c["slots"][int(cos.argmax())]), True
        row = row_of(c, r, p, z["complete"][i], int(z["frame"][i]), float(z["time_s"][i]))
        if row:
            row["inferred"] = inferred
            rows.append(row)
    return pack(rows)


def sim_match(m, vel_scale=1.0):
    """Passes of a simulated game: the owner of the ball changes from A to B of the same team, with the ball in the air for
    0.2-3 s, live play. A change to the other team in the same way is an interception (counted, for the completion rate).
    vel_scale multiplies every velocity (speed_check.py slows the simulated players down to real speed with it)."""
    d = pd.read_csv(settings.csv(m), usecols=["frame", "time_s", "team", "player_id", "x_m", "y_m", "vx_ms", "vy_ms", "ball_x_m", "ball_y_m",
                                               "ball_owner_team", "ball_owner_player", "game_mode"])
    f = d.drop_duplicates("frame").sort_values("frame")
    fr, t = f["frame"].to_numpy(), f["time_s"].to_numpy()
    live = (f["game_mode"] == "normal").to_numpy()
    slot = np.where(f["ball_owner_team"] == "home", f["ball_owner_player"], np.where(f["ball_owner_team"] == "away", f["ball_owner_player"] + 11, -1))
    own = np.where(live, slot, -1)
    found, n_cut, last = [], 0, None
    for k in np.flatnonzero(own >= 0):
        if last is not None and own[k] != own[last]:
            gap = t[k] - t[last]
            if SIM_GAP_MIN_S <= gap <= SIM_GAP_MAX_S and live[last:k + 1].all():
                if (own[k] < 11) == (own[last] < 11):
                    found.append((fr[last], int(own[last]), int(own[k]), t[last]))
                else:
                    n_cut += 1
        last = k
    frames = np.array(sorted({x[0] for x in found}))
    row = {fm: i for i, fm in enumerate(frames)}
    sub = d[d["frame"].isin(frames)]
    n = len(frames)
    xy, vel = np.full((n, 23, 2), np.nan), np.zeros((n, 23, 2))
    r = sub["frame"].map(row).to_numpy()
    s = (sub["player_id"] + np.where(sub["team"] == "away", 11, 0)).to_numpy()
    xy[r, s] = sub[["x_m", "y_m"]].to_numpy()
    vel[r, s] = np.nan_to_num(sub[["vx_ms", "vy_ms"]].to_numpy()) * vel_scale
    b = sub.drop_duplicates("frame")
    xy[b["frame"].map(row).to_numpy(), 22] = b[["ball_x_m", "ball_y_m"]].to_numpy()
    team = np.array([0] * 11 + [1] * 11 + [2])
    rows = []
    for frame, passer, receiver, tm in found:
        i = row[frame]
        pres = ~np.isnan(xy[i, :, 0])
        c = candidates(xy[i], vel[i], pres, team, passer)
        out = c and row_of(c, receiver, passer, True, int(frame), float(tm))
        if out:
            out["inferred"] = False
            rows.append(out)
    return pack(rows, n_cut)


def build(m):
    out = real_match(m) if m in settings.REAL else sim_match(m)
    np.savez_compressed(settings.base(m) + "_passfeat.npz", **out)
    return {"match": m, "group": settings.variant(m), "passes": len(out["target"]), "inferred": int(out["inferred"].sum()), "complete_%": round(100 * float(out["complete"].mean()), 1),
            "intercepted": int(out["n_intercept"]), "median_teammates": int(np.median(out["mask"].sum(1)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matches", nargs="*")
    a = ap.parse_args()
    todo = a.matches or [m for m in settings.available(settings.REAL) if os.path.exists(settings.base(m) + "_passes.npz")] + settings.available(settings.SIM)
    rows = []
    for m in todo:
        rows.append(build(m))
        print(f"  {m} done", flush=True)
    t = pd.DataFrame(rows).set_index("match")
    print("\nPASS OPTIONS WITH MOVEMENT AND TIMING\n")
    print(t.to_string())
    g = t.groupby("group").agg(matches=("passes", "size"), passes=("passes", "sum"), inferred=("inferred", "sum"), intercepted=("intercepted", "sum"))
    print("\nPER GROUP\n")
    print(g.to_string())
    print("\n  passes      = usable passes; real: open play only. The receiver is known (completed pass) or inferred (failed pass: the")
    print("                teammate the ball first headed for, within 20 degrees over its first 0.5 s; 'inferred' counts these)")
    print("  intercepted = simulator only: the ball went from one team to the other with the ball in the air (a failed pass)")
    print("\nsaved <base>_passfeat.npz per match")


if __name__ == "__main__":
    main()
