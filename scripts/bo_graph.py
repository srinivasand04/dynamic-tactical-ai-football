"""The ball-owner (BO) graph, as drawn on the whiteboard, with the agreed weight.

Edges
    BO -- every teammate         weight = free x reach x goal x lane  (geometric mean by default), signed
    BO -- opponents within thr   the distance in metres (the pressure on the ball owner)

Ingredients of a BO-teammate weight, each between 0 and 1
    free   space around the teammate        min(distance to his nearest opponent / 8 m, 1)
    reach  how far the pass is              exp(-distance BO to teammate / 20 m)
    goal   how close he is to the goal      exp(-distance teammate to the CENTRE of the opponent goal / 30 m)
    lane   is the passing lane open?        1 - the strongest block, where an opponent between BO and the teammate blocks with
                                            strength  max(0, 1 - distance from the pass line / 2.5 m)  x  exp(-distance of that
                                            opponent from BO / 15 m).
                                            The second factor is for lofted passes: an opponent far from the ball owner is easier
                                            to pass over, so he blocks less.
Offside   the teammate is past the halfway line, ahead of the ball and ahead of the second-last opponent (goalkeeper included).
          His weight is multiplied by -1, not by 0, so the size is kept (what the pass would have been worth).

Who is BO?  Given (bo=slot), for example the passer from the event data (extract_passes.py). Otherwise a guess: the tracking
only says which TEAM has the ball, so BO is the player of that team closest to the ball, provided he is within max_ball_dist
metres of it (otherwise the ball is in flight and there is no BO, so no graph).

Learned numbers  fit_bo_weights.py learns the five distances and how much each ingredient counts ("importance", summing to 1)
from real passes, and saves them in outputs/bo_params_learned.json. Use them with  build(..., **bo_graph.learned()).
With equal importance (0.25 each) the weight is the plain geometric mean.

    import bo_graph
    g = bo_graph.build(xy, present, team, owner_team)       # one Key State, BO guessed
    g = bo_graph.build(xy, present, team, None, bo=7)       # BO given (slot 7)
xy (23, 2) metres, present (23,), team (23,) 0 home / 1 away / 2 ball; slot 22 = the ball. Home attacks toward x = 105.
Every number above is a starting value, set in PARAMS.
"""
import json
import os

import numpy as np

BALL = 22
L, W = 105.0, 68.0
PARAMS = dict(free_m=8.0, reach_m=20.0, goal_m=30.0, lane_width_m=2.5, loft_m=15.0, thr_m=8.0, max_ball_dist_m=4.0, mean="geometric",
              importance=(0.25, 0.25, 0.25, 0.25))
LEARNED = "outputs/bo_params_learned.json"


def learned(path=LEARNED):
    """The numbers learned by fit_bo_weights.py, ready to pass to build(**...)."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run scripts/fit_bo_weights.py first")
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    return {**s["scales"], "importance": tuple(s["importance"][k] for k in ("free", "reach", "goal", "lane"))}


def _lane(xy, bo, t, opps, p):
    """Return (lane score between 0 and 1, slot of the strongest blocker or None)."""
    a, b = xy[bo], xy[t]
    v = b - a
    n2 = float(v @ v)
    if n2 < 1e-6:
        return 1.0, None
    best, who = 0.0, None
    for o in opps:
        u = float(((xy[o] - a) @ v) / n2)
        if not (0.0 < u < 1.0):
            continue                                                    # not between the ball owner and the teammate
        along = u * np.sqrt(n2)                                         # how far from BO, along the pass
        across = float(np.linalg.norm(xy[o] - (a + u * v)))             # how far from the pass line
        strength = max(0.0, 1.0 - across / p["lane_width_m"]) * np.exp(-along / p["loft_m"])
        if strength > best:
            best, who = strength, o
    return 1.0 - best, who


def build(xy, present, team, owner_team, bo=None, **kw):
    p = {**PARAMS, **kw}
    if bo is not None:
        side = int(team[bo])
    else:
        side = 0 if owner_team == "home" else 1 if owner_team == "away" else None
    if side is None or np.isnan(xy[BALL, 0]):
        return None
    mates = [i for i in range(22) if team[i] == side and present[i]]
    opps = [i for i in range(22) if team[i] == 1 - side and present[i]]
    if not mates or len(opps) < 2:
        return None
    if bo is None:
        d_ball = {i: float(np.linalg.norm(xy[i] - xy[BALL])) for i in mates}
        bo = min(d_ball, key=d_ball.get)
        if d_ball[bo] > p["max_ball_dist_m"]:
            return None

    att = (lambda x: x) if side == 0 else (lambda x: L - x)                 # distance along the attacking direction
    goal = np.array([L if side == 0 else 0.0, W / 2])
    last2 = sorted((att(xy[o, 0]) for o in opps), reverse=True)[1]          # the second-last opponent (the offside line)
    edges = []
    for t in mates:
        if t == bo:
            continue
        free = min(min(float(np.linalg.norm(xy[t] - xy[o])) for o in opps) / p["free_m"], 1.0)
        reach = float(np.exp(-float(np.linalg.norm(xy[t] - xy[bo])) / p["reach_m"]))
        gl = float(np.exp(-float(np.linalg.norm(xy[t] - goal)) / p["goal_m"]))
        lane, blocker = _lane(xy, bo, t, opps, p)
        parts = np.array([free, reach, gl, lane])
        imp = np.asarray(p["importance"], float) / np.sum(p["importance"])
        w = float(np.prod(parts ** imp)) if p["mean"] == "geometric" else float(np.prod(parts))
        offside = bool(att(xy[t, 0]) > L / 2 and att(xy[t, 0]) > att(xy[BALL, 0]) and att(xy[t, 0]) > last2)
        edges.append({"from": bo, "to": t, "type": "teammate", "free": free, "reach": reach, "goal": gl, "lane": lane, "blocker": blocker,
                      "offside": offside, "weight_abs": w, "weight": -w if offside else w, "metres": float(np.linalg.norm(xy[t] - xy[bo]))})
    for o in opps:
        d = float(np.linalg.norm(xy[bo] - xy[o]))
        if d <= p["thr_m"]:
            edges.append({"from": bo, "to": o, "type": "opponent", "metres": d})
    return {"bo": bo, "side": side, "mates": mates, "opps": opps, "edges": edges, "offside_line": last2, "params": p}
