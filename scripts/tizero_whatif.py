"""What-if with TiZero playing both teams (instead of our team models), from a real moment.

Same moments, scenarios and paired-seed comparison as whatif.py, so the two can be compared directly:
  whatif.py       : our team models (team_agent.Brain) play both sides
  tizero_whatif.py: TiZero (open GRF agent, OpenRL/tizero, scripts/tizero_eval.py) plays both sides
Only scenarios that change where players stand can be used (TiZero has no 'early pass' or team-style switch):
baseline, line_deeper, line_higher, and the coach move (--move PLAYER FORWARD LEFT).
GRF gives each right-team agent its own mirrored view and mirrors its actions, so TiZero needs no change for the right side.

Usage : .venv\\Scripts\\python.exe scripts\\tizero_whatif.py --seeds 20 --procs 4 [--start-s 3610] [--move NAME 0 5]
Output: outputs/whatif/<key>_tizero_runs.csv and _summary.csv (whatif.py's files are not touched)
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import whatif as WI  # noqa: E402

L, W, FPS = WI.L, WI.W, WI.FPS
SHOT, PASSES = 12, (9, 10, 11)
_M = {}


def play(seg, start_s, scen, shift, seed, seconds, move=None):
    import gfootball.env as football_env
    from grf_replay import register_scenario, to_metres
    from team_agent import shot_xg
    from tizero_eval import TiZero

    pos, ball, vel, att, mirrored, _ = WI.moment(seg, start_s, scen, shift, move)
    z = np.load(seg, allow_pickle=True)
    register_scenario({"ball": np.clip(ball[-1], [0.5, 0.5], [L - 0.5, W - 0.5]), "home": pos[:11, -1], "away": pos[11:, -1],
                       "role_home": z["role_home"], "role_away": z["role_away"]})
    env = football_env.create_environment(env_name="real_keystate", representation="raw",
                                          number_of_left_players_agent_controls=11, number_of_right_players_agent_controls=11,
                                          other_config_options={"game_engine_random_seed": int(seed)})
    obs = env.reset()
    if "tz" not in _M:
        _M["tz"] = (TiZero(), TiZero())
    tl, tr = _M["tz"]
    tl.reset(); tr.reset()
    att_side = "left" if att == "home" else "right"
    att_id = 0 if att_side == "left" else 1
    goal_x = L if att_side == "left" else 0.0
    res = {"scenario": scen, "seed": int(seed), "goal": 0, "goal_s": np.nan, "conceded": 0, "lost_ball_s": np.nan,
           "shots": 0, "xg": 0.0, "passes": 0}
    balls, players = [], []
    steps = int(seconds * FPS)
    for t in range(steps + 1):
        o = obs[0]
        lt, rt = to_metres(np.array(o["left_team"])), to_metres(np.array(o["right_team"]))
        b = to_metres(np.array(o["ball"][:2]))
        if o["score"][0] + o["score"][1] > 0:
            res["goal" if o["score"][att_id] > 0 else "conceded"] = 1
            res["goal_s"] = t / FPS
            break
        balls.append(b); players.append(np.concatenate([lt, rt]))
        if o["ball_owned_team"] == 1 - att_id and np.isnan(res["lost_ball_s"]):
            res["lost_ball_s"] = t / FPS
        if t == steps:
            break
        al, ar = tl.act(obs[:11]), tr.act(obs[11:])
        side_obs, side_act = (obs[:11], al) if att_side == "left" else (obs[11:], ar)
        for oo, a in zip(side_obs, side_act):                      # the attacking team's ball carrier: shots and passes
            if oo["ball_owned_team"] == 0 and oo["ball_owned_player"] == oo["active"] and np.isnan(res["lost_ball_s"]):
                own = to_metres(np.array(oo["left_team"])); opp = to_metres(np.array(oo["right_team"]))   # own view: attacking to x = 105
                k = int(oo["active"])
                if a == SHOT:
                    d = float(np.hypot(L - own[k, 0], W / 2 - own[k, 1]))
                    res["shots"] += 1; res["xg"] += shot_xg(own[k], opp, d)
                elif a in PASSES:
                    res["passes"] += 1
        obs, _, done, _ = env.step(al + ar)
        if done:
            break
    env.close()
    bxy = np.array(balls)
    bxy = bxy if att_side == "left" else np.c_[L - bxy[:, 0], W - bxy[:, 1]]
    keep = len(bxy) if np.isnan(res["lost_ball_s"]) else max(1, int(res["lost_ball_s"] * FPS))
    try:
        import value_models
        xt_path = value_models.xt(bxy[:keep])
        res["max_xt"] = round(float(xt_path.max()), 4); res["xt_gain"] = round(float(xt_path.max() - xt_path[0]), 4)
    except (OSError, ImportError, TypeError):
        res["max_xt"] = res["xt_gain"] = np.nan
    bx = np.array(balls)[:, 0]
    res.update({"xg": round(res["xg"], 3), "furthest_m": round(float(bx.max() if att_side == "left" else L - bx.min()), 1),
                "end_ball_to_goal_m": round(float(np.hypot(balls[-1][0] - goal_x, balls[-1][1] - W / 2)), 1),
                "seconds": round(len(balls) / FPS, 1)})
    return res


def _job(j):
    try:
        return play(*j)
    except Exception as e:
        return {"scenario": j[2], "seed": j[4], "error": repr(e)}


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default="data/real/processed/J03WOH_segment_3500.npz")
    ap.add_argument("--start-s", type=float, default=3610.0)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--scenarios", nargs="+", default=["baseline", "line_deeper", "line_higher"])
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--move", nargs=3, metavar=("PLAYER", "FORWARD", "LEFT"))
    a = ap.parse_args()
    if not os.path.isabs(a.segment):
        a.segment = os.path.join(WI.ROOT, a.segment)
    base = os.path.basename(a.segment).replace("_segment_", "_").replace(".npz", "")
    key = f"{base}_{int(a.start_s)}"
    move = None
    if a.move:
        z = np.load(a.segment, allow_pickle=True)
        k, name = WI.find_player(z, a.move[0])
        move = (k, name, float(a.move[1]), float(a.move[2]))
        a.scenarios = ["baseline", "move"]
        key += "_move_" + name.split(" ")[-1].replace(".", "") + f"_{a.move[1]}_{a.move[2]}"
    bad = [s for s in a.scenarios if s not in ("baseline", "line_deeper", "line_higher", "move")]
    if bad:
        raise SystemExit(f"TiZero can only play position scenarios, not {bad}")
    if "baseline" not in a.scenarios:
        a.scenarios = ["baseline"] + a.scenarios
    key += "_tizero"
    os.makedirs(WI.OUT, exist_ok=True)
    jobs = [(a.segment, a.start_s, s, a.shift, seed, a.seconds, move) for s in a.scenarios for seed in range(a.seeds)]
    print(f"{len(jobs)} TiZero runs ({len(a.scenarios)} scenarios x {a.seeds} seeds, {a.seconds:g} s), {a.procs} at a time", flush=True)
    t0, rows = time.time(), []
    with Pool(a.procs) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(r)
            if (k + 1) % 10 == 0 or k + 1 == len(jobs):
                print(f"  {k + 1}/{len(jobs)} runs ({(time.time() - t0) / 60:.1f} min)", flush=True)
    df = pd.DataFrame(rows).sort_values(["scenario", "seed"])
    df.to_csv(f"{WI.OUT}/{key}_runs.csv", index=False)
    if "error" in df:
        print("errors:", df["error"].dropna().unique()[:3])
    sm = WI.summarise(df.dropna(subset=["goal"]) if "error" in df else df, a.shift, move)
    sm.round(3).to_csv(f"{WI.OUT}/{key}_summary.csv", index=False)
    cols = ["what_changed", "goal_rate", "shots", "xg", "max_xt", "d_goal", "d_goal_ci95", "d_xg", "d_xg_ci95", "d_max_xt", "d_max_xt_ci95"]
    print(sm[[c for c in cols if c in sm]].round(2).to_string(index=False))
    print(f"wrote {WI.OUT}/{key}_runs.csv and _summary.csv")


if __name__ == "__main__":
    main()
