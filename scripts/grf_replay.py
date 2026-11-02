"""Start GRF from real Key States and let it play on (stage 2: runs in the GRF environment, numpy only).

For each chosen Key State: put the 22 real players and the ball where they stood, let GRF's built-in AI play the
next --steps steps (1 step = 0.1 s = 1 real frame), and measure how far GRF's players/ball are from where the real
players/ball went. Two reference guesses are scored alongside: 'still' (everyone stays put) and 'keep-velocity'.

Input : data/real/processed/<match>_ks_states.npz   (made by export_ks_states.py)
Output: outputs/grf_replay_<match>.csv  (one row per Key State and horizon) and a printed summary

Usage :  python scripts\\grf_replay.py --match J03WOH --n 20
         python scripts\\grf_replay.py --match J03WOH --n 3 --render      (watch it in a window)
"""
import argparse, csv, sys, types
import numpy as np
import gfootball.env as football_env
from gfootball import scenarios as S

L, W, FPS = 105.0, 68.0, 10
ROLE = {"GK": S.e_PlayerRole_GK, "CB": S.e_PlayerRole_CB, "LB": S.e_PlayerRole_LB, "RB": S.e_PlayerRole_RB,
        "DM": S.e_PlayerRole_DM, "CM": S.e_PlayerRole_CM, "LM": S.e_PlayerRole_LM, "RM": S.e_PlayerRole_RM,
        "AM": S.e_PlayerRole_AM, "CF": S.e_PlayerRole_CF}
BUILTIN_AI = 19            # index of action_builtin_ai in the 'v2' action set
HORIZONS = (10, 30, 50)    # 1 s, 3 s, 5 s


def to_grf(xy):            # metres (105 x 68, origin bottom-left) -> GRF ([-1, 1] x [-0.42, 0.42])
    return xy[..., 0] / L * 2 - 1, 0.42 - xy[..., 1] / W * 0.84


def to_metres(xy):         # inverse of to_grf, same convention as export_tracking.py
    return np.stack([(xy[..., 0] + 1) / 2 * L, (0.42 - xy[..., 1]) / 0.84 * W], axis=-1)


def register_scenario(snap):
    """GRF loads scenarios as modules of gfootball.scenarios; put one in memory that holds this Key State."""
    mod = types.ModuleType("gfootball.scenarios.real_keystate")

    def build_scenario(builder):
        cfg = builder.config()
        cfg.game_duration = 60000                      # steps of 0.1 s (GRF's default 3000 would stop a replay after 5 min)
        cfg.deterministic = False
        cfg.offsides = True
        cfg.end_episode_on_score = False
        cfg.end_episode_on_out_of_play = False
        cfg.end_episode_on_possession_change = False
        bx, by = to_grf(snap["ball"])
        builder.SetBallPosition(float(bx), float(by))
        for team, key in ((S.Team.e_Left, "home"), (S.Team.e_Right, "away")):
            builder.SetTeam(team)
            gx, gy = to_grf(snap[key])
            if key == "away":                  # a scenario states the right team from its own side: mirror both axes
                gx, gy = -gx, -gy
            for k in range(11):
                builder.AddPlayer(float(gx[k]), float(gy[k]), ROLE[snap["role_" + key][k]])

    mod.build_scenario = build_scenario
    sys.modules["gfootball.scenarios.real_keystate"] = mod


def rollout(snap, steps, seed, render):
    register_scenario(snap)
    env = football_env.create_environment(
        env_name="real_keystate", representation="raw", render=render,
        number_of_left_players_agent_controls=1,
        other_config_options={"game_engine_random_seed": seed, "action_set": "v2"})
    obs = env.reset()
    home, away, ball = [], [], []
    for t in range(steps + 1):
        o = obs[0]
        home.append(to_metres(np.array(o["left_team"])))
        away.append(to_metres(np.array(o["right_team"])))
        ball.append(to_metres(np.array(o["ball"][:2])))
        if t < steps:
            obs, _, done, _ = env.step(BUILTIN_AI)
            if done:
                break
    env.close()
    return np.array(home).transpose(1, 0, 2), np.array(away).transpose(1, 0, 2), np.array(ball)   # (11, T, 2), (T, 2)


def dist(a, b):
    return float(np.nanmean(np.linalg.norm(a - b, axis=-1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="J03WOH")
    ap.add_argument("--dir", default="data/real/processed")
    ap.add_argument("--n", type=int, default=20, help="number of Key States (evenly spread over the match)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()

    z = np.load(f"{a.dir}/{a.match}_ks_states.npz", allow_pickle=True)
    assert a.steps <= int(z["horizon"]), "export_ks_states.py was run with a shorter --horizon"
    idx = np.unique(np.linspace(0, len(z["ks_id"]) - 1, a.n).astype(int))
    rows = []
    for j in idx:
        snap = {"ball": z["ball"][j][0], "home": z["home"][j][:, 0], "away": z["away"][j][:, 0],
                "role_home": z["role_home"][j], "role_away": z["role_away"][j]}
        gh, ga, gb = rollout(snap, a.steps, a.seed, a.render)
        placed = max(dist(gh[:, 0], snap["home"]), dist(ga[:, 0], snap["away"]))
        for h in HORIZONS:
            if h > a.steps or gh.shape[1] <= h:
                continue
            rh, ra, rb = z["home"][j][:, h], z["away"][j][:, h], z["ball"][j][h]
            kvh = z["home"][j][:, 0] + z["vel_home"][j] * h / FPS
            kva = z["away"][j][:, 0] + z["vel_away"][j] * h / FPS
            rows.append({
                "ks_id": int(z["ks_id"][j]), "time_s": float(z["time_s"][j]), "horizon_s": h / FPS, "placed_err_m": placed,
                "grf_player_m": (dist(gh[:, h], rh) + dist(ga[:, h], ra)) / 2,
                "still_player_m": (dist(z["home"][j][:, 0], rh) + dist(z["away"][j][:, 0], ra)) / 2,
                "keepvel_player_m": (dist(kvh, rh) + dist(kva, ra)) / 2,
                "grf_ball_m": dist(gb[h], rb), "still_ball_m": dist(z["ball"][j][0], rb)})
        print(f"Key State {int(z['ks_id'][j]):5d}  t={float(z['time_s'][j]):7.1f}s  start offset {placed:.2f} m", flush=True)

    out = f"outputs/grf_replay_{a.match}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nMean error in metres vs the real next seconds ({len(idx)} Key States, lower is better)")
    print("horizon |  players: GRF   still  keep-vel | ball: GRF   still")
    for h in HORIZONS:
        r = [x for x in rows if x["horizon_s"] == h / FPS]
        if r:
            m = lambda k: np.mean([x[k] for x in r])
            print(f"{h / FPS:5.1f} s |           {m('grf_player_m'):5.2f}  {m('still_player_m'):5.2f}  {m('keepvel_player_m'):5.2f}"
                  f"    |       {m('grf_ball_m'):5.2f}  {m('still_ball_m'):5.2f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
