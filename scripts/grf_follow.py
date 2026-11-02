"""Drive all 22 GRF players toward real targets, re-starting from every real Key State (stage 2, GRF environment).

Each window starts from a real Key State (positions as in grf_replay.py). Every player is then steered by a simple
controller: walk toward a target position, sprint when it is far, stop when close. The built-in AI is not used.

  --mode ks     target = where that player really stood at the NEXT Key State (fixed for the window).
                This is an oracle: it is told the answer, so it is the upper bound a learned Key-State model can reach.
                A model that predicts the next Key State (the GNN) replaces these targets later.
  --mode track  target = where the player really stood 0.2 s from now (follow the real trajectory).

Score: mean distance (m) between GRF and the real players/ball at the end of the window (ks) or after 1/3/5 s (track),
next to the 'still' and 'keep-velocity' guesses. The ball is not steered, so its error shows what players alone achieve.

Input : data/real/processed/<match>_ks_states.npz  (export_ks_states.py)
Output: outputs/grf_follow_<mode>_<match>.csv and a printed summary

Usage :  python scripts\\grf_follow.py --match J03WOH --mode ks --n 30
         python scripts\\grf_follow.py --match J03WOH --mode track --n 10 --render
"""
import argparse, csv
import cv2
import numpy as np
import gfootball.env as football_env
from grf_replay import register_scenario, to_metres, dist, FPS

DIR_ACTION = [5, 4, 3, 2, 1, 8, 7, 6]       # east, NE, N, NW, W, SW, S, SE (GRF 'top' = larger y in metres)
MIRROR = {a: DIR_ACTION[(DIR_ACTION.index(a) + 4) % 8] for a in DIR_ACTION}
SPRINT, RELEASE_DIR, RELEASE_SPRINT = 13, 14, 15
STOP_M, SPRINT_M = 1.0, 6.0                 # stop within 1 m of the target, sprint when more than 6 m away
LAST = {"scored": False}                    # set by window(): did a goal happen in the last window


class Walker:
    """Turns 'go to this point' into GRF's sticky actions (a direction stays on until it is released)."""
    def __init__(self):
        self.moving, self.sprinting = False, False

    def act(self, pos, target):
        d = target - pos
        r = float(np.hypot(*d))
        if r < STOP_M:
            if self.sprinting:
                self.sprinting = False
                return RELEASE_SPRINT
            if self.moving:
                self.moving = False
                return RELEASE_DIR
            return 0
        want_sprint = r > SPRINT_M
        if want_sprint != self.sprinting and want_sprint:
            self.sprinting = True
            return SPRINT
        if self.sprinting and not want_sprint:
            self.sprinting = False
            return RELEASE_SPRINT
        self.moving = True
        return DIR_ACTION[int(round(np.degrees(np.arctan2(d[1], d[0])) / 45.0)) % 8]


def window(snap, targets_at, steps, seed, render, real_time=False, video=None, overrides=None, stop=None, skip=0, frame_hook=None):
    """Play `steps` steps; targets_at(t) -> (22, 2) metres. Returns player tracks (22, steps+1, 2) and ball (steps+1, 2).
    overrides: {step: {player slot: action}} replaces the walker's action of that player at that step (a turn, a pass, a shot).
    stop: stop(t, players (22, 2), ball (2,)) -> True ends the window at step t (that state is returned, its picture is not written)."""
    register_scenario(snap)
    env = football_env.create_environment(
        env_name="real_keystate", representation="raw", render=render,
        number_of_left_players_agent_controls=11, number_of_right_players_agent_controls=11,
        other_config_options={"game_engine_random_seed": seed, "real_time": real_time})
    obs = env.reset()
    walkers = [Walker() for _ in range(22)]
    players, ball = [], []
    LAST["scored"] = False
    for t in range(steps + 1):
        o = obs[0]
        if o["score"][0] + o["score"][1] > 0:                    # a goal: GRF has already put everybody back to kick-off, drop that frame
            LAST["scored"] = True
            break
        players.append(np.concatenate([to_metres(np.array(o["left_team"])), to_metres(np.array(o["right_team"]))]))
        ball.append(to_metres(np.array(o["ball"][:2])))
        if stop is not None and t > 0 and stop(t, players[-1], ball[-1]):
            break
        if video is not None and skip <= t < steps:          # GRF's own rendered picture (the next window supplies the last frame)
            fr = cv2.cvtColor(o["frame"], cv2.COLOR_RGB2BGR)
            video.write(frame_hook(t, fr) if frame_hook else fr)
        if t < steps:
            tg = targets_at(t)
            acts = [walkers[k].act(players[-1][k], tg[k]) for k in range(22)]
            for k, x in (overrides or {}).get(t, {}).items():
                acts[k] = x
                if x in DIR_ACTION:                              # a direction stays on (sticky), so the walker has to know
                    walkers[k].moving = True
            # the right team's agents act from their own side: a direction command is turned 180 degrees
            acts = [MIRROR.get(x, x) if k >= 11 else x for k, x in enumerate(acts)]
            obs, _, done, _ = env.step(acts)
            if done:
                break
    env.close()
    return np.array(players).transpose(1, 0, 2), np.array(ball)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="J03WOH")
    ap.add_argument("--dir", default="data/real/processed")
    ap.add_argument("--mode", choices=["ks", "track"], default="ks")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()

    z = np.load(f"{a.dir}/{a.match}_ks_states.npz", allow_pickle=True)
    H = int(z["horizon"])
    real = np.concatenate([z["home"], z["away"]], axis=1)             # (n, 22, H+1, 2)
    vel = np.concatenate([z["vel_home"], z["vel_away"]], axis=1)      # (n, 22, 2)
    ids, frames = z["ks_id"], z["frame"]
    if a.mode == "ks":                                                # windows whose next Key State is also in the file
        ok = np.where((np.r_[ids[1:] == ids[:-1] + 1, False]) & (np.r_[frames[1:] - frames[:-1], 0] <= H))[0]
        gap = {j: int(frames[j + 1] - frames[j]) for j in ok}
    else:
        ok = np.arange(len(ids))
    pick = ok[np.unique(np.linspace(0, len(ok) - 1, a.n).astype(int))]

    rows = []
    for j in pick:
        snap = {"ball": z["ball"][j][0], "home": z["home"][j][:, 0], "away": z["away"][j][:, 0],
                "role_home": z["role_home"][j], "role_away": z["role_away"][j]}
        if a.mode == "ks":
            steps, evals = gap[j], [gap[j]]
            fixed = np.concatenate([z["home"][j], z["away"][j]], axis=0)[:, gap[j]]
            targets_at = lambda t, fixed=fixed: fixed
        else:
            steps, evals = 50, [10, 30, 50]
            targets_at = lambda t, j=j: real[j][:, min(t + 2, H)]
        pl, gb = window(snap, targets_at, steps, a.seed, a.render)
        for h in evals:
            if pl.shape[1] <= h:
                continue
            rows.append({
                "ks_id": int(ids[j]), "time_s": float(z["time_s"][j]), "eval_s": h / FPS,
                "grf_player_m": dist(pl[:, h], real[j][:, h]),
                "still_player_m": dist(real[j][:, 0], real[j][:, h]),
                "keepvel_player_m": dist(real[j][:, 0] + vel[j] * h / FPS, real[j][:, h]),
                "grf_ball_m": dist(gb[h], z["ball"][j][h]), "still_ball_m": dist(z["ball"][j][0], z["ball"][j][h])})
        print(f"Key State {int(ids[j]):5d}  t={float(z['time_s'][j]):7.1f}s  {steps} steps", flush=True)

    out = f"outputs/grf_follow_{a.mode}_{a.match}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    m = lambda k, r: np.mean([x[k] for x in r])
    print(f"\nMean error in metres vs the real game ({len(pick)} windows, mode {a.mode}; lower is better)")
    print("window |  players: GRF   still  keep-vel | ball: GRF   still")
    groups = [("end of window", rows)] if a.mode == "ks" else [(f"{h / FPS:.0f} s", [x for x in rows if x["eval_s"] == h / FPS]) for h in (10, 30, 50)]
    for name, r in groups:
        if r:
            print(f"{name:>13} |           {m('grf_player_m', r):5.2f}  {m('still_player_m', r):5.2f}  {m('keepvel_player_m', r):5.2f}"
                  f"    |       {m('grf_ball_m', r):5.2f}  {m('still_ball_m', r):5.2f}")
    if a.mode == "ks":
        print(f"mean window length {np.mean([x['eval_s'] for x in rows]):.1f} s")
    print("wrote", out)


if __name__ == "__main__":
    main()
