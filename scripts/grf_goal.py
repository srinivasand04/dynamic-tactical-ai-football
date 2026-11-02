"""Replay a real shot in GRF: the real shooter shoots, GRF's physics decides what the ball does (stage 2, GRF environment).

Starts GRF `lead` steps before the real shot from the real positions and ball, walks all 22 players along their real
paths (as grf_match.py does), and at the shot frame the player who really shot (the one nearest the ball on the
attacking team) faces the goal and presses 'shot'. The ball is not placed any more: from the shot on it is GRF's.
Because the ball's flight is GRF's, a goal is not guaranteed, so a small grid of lead times, aims and seeds is tried
and the first run that scores (else the closest) is kept.

Input : data/real/processed/<match>_segment_<start>.npz   (export_segment.py, a stretch that contains the shot)
Output: outputs/grf_goal_<match>_<start>.npz  (same layout as grf_match.py, for make_match_video.py --mark-s)

Usage :  python scripts\\grf_goal.py --segment data\\real\\processed\\J03WOH_segment_3600.npz --shot-s 3612.6
         python scripts\\grf_goal.py --segment ... --shot-s 3612.6 --watch      (replay the winning run in a GRF window)
"""
import argparse, os, time
import cv2
import numpy as np
import gfootball.env as football_env
from grf_follow import Walker, MIRROR
from grf_replay import register_scenario, to_metres, L, W, FPS

SHOT = 12
SHOT_HOLD = 3                                       # steps the shot button is held (set by the sweep)
AIMS = {"centre": 0, "high": 1, "low": -1}          # 'high' = towards larger y in metres


def aim_action(home_attacks_right, aim):
    east, ne, se, west, nw, sw = 5, 4, 6, 1, 2, 8
    return {0: (east if home_attacks_right else west), 1: (ne if home_attacks_right else nw),
            -1: (se if home_attacks_right else sw)}[AIMS[aim]]


def play(z, real, ball_real, m, lead, post, aim, seed, render=False, slow=0.0, video=None):
    T = real.shape[1]
    s = m - lead
    owner = str(z["owner"][m])
    team0 = 0 if owner == "home" else 11
    shooter = team0 + int(np.argmin(np.linalg.norm(real[team0:team0 + 11, m] - ball_real[m], axis=-1)))
    snap = {"ball": np.clip(ball_real[s], [0.5, 0.5], [L - 0.5, W - 0.5]), "home": real[:11, s], "away": real[11:, s],
            "role_home": z["role_home"], "role_away": z["role_away"]}
    register_scenario(snap)
    env = football_env.create_environment(
        env_name="real_keystate", representation="raw", render=render,
        number_of_left_players_agent_controls=11, number_of_right_players_agent_controls=11,
        other_config_options={"game_engine_random_seed": seed, "real_time": render and video is None})
    obs = env.reset()
    walkers = [Walker() for _ in range(22)]
    attacks_right = owner == "home"
    players, ball, scored = [], [], False
    for t in range(lead + post + 1):
        o = obs[0]
        players.append(np.concatenate([to_metres(np.array(o["left_team"])), to_metres(np.array(o["right_team"]))]))
        ball.append(to_metres(np.array(o["ball"][:2])))
        if o["score"][0 if owner == "home" else 1] > 0:
            scored = True
            players.pop(), ball.pop()                  # this frame is already the kickoff re-spawn: keep the one before
            break
        if video is not None:
            video.write(cv2.cvtColor(o["frame"], cv2.COLOR_RGB2BGR))
        if t == lead + post:
            break
        acts = [walkers[k].act(players[-1][k], real[k, min(s + t + 2, T - 1)]) for k in range(22)]
        if lead - 2 <= t < lead:
            acts[shooter] = aim_action(attacks_right, aim)                 # turn to the goal
        elif lead <= t < lead + SHOT_HOLD:
            acts[shooter] = SHOT
        acts = [MIRROR.get(x, x) if k >= 11 else x for k, x in enumerate(acts)]
        obs, _, done, _ = env.step(acts)
        if slow:
            time.sleep(slow)                                       # slow motion for the GRF window
        if done:
            break
    env.close()
    n = len(players)
    goal_x = L if attacks_right else 0.0
    closest = float(np.min(np.linalg.norm(np.array(ball) - np.array([goal_x, W / 2]), axis=-1)))
    return np.array(players).transpose(1, 0, 2), np.array(ball), scored, closest, shooter, n


def find_kick(ball_real, t_s, shot_s):
    """Frame index of the real kick: the event time lags the kick, so walk back from the goal line to where the ball starts to fly."""
    speed = np.r_[0, np.linalg.norm(np.diff(ball_real, axis=0), axis=1) * FPS]
    e = int(np.argmin(abs(t_s - shot_s)))
    window = np.arange(max(e - 30, 1), min(e + 60, len(t_s)))
    out = window[(ball_real[window, 0] < 0) | (ball_real[window, 0] > L)]       # ball beyond a goal line
    i = int(out[0]) if len(out) else int(window[np.argmax(speed[window] > 20)])
    while i > 1 and speed[i] > 12:                       # walk back to where the final fast stretch begins
        i -= 1
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", required=True)
    ap.add_argument("--shot-s", type=float, required=True, help="match time (s) of the real shot")
    ap.add_argument("--post", type=int, default=40, help="steps to play after the shot (10 = 1 s)")
    ap.add_argument("--watch", action="store_true", help="replay the chosen run in a GRF window, slow motion, three times")
    a = ap.parse_args()

    z = np.load(a.segment, allow_pickle=True)
    real = np.concatenate([z["home"], z["away"]], axis=1).transpose(1, 0, 2)
    ball_real, t_s = z["ball"], z["time_s"]
    m = find_kick(ball_real, t_s, a.shot_s)
    print(f"real kick at index {m} ({t_s[m]:.1f} s), team {z['owner'][m]}")

    global SHOT_HOLD
    best, tried = None, 0
    for hold in (1, 5):                                  # how many steps the shot button is pressed
        SHOT_HOLD = hold
        for lead in (3, 5, 8):
            for aim in AIMS:
                for seed in range(3):
                    pl, ball, scored, closest, shooter, n = play(z, real, ball_real, m, lead, a.post, aim, seed)
                    tried += 1
                    print(f"  press {hold} step(s)  lead {lead} steps  aim {aim:6s}  seed {seed}  goal={scored!s:5}  closest to goal {closest:5.1f} m", flush=True)
                    if best is None or (scored, -closest) > (best[0], -best[1]):
                        best = (scored, closest, lead, aim, seed, pl, ball, shooter, hold)
                    if scored:
                        break
                if best[0]:
                    break
            if best[0]:
                break
        if best[0]:
            break
    scored, closest, lead, aim, seed, pl, ball, shooter, SHOT_HOLD = best
    print(f"\nkept: lead {lead}, aim {aim}, seed {seed}, shooter player {shooter}: "
          f"{'GOAL in GRF' if scored else f'no goal (closest {closest:.1f} m)'}  ({tried} runs tried)")

    s, n = m - lead, pl.shape[1]
    key = os.path.basename(a.segment).replace("_segment_", "_").replace(".npz", "")
    out = f"outputs/grf_goal_{key}.npz"
    np.savez_compressed(out, grf=pl, ball_grf=ball, real=real[:, s:s + n], ball_real=ball_real[s:s + n],
                        time_s=t_s[s:s + n], live=z["live"][s:s + n], shot_index=lead, goal=scored)
    print("wrote", out, "- for the video: python scripts\\make_match_video.py --run", out, f"--mark-s {t_s[m]:.1f} --fps 5")
    if a.watch:
        for _ in range(3):                                   # three plays, in slow motion
            play(z, real, ball_real, m, lead, a.post, aim, seed, render=True, slow=0.25)


if __name__ == "__main__":
    main()
