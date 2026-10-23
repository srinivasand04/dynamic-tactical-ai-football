"""Run one Google Research Football episode and export it as tracking data.

Output: a long-format CSV (one row per player per frame) in metres on a 105 x 68 pitch,
origin at the bottom-left corner, 10 frames per second - the same shape as open
tracking data (DFL/Sportec via kloppy), so the same analysis tools can read it.

Usage:  python export_tracking.py --steps 3000 --out match_001.csv
"""
import argparse, csv
import numpy as np
import gfootball.env as football_env

PITCH_L, PITCH_W = 105.0, 68.0
FPS = 10  # GRF advances 100 ms per env.step()

def to_metres(x, y):
    # GRF: x in [-1, 1] (left goal -> right goal), y in [-0.42, 0.42] (top -> bottom)
    return (x + 1.0) / 2.0 * PITCH_L, (0.42 - y) / 0.84 * PITCH_W

ROLE = {0: "GK", 1: "CB", 2: "LB", 3: "RB", 4: "DM", 5: "CM", 6: "LM", 7: "RM", 8: "AM", 9: "CF"}
MODE = {0: "normal", 1: "kickoff", 2: "goal_kick", 3: "free_kick", 4: "corner", 5: "throw_in", 6: "penalty"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="11_vs_11_stochastic")
    ap.add_argument("--steps", type=int, default=3000)   # 3000 steps = one full GRF match
    ap.add_argument("--out", default="grf_match.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    env = football_env.create_environment(env_name=a.scenario, representation="raw", render=False,
                                          number_of_left_players_agent_controls=1,
                                          other_config_options={"game_engine_random_seed": a.seed})
    obs = env.reset()
    rng = np.random.default_rng(a.seed)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "team", "player_id", "role", "x_m", "y_m", "vx_ms", "vy_ms",
                    "ball_x_m", "ball_y_m", "ball_z", "ball_owner_team", "ball_owner_player",
                    "game_mode", "score_home", "score_away"])
        for frame in range(a.steps):
            o = obs[0]
            bx, by = to_metres(o["ball"][0], o["ball"][1])
            owner = {-1: "none", 0: "home", 1: "away"}[int(o["ball_owned_team"])]
            for team, key in (("home", "left_team"), ("away", "right_team")):
                for pid, ((x, y), (dx, dy), role) in enumerate(zip(o[key], o[key + "_direction"], o[key + "_roles"])):
                    xm, ym = to_metres(x, y)
                    # direction is per-step displacement in GRF units -> metres per second
                    vx, vy = dx / 2.0 * PITCH_L * FPS, -dy / 0.84 * PITCH_W * FPS
                    w.writerow([frame, frame / FPS, team, pid, ROLE.get(int(role), role), round(xm, 2), round(ym, 2),
                                round(vx, 2), round(vy, 2), round(bx, 2), round(by, 2), round(float(o["ball"][2]), 2),
                                owner, int(o["ball_owned_player"]), MODE.get(int(o["game_mode"]), o["game_mode"]),
                                o["score"][0], o["score"][1]])
            # placeholder policy: the built-in AI plays everyone except our one player, who acts randomly.
            # Replace this line with your scripted / learned policy.
            obs, reward, done, info = env.step(int(rng.integers(19)))
            if done:
                break
    env.close()
    print(f"wrote {frame + 1} frames ({(frame + 1) / FPS:.0f} s of play) to {a.out}")

if __name__ == "__main__":
    main()
