"""TiZero (open GRF agent, Hugging Face OpenRL/tizero @53ce96d, Apache-2.0) in our GRF setup.

TiZero (Lin et al., AAMAS 2023) was trained by self-play and curriculum learning; it ranked 1st on the JIDI GRF board (Oct 2022).
Only its weights and agent code are used (models/tizero/, no pip install): a 330-feature observation per player, an LSTM-like
policy with 512 hidden units, deterministic actions; the goalkeeper is TiZero's own rule-based keeper (goal_keeper.py).

This script plays TiZero (left, all 10 outfield players + keeper) against GRF's built-in AI (right) and writes one row per game.
Usage : .venv\\Scripts\\python.exe scripts\\tizero_eval.py --games 20 --procs 4 [--env 11_vs_11_hard_stochastic] [--steps 3000]
Output: outputs/tizero/eval_<env>.csv  (seed, goals_for, goals_against, shots_for proxy, possession, seconds)
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TZ = os.path.join(ROOT, "models", "tizero")


class TiZero:
    """One TiZero team. act(obs_list) -> 11 action indices (GRF default action set, 19 actions)."""

    def __init__(self):
        sys.path.insert(0, TZ)
        import torch
        from openrl_policy import PolicyNetwork
        from openrl_utils import openrl_obs_deal, _t2n
        from goal_keeper import agent_get_action
        self.torch, self.deal, self.t2n, self.keeper = torch, openrl_obs_deal, _t2n, agent_get_action
        torch.set_num_threads(1)
        self.model = PolicyNetwork()
        self.model.load_state_dict(torch.load(os.path.join(TZ, "actor.pt"), map_location="cpu", weights_only=True))
        self.model.eval()
        self.reset()

    def reset(self):
        self.h = [np.zeros((1, 1, 512), np.float32) for _ in range(11)]

    def act_one(self, o):
        idx = int(o["active"])
        if idx == 0:                                                      # rule-based keeper, as in TiZero's submission
            return int(self.keeper(o)[0])
        d = self.deal(o)
        obs = d["obs"].reshape(1, 330)
        avail = np.zeros((1, 20)); avail[0, :19] = d["available_action"]
        with self.torch.no_grad():
            a, h = self.model(obs, self.h[idx], available_actions=avail, deterministic=True)
        a = int(a[0][0])
        if a == 17 and o["sticky_actions"][8] == 1:                       # same fix as the submission: no double dribble
            a = 15
        self.h[idx] = self.t2n(h).reshape(1, 1, 512)
        return a

    def act(self, obs_list):
        return [self.act_one(o) for o in obs_list]


def play(args):
    seed, env_name, steps = args
    import gfootball.env as football_env
    env = football_env.create_environment(env_name=env_name, representation="raw", number_of_left_players_agent_controls=11,
                                          other_config_options={"game_engine_random_seed": int(seed)})
    tz = TiZero()
    obs = env.reset()
    t0, t, done, own, shots = time.time(), 0, False, 0, 0
    prev_mode = 0
    while not done and t < steps:
        obs, _, done, _ = env.step(tz.act(obs))
        o = obs[0]
        own += o["ball_owned_team"] == 0
        if o["game_mode"] == 3 and prev_mode != 3 and o["ball"][0] < 0:  # goal kick for the right team = left shot/ball went out (proxy)
            shots += 1
        prev_mode = o["game_mode"]
        t += 1
    sc = obs[0]["score"]
    env.close()
    return dict(seed=seed, env=env_name, steps=t, goals_for=sc[0], goals_against=sc[1], goal_kicks_won=shots,
                possession=round(own / max(t, 1), 3), seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--env", default="11_vs_11_hard_stochastic")
    ap.add_argument("--steps", type=int, default=3000, help="3000 = a full GRF match")
    ap.add_argument("--seed0", type=int, default=0)
    a = ap.parse_args()
    import pandas as pd
    out = os.path.join(ROOT, "outputs", "tizero", f"eval_{a.env}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    jobs = [(a.seed0 + i, a.env, a.steps) for i in range(a.games)]
    rows = []
    with Pool(a.procs) as p:
        for r in p.imap_unordered(play, jobs):
            rows.append(r)
            print(f"seed {r['seed']}: {r['goals_for']}-{r['goals_against']}  possession {r['possession']:.0%}  ({r['seconds']:.0f}s)", flush=True)
            pd.DataFrame(rows).sort_values("seed").to_csv(out, index=False)
    d = pd.DataFrame(rows)
    gd = d.goals_for - d.goals_against
    print(f"\n{len(d)} games vs built-in AI ({a.env}): won {(gd > 0).mean():.0%}, drew {(gd == 0).mean():.0%}, lost {(gd < 0).mean():.0%}; "
          f"goals {d.goals_for.mean():.2f} - {d.goals_against.mean():.2f}; goal difference {gd.mean():+.2f} "
          f"+- {1.96 * gd.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0:.2f}\nwrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
