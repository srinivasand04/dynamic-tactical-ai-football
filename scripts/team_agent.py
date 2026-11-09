"""A team model playing in GRF (Part B1): one GRF team of eleven, controlled by a team's movement model and decision model.

Every second the brain keeps a snapshot of the game seen from its own side (its team attacking towards x = 105, as in training).
Every half second the team movement model (team_policy.py) reads the last three snapshots, with its own team identity and the
opponent's, and gives each of its players where to be 2 s from now; the walker of grf_follow.py turns that into GRF actions.
On top of it:
  * the ball carrier follows the team's decision model (team_decisions.py): he shoots when he is nearer to goal than the team's usual
    shooting distance, otherwise keeps the ball for the team's usual time and then passes to the teammate the team's receiver
    weights prefer (turn towards him for 2 steps, then short or long pass);
  * when the other team has the ball (or nobody has it) the player nearest to the ball goes for it;
  * a set piece (throw-in, free kick, corner, kick-off, goal kick) is taken by the team's player nearest to the ball.
The game statistics of each side are counted on the way (shots, a distance-only xG proxy, passes and completed passes, possession,
time with the ball in the opponent's half).

play_game(left, right, ...) plays one GRF match (3000 steps = GRF's full match, about 5 minutes of play) and returns the numbers.
"left"/"right" are team names of the model (e.g. Argentina), "pooled" (the same model told 'other') or "GRF AI" (GRF's own bot,
right side only).
"""
import json
import os

import numpy as np
import torch
import gfootball.env as football_env

import train_gnn as G
from build_graphs import make_edges
from grf_follow import Walker, MIRROR, DIR_ACTION, SPRINT
from grf_replay import to_metres
from team_decisions import candidate_features

L, W = 105.0, 68.0
SHORT, LONG, SHOT = 11, 9, 12
HORIZON_S = 2.0                     # targets: where each player should be this many seconds from now
SNAP_EVERY, THINK_EVERY = 10, 5     # steps (0.1 s each)
FACE = 2


def flip(p):
    return np.stack([L - p[..., 0], W - p[..., 1]], axis=-1)


def face_action(frm, to):
    ang = np.degrees(np.arctan2(to[1] - frm[1], to[0] - frm[0]))
    return DIR_ACTION[int(round(ang / 45.0)) % 8]


def xg_proxy(d):
    """Distance-only chance quality (about 0.15 from 11 m, 0.04 from 20 m); a proxy for comparing models, not a real xG model."""
    return float(0.8 * np.exp(-0.15 * d))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_models(dataset="pff"):
    base = os.path.join(ROOT, "outputs", "team_policy", dataset)
    cfg = json.load(open(base + "_config.json"))
    net = G.Net(cfg["f_in"], cfg["dim"], "ours")
    net.load_state_dict(torch.load(base + "_model.pt", map_location="cpu"))
    net.eval()
    dec = json.load(open(base + "_decisions.json"))
    return cfg, net, dec


class Brain:
    def __init__(self, team, opponent, side, cfg, net, dec, rng):
        names = cfg["teams"]
        other = len(names) - 1
        self.side, self.net, self.rng = side, net, rng
        self.own_id = names.index(team) if team in names else other
        self.opp_id = names.index(opponent) if opponent in names else other
        self.n_ids = len(names)
        d = dec["teams"].get(team, dec["teams"]["other"])
        self.w = np.array(d["weights"])
        self.hold_steps = max(3, int(round(d["hold_s"] * 10)))
        self.shot_d = d["shot_dist_m"]
        self.long_share = d["long_share"]
        self.snaps, self.targets = [], None
        self.walkers = [Walker() for _ in range(11)]
        self.carrier, self.hold, self.plan = None, 0, []
        self.stats = {"shots": 0, "xg": 0.0, "passes": 0, "passes_done": 0, "own_steps": 0, "opp_half_steps": 0}
        self.pending_pass = False
        self.early_pass = False             # what-if: the first carrier passes at once to the most advanced teammate

    # ---- what the brain sees: its own team attacking towards x = 105, own players in slots 0-10 ----
    def view(self, lt, rt, ball):
        if self.side == "left":
            return lt, rt, ball
        return flip(rt), flip(lt), flip(ball)

    def to_abs(self, p):
        return p if self.side == "left" else flip(p)

    def snapshot(self, own, opp, ball, own_v, opp_v, ball_v, roles_own, roles_opp):
        xy = np.concatenate([own, opp, ball[None]])
        vel = np.concatenate([own_v, opp_v, ball_v[None]])
        role = np.r_[roles_own, roles_opp, -1]
        self.snaps = (self.snaps + [(xy, vel, role)])[-G.L:]

    def think(self):
        """Team movement model: targets (own frame) for the eleven players."""
        if len(self.snaps) < G.L:
            return
        feats, adjs, dists = [], [], []
        for xy, vel, role in self.snaps:
            pos = (xy - np.array([52.5, 34.0])) / np.array([52.5, 34.0])
            v = vel.copy()
            v[:22] = np.clip(v[:22], -8, 8) / 8.0
            v[22] = np.clip(v[22], -30, 30) / 30.0
            db = np.linalg.norm(xy - xy[22], axis=1) / 50.0
            team1 = np.eye(3)[np.r_[np.zeros(11, int), np.ones(11, int), 2]]
            role1 = np.eye(11)[np.clip(role + 1, 0, 10)]
            ids = np.zeros((G.N, self.n_ids))
            ids[:11, self.own_id] = 1
            ids[11:22, self.opp_id] = 1
            feats.append(np.concatenate([pos, v, db[:, None], team1, role1, np.ones((G.N, 1)), ids], axis=1))
            a_t, a_o, a_b, D, _ = make_edges(xy[None], "delaunay", 8.0, 3, 5)
            adjs.append(np.stack([a_t[0], a_o[0], a_b[0]]))
            dists.append(np.clip(np.nan_to_num(D[0], nan=60.0), 0, 30) / 30.0)
        gaps = np.log1p(np.array([[1.0, HORIZON_S]] * G.L, np.float32))
        b = {"feat": torch.tensor(np.array(feats)[None], dtype=torch.float32), "adj": torch.tensor(np.array(adjs)[None]),
             "dist": torch.tensor(np.array(dists)[None], dtype=torch.float32), "gaps": torch.tensor(gaps[None])}
        with torch.no_grad():
            pred, _ = self.net(b)
        disp = pred[0].numpy() * G.D_MOVE
        self.targets = self.snaps[-1][0][:11] + disp[:11]

    # ---- the actions of the eleven players for this step (absolute frame for the walkers) ----
    def act(self, o, own, opp, ball, own_abs):
        side_id = 0 if self.side == "left" else 1
        owner_team, owner = o["ball_owned_team"], o["ball_owned_player"]
        ours = owner_team == side_id
        ball_abs = self.to_abs(ball)
        if self.pending_pass and owner_team != -1:              # first touch after our pass: did a teammate get it?
            self.stats["passes_done"] += int(ours)
            self.pending_pass = False
        if ours:
            self.stats["own_steps"] += 1
        if ball[0] > L / 2:
            self.stats["opp_half_steps"] += 1
        tg = own.copy() if self.targets is None else self.targets.copy()
        # set piece: our player nearest to the ball takes it
        set_piece = o["game_mode"] != 0
        d_ball = np.linalg.norm(own - ball, axis=1)
        carrier = owner if ours else (int(d_ball.argmin()) if set_piece and d_ball.min() < 2.5 else None)
        acts = [None] * 11
        if carrier is not None:
            self.hold = self.hold + 1 if carrier == self.carrier else 0
            self.carrier = carrier
            if self.plan:
                acts[carrier] = self.plan.pop(0)
            else:
                goal = np.array([L, W / 2])
                dg = float(np.linalg.norm(own[carrier] - goal))
                if self.early_pass and not set_piece:
                    self.early_pass = False
                    j = max((k for k in range(1, 11) if k != carrier), key=lambda k: own[k][0])
                    far = np.linalg.norm(own[j] - own[carrier]) > 25
                    self.plan = [face_action(own[carrier], own[j])] * FACE + [LONG if far else SHORT]
                    self.stats["passes"] += 1
                    self.pending_pass = True
                    self.hold = 0
                elif dg < self.shot_d and abs(own[carrier][1] - W / 2) < 20 and not set_piece:
                    self.plan = [face_action(own[carrier], goal)] * FACE + [SHOT]
                    self.stats["shots"] += 1
                    self.stats["xg"] += xg_proxy(dg)
                elif self.hold >= (3 if set_piece else self.hold_steps):
                    P = np.concatenate([own, opp])[:, None]
                    cands, X = candidate_features(P, 0, carrier)
                    if len(cands):
                        s = X @ self.w
                        p = np.exp(s - s.max())
                        p /= p.sum()
                        j = cands[int(self.rng.choice(len(cands), p=p))]
                        far = np.linalg.norm(own[j] - own[carrier]) > 25 or self.rng.random() < self.long_share * 0.3
                        self.plan = [face_action(own[carrier], own[j])] * FACE + [LONG if far else SHORT]
                        self.stats["passes"] += 1
                        self.pending_pass = True
                        self.hold = 0
                if self.plan:
                    acts[carrier] = self.plan.pop(0)
        else:
            self.carrier, self.hold, self.plan = None, 0, []
            if not set_piece:                                   # the nearest outfield player goes for the ball
                k = 1 + int(d_ball[1:].argmin())
                tg[k] = ball
        out = []
        tg_abs = self.to_abs(tg)
        for k in range(11):
            if acts[k] is not None:
                if acts[k] in DIR_ACTION:
                    self.walkers[k].moving = True
                x = acts[k]
            else:
                x = self.walkers[k].act(own_abs[k], tg_abs[k])
            if acts[k] is None and k != carrier and tg is not None and np.linalg.norm(tg[k] - own[k]) > 12 and not self.walkers[k].sprinting:
                self.walkers[k].sprinting = True
                x = SPRINT
            out.append(MIRROR.get(x, x) if self.side == "right" else x)
        return out


def play_game(left, right, cfg, net, dec, seed=0, video=None, steps=None, tracks=None):
    """One GRF match, left team vs right team. Returns a dict of the result and both sides' statistics.
    tracks: a list that gets one (players (22, 2) m, ball (3,), score (2,), game mode) per step, for the 3D viewer."""
    rng = np.random.default_rng(seed)
    ai_right = right == "GRF AI"
    env = football_env.create_environment(
        env_name="11_vs_11_stochastic", representation="raw", render=video is not None,
        number_of_left_players_agent_controls=11, number_of_right_players_agent_controls=0 if ai_right else 11,
        other_config_options={"game_engine_random_seed": seed})
    obs = env.reset()
    brains = {"left": Brain(left, right, "left", cfg, net, dec, rng)}
    if not ai_right:
        brains["right"] = Brain(right, left, "right", cfg, net, dec, rng)
    prev, t, done = None, 0, False
    while not done and (steps is None or t < steps):
        o = obs[0]
        lt, rt = to_metres(np.array(o["left_team"])), to_metres(np.array(o["right_team"]))
        ball = to_metres(np.array(o["ball"][:2]))
        cur = (lt, rt, ball)
        if prev is None:
            prev = cur
        if video is not None:
            import cv2
            video.write(cv2.cvtColor(o["frame"], cv2.COLOR_RGB2BGR))
        if tracks is not None:
            tracks.append((np.concatenate([lt, rt]), np.r_[ball, o["ball"][2]], tuple(o["score"]), int(o["game_mode"])))
        acts = []
        for side, br in brains.items():
            own, opp, bl = br.view(lt, rt, ball)
            pown, popp, pbl = br.view(*prev)
            if t % SNAP_EVERY == 0:
                roles_own = np.array(o["left_team_roles"] if side == "left" else o["right_team_roles"])
                roles_opp = np.array(o["right_team_roles"] if side == "left" else o["left_team_roles"])
                br.snapshot(own, opp, bl, (own - pown) * 10, (opp - popp) * 10, (bl - pbl) * 10, roles_own, roles_opp)
            if t % THINK_EVERY == 0:
                br.think()
            acts += br.act(o, own, opp, bl, lt if side == "left" else rt)
        prev = cur
        obs, _, done, _ = env.step(acts)
        t += 1
    score = obs[0]["score"]
    env.close()
    res = {"left": left, "right": right, "seed": seed, "steps": t, "goals_left": int(score[0]), "goals_right": int(score[1])}
    owned = sum(br.stats["own_steps"] for br in brains.values()) or 1
    for side, br in brains.items():
        s = br.stats
        res.update({f"{side}_shots": s["shots"], f"{side}_xg": round(s["xg"], 3), f"{side}_passes": s["passes"],
                    f"{side}_pass_completion": round(s["passes_done"] / s["passes"], 3) if s["passes"] else np.nan,
                    f"{side}_possession": round(s["own_steps"] / owned, 3) if not ai_right else np.nan,
                    f"{side}_opp_half": round(s["opp_half_steps"] / max(t, 1), 3)})
    return res
