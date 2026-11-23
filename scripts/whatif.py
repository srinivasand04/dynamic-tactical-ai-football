"""What-if (Part C): take a real moment, change one thing, and let the team models play it out in GRF, many times.

From a real moment (a few seconds before a real attack ends) GRF is started with the real positions of all 22 players and the
ball. From there both teams are played by the team models (team_agent.py: movement model + decision model), the same way as in
the league. Every scenario is played with the same list of seeds, so each what-if run has a 'no change' twin with the same
seed: the difference between the two is the effect of the change, and the 95% interval over the seeds says whether it is real.

Scenarios (attacking team = the team with the ball at the start, defending team = the other):
  baseline         nothing changed (the models play the real moment)
  line_deeper      the defending team's back four start --shift metres nearer their own goal
  line_higher      the back four start --shift metres further up the pitch
  early_pass       the first ball carrier passes at once to his most advanced teammate
  press            the defending team's 3 players nearest the ball start 3 m closer to it
  compact          the defending team starts 25% narrower (towards its own middle)
  wide_attack      the attacking team's 2 widest players (not the carrier) start 5 m wider
  striker_deeper   the attacking team's most advanced player starts 5 m deeper
  extra_runner     the attacking midfielder nearest behind the ball starts 8 m further forward (a run)
  shoot_early      the attacking team shoots from 6 m further out than its usual distance
  patient          the attacking carrier keeps the ball twice as long before passing
  quick_play       the attacking carrier passes twice as fast
  team_press       every defending outfield player starts 3 m closer to the ball
  low_block        the whole defending team starts 6 m nearer its own goal
  man_marking      each defending outfield player starts halfway to the attacker nearest him
  defence_wider    the defending team starts 25% wider
  keeper_off_line  the defending keeper starts 4 m off his line
  double_team      the 2 defenders nearest the carrier close him down to 1.5 m
  overload_ball_side  the 2 attackers furthest from the ball's side come up to 8 m across towards it
  fullbacks_overlap   the attacking team's full-backs (2 widest of its 4 deepest) start 10 m further forward
  switch_play      the first carrier passes at once to the teammate furthest across the pitch
  long_ball        the attacking team plays every pass long
  att_as_<Team>    the attacking team plays with that team's model and style (e.g. att_as_Croatia)
  move             one chosen player starts somewhere else (--move PLAYER FORWARD LEFT): the coach's "you should have been
                   there". PLAYER = part of the name ('Klaus') or side:shirt ('home:11'); FORWARD = metres towards the goal
                   his team attacks (negative = back), LEFT = metres towards the touchline on his left (negative = right).
                   His last 2 s move with him, so he arrives at the new spot with his real speed. Results get their own
                   files (..._move_<name>_...), the earlier results stay.

Scores per run (attacking team's point of view, within --seconds): goal, shots, xG (value_models.py), max xT the ball reached
(before losing it), furthest point the ball reached,
seconds until the defending team wins the ball.

Input : data/real/processed/<match>_segment_<start>.npz (export_segment.py), the team models (team_policy.py, team_decisions.py)
Output: outputs/whatif/<match>_<t>_runs.csv, _summary.csv
        --video: per scenario a GRF video next to its 'no change' twin (<scenario>_grf.mp4) and a run file
                 grf_whatif_<match>_<start>_<scenario>.npz for make_match_video.py and export_viewer.py
Needs : the GRF environment with torch (.venv)

Usage :  .venv\\Scripts\\python.exe scripts\\whatif.py --segment data\\real\\processed\\J03WOH_segment_3500.npz --start-s 3610 --seeds 20
         .venv\\Scripts\\python.exe scripts\\whatif.py ... --video-only --video-seed 0       (only the videos and run files)
         .venv\\Scripts\\python.exe scripts\\whatif.py --start-s 3610 --move Klaus 5 -3 --seeds 20   (coach: Klaus 5 m forward, 3 m right)
         .venv\\Scripts\\python.exe scripts\\whatif.py --list-players --start-s 3610                  (who is where at that moment)
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

L, W, FPS = 105.0, 68.0, 10
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "whatif")
TEAMS = ["Argentina", "France", "Croatia"]
_M = {}


def flip(p):
    return np.stack([L - p[..., 0], W - p[..., 1]], axis=-1)


def find_player(z, who):
    """Slot (0-10 home, 11-21 away) and name of a player given as part of his name or as side:shirt."""
    if ":" in who:
        side, shirt = who.split(":")
        shirts = [int(x) for x in z[f"jersey_{side}"]]
        assert int(shirt) in shirts, f"no {side} player with shirt {shirt}; shirts: {shirts}"
        k = shirts.index(int(shirt))
        return k + (11 if side == "away" else 0), str(z[f"name_{side}"][k])
    names = [str(n) for n in z["name_home"]] + [str(n) for n in z["name_away"]]
    hits = [k for k, n in enumerate(names) if who.lower() in n.lower()]
    assert len(hits) == 1, f"'{who}' matches {[names[k] for k in hits] or 'nobody'}; use more of the name, or side:shirt"
    return hits[0], names[hits[0]]


def move_text(move):
    _, name, fwd, left = move
    parts = [f"{abs(fwd):g} m {'forward' if fwd >= 0 else 'back'}" if fwd else "", f"{abs(left):g} m {'left' if left >= 0 else 'right'}" if left else ""]
    return f"{name} " + ", ".join(p for p in parts if p)


def describe(name, shift, move=None):
    if name == "move":
        return move_text(move) if move else "one player moved"
    if name == "baseline":
        return "no change"
    if name == "line_deeper":
        return f"back four {shift:g} m deeper"
    if name == "line_higher":
        return f"back four {shift:g} m higher"
    if name == "early_pass":
        return "carrier passes forward at once"
    if name in NEW_SCEN:
        return NEW_SCEN[name]
    return f"attack plays as {name[7:]}"


NEW_SCEN = {"press": "3 nearest defenders 3 m closer to the ball",
            "compact": "defending team 25% narrower",
            "wide_attack": "2 widest attackers 5 m wider",
            "striker_deeper": "most advanced attacker 5 m deeper",
            "extra_runner": "a midfielder behind the ball runs 8 m forward",
            "shoot_early": "carrier shoots from 6 m further out",
            "patient": "carrier keeps the ball twice as long",
            "quick_play": "carrier passes twice as fast",
            "team_press": "whole defending team 3 m closer to the ball",
            "low_block": "whole defending team 6 m deeper",
            "man_marking": "each defender halfway to his nearest attacker",
            "defence_wider": "defending team 25% wider",
            "keeper_off_line": "defending keeper 4 m off his line",
            "double_team": "2 nearest defenders close the carrier to 1.5 m",
            "overload_ball_side": "2 attackers 8 m towards the ball's side",
            "fullbacks_overlap": "attacking full-backs 10 m forward",
            "switch_play": "carrier switches play to the far side at once",
            "long_ball": "attacking team always plays long passes"}
SHAPE = ("press", "compact", "wide_attack", "striker_deeper", "extra_runner", "team_press", "low_block", "man_marking",
         "defence_wider", "keeper_off_line", "double_team", "overload_ball_side", "fullbacks_overlap")
DEFENCE = ("line_deeper", "line_higher", "press", "compact", "team_press", "low_block", "man_marking", "defence_wider",
           "keeper_off_line", "double_team")          # changes made by the defending team (the rest: by the attacking team)


def change_shape(pos, prev, ball, att, scen):
    """The new position scenarios, on positions (22, T, 2) with home attacking +x; changes pos and prev in place."""
    a0, d0 = (0, 11) if att == "home" else (11, 0)
    s = 1.0 if att == "home" else -1.0                                     # attacking direction along x
    b = ball[-1]
    att_out, def_out = np.arange(a0 + 1, a0 + 11), np.arange(d0 + 1, d0 + 11)
    carrier = att_out[np.argmin(np.hypot(*(pos[att_out, -1] - b).T))]
    others = att_out[att_out != carrier]
    shift = np.zeros((22, 2))
    if scen == "press":
        near = def_out[np.argsort(np.hypot(*(pos[def_out, -1] - b).T))[:3]]
        for k in near:
            v = b - pos[k, -1]
            dist = np.hypot(*v)
            shift[k] = v / max(dist, 1e-6) * min(3.0, max(dist - 1.0, 0.0))
    elif scen == "compact":
        mid = pos[def_out, -1, 1].mean()
        shift[def_out, 1] = (mid - pos[def_out, -1, 1]) * 0.25
    elif scen == "wide_attack":
        wide = others[np.argsort(-abs(pos[others, -1, 1] - W / 2))[:2]]
        shift[wide, 1] = np.sign(pos[wide, -1, 1] - W / 2) * 5.0
    elif scen == "striker_deeper":
        k = others[np.argmax(s * pos[others, -1, 0])]
        shift[k, 0] = -s * 5.0
    elif scen == "extra_runner":
        behind = others[s * pos[others, -1, 0] < s * b[0]]
        if len(behind):
            k = behind[np.argmin(np.hypot(*(pos[behind, -1] - b).T))]
            shift[k, 0] = s * 8.0
    elif scen == "team_press":
        for k in def_out:
            v = b - pos[k, -1]
            dist = np.hypot(*v)
            shift[k] = v / max(dist, 1e-6) * min(3.0, max(dist - 1.0, 0.0))
    elif scen == "low_block":
        shift[def_out, 0] = s * 6.0                                        # towards the goal the defenders protect
    elif scen == "man_marking":
        for k in def_out:
            j = att_out[np.argmin(np.hypot(*(pos[att_out, -1] - pos[k, -1]).T))]
            shift[k] = (pos[j, -1] - pos[k, -1]) * 0.5
    elif scen == "defence_wider":
        mid = pos[def_out, -1, 1].mean()
        shift[def_out, 1] = (pos[def_out, -1, 1] - mid) * 0.25
    elif scen == "keeper_off_line":
        shift[d0, 0] = -s * 4.0                                            # out of his goal, towards the play
    elif scen == "double_team":
        for k in def_out[np.argsort(np.hypot(*(pos[def_out, -1] - pos[carrier, -1]).T))[:2]]:
            v = pos[carrier, -1] - pos[k, -1]
            dist = np.hypot(*v)
            shift[k] = v / max(dist, 1e-6) * max(dist - 1.5, 0.0)
    elif scen == "overload_ball_side":
        far = others[np.argsort(-abs(pos[others, -1, 1] - b[1]))[:2]]      # the two furthest from the ball's side come over
        shift[far, 1] = np.sign(b[1] - pos[far, -1, 1]) * np.minimum(8.0, abs(b[1] - pos[far, -1, 1]))
    elif scen == "fullbacks_overlap":
        deep = others[np.argsort(s * pos[others, -1, 0])[:4]]              # the four deepest outfield players
        fb = deep[np.argsort(-abs(pos[deep, -1, 1] - W / 2))[:2]]          # the two widest of them
        shift[fb, 0] = s * 10.0
    for arr in (pos, prev):
        arr += shift[:, None, :]
        arr[..., 0] = np.clip(arr[..., 0], 0.5, L - 0.5)
        arr[..., 1] = np.clip(arr[..., 1], 0.5, W - 0.5)


def moment(seg, start_s, scen, shift, move=None):
    """The real moment as GRF sees it (home team on the left, attacking right) with the scenario's change applied.
    move = (slot, name, forward_m, left_m) for the 'move' scenario. Returns positions (22, 3, 2) at start-2 s, start-1 s, start; ball (3, 2); velocities (22+1, 3, 2); attacking side; mirrored."""
    z = np.load(seg, allow_pickle=True)
    t = z["time_s"]
    i = int(np.argmin(abs(t - start_s)))
    assert i >= 2 * FPS + 1, "the start needs 2 s of real data before it in the segment"
    P = np.concatenate([z["home"], z["away"]], axis=1).astype(float)      # (T, 22, 2)
    B = z["ball"].astype(float)
    mirrored = float(np.nanmean(z["home"][i, 0, 0])) > L / 2               # home keeper on the right: turn the pitch round
    if mirrored:
        P, B = flip(P), flip(B)
    att = "home" if str(z["owner"][i]) != "away" else "away"
    idx = [i - 2 * FPS, i - FPS, i]
    pos = P[idx].transpose(1, 0, 2).copy()     # (22, 3, 2)
    prev = P[[j - 1 for j in idx]].transpose(1, 0, 2).copy()
    ball, bprev = B[idx].copy(), B[[j - 1 for j in idx]].copy()
    parts = scen.split("+")                                                # "a+b" = two changes at once (whatif_pairs.py)
    for part in [p for p in parts if p in ("line_deeper", "line_higher")]:
        d0 = 11 if att == "home" else 0                                    # the defending team's slots
        own_goal_x = L if att == "home" else 0.0
        out = np.arange(d0 + 1, d0 + 11)                                   # outfield players
        back = out[np.argsort(abs(pos[out, -1, 0] - own_goal_x))[:4]]      # the four nearest to their own goal
        sign = np.sign(own_goal_x - L / 2) * (1 if part == "line_deeper" else -1)
        for arr in (pos, prev):
            arr[back, :, 0] = np.clip(arr[back, :, 0] + sign * shift, 0.5, L - 0.5)
    if scen == "move":
        k, _, fwd, left = move
        d = 1.0 if k < 11 else -1.0                                        # home attacks +x here; y grows to the attacker's left
        for arr in (pos, prev):
            arr[k, :, 0] = np.clip(arr[k, :, 0] + d * fwd, 0.5, L - 0.5)
            arr[k, :, 1] = np.clip(arr[k, :, 1] + d * left, 0.5, W - 0.5)
    for part in parts:
        if part in SHAPE:
            change_shape(pos, prev, ball, att, part)
    vel = np.concatenate([(pos - prev) * FPS, ((ball - bprev) * FPS)[None]], axis=0)
    return pos, ball, vel, att, mirrored, i


def play(seg, start_s, scen, shift, seed, seconds, models, video=None, move=None, defender="ours"):
    """One GRF run of a scenario. defender='tizero': the defending team is played by TiZero (open GRF agent,
    scripts/tizero_eval.py) instead of our team model; the attack is still our team model. Returns the scores and the tracks (22, T, 2), ball (T, 2) in GRF orientation."""
    import gfootball.env as football_env
    from grf_replay import register_scenario, to_metres
    from team_agent import Brain, SNAP_EVERY, THINK_EVERY, xg_proxy
    import cv2

    cfg, net, dec = models
    pos, ball, vel, att, mirrored, _ = moment(seg, start_s, scen, shift, move)
    z = np.load(seg, allow_pickle=True)
    snap = {"ball": np.clip(ball[-1], [0.5, 0.5], [L - 0.5, W - 0.5]), "home": pos[:11, -1], "away": pos[11:, -1],
            "role_home": z["role_home"], "role_away": z["role_away"]}
    register_scenario(snap)
    env = football_env.create_environment(
        env_name="real_keystate", representation="raw", render=video is not None,
        number_of_left_players_agent_controls=11, number_of_right_players_agent_controls=11,
        other_config_options={"game_engine_random_seed": int(seed)})
    obs = env.reset()
    rng = np.random.default_rng(seed)
    names = {"home": "real home", "away": "real away"}
    if scen.startswith("att_as_"):
        names[att] = scen[7:]
    brains = {"left": Brain(names["home"], names["away"], "left", cfg, net, dec, rng),
              "right": Brain(names["away"], names["home"], "right", cfg, net, dec, rng)}
    att_side = "left" if att == "home" else "right"
    tz = None
    if defender == "tizero":
        from tizero_eval import TiZero
        if "tz" not in _M:
            _M["tz"] = TiZero()
        tz = _M["tz"]
        tz.reset()
    brains[att_side].early_pass = scen in ("early_pass", "switch_play")
    brains[att_side].early_kind = "switch" if scen == "switch_play" else "advanced"
    if scen == "long_ball":
        brains[att_side].long_share = 10.0                                  # every pass is a long one
    if scen == "shoot_early":
        brains[att_side].shot_d += 6.0
    elif scen == "patient":
        brains[att_side].hold_steps *= 2
    elif scen == "quick_play":
        brains[att_side].hold_steps = max(3, brains[att_side].hold_steps // 2)
    o = obs[0]
    for side, br in brains.items():                                       # the brains remember the last 2 s of the real moment
        rown = np.array(o["left_team_roles"] if side == "left" else o["right_team_roles"])
        ropp = np.array(o["right_team_roles"] if side == "left" else o["left_team_roles"])
        for h in range(2):
            lt, rt, b = pos[:11, h], pos[11:, h], ball[h]
            vl, vr, vb = vel[:11, h], vel[11:22, h], vel[22, h]
            if side == "left":
                br.snapshot(lt, rt, b, vl, vr, vb, rown, ropp)
            else:
                br.snapshot(flip(rt), flip(lt), flip(b), -vr, -vl, -vb, rown, ropp)
    prev = (pos[:11, -1] - vel[:11, -1] / FPS, pos[11:22, -1] - vel[11:22, -1] / FPS, ball[-1] - vel[22, -1] / FPS)
    att_id = 0 if att_side == "left" else 1
    goal_x = L if att_side == "left" else 0.0
    players, balls = [], []
    res = {"scenario": scen, "seed": int(seed), "goal": 0, "goal_s": np.nan, "conceded": 0, "lost_ball_s": np.nan}
    steps = int(seconds * FPS)
    for t in range(steps + 1):
        o = obs[0]
        lt, rt = to_metres(np.array(o["left_team"])), to_metres(np.array(o["right_team"]))
        b = to_metres(np.array(o["ball"][:2]))
        if o["score"][0] + o["score"][1] > 0:                            # a goal: GRF has already reset to kick-off
            res["goal" if o["score"][att_id] > 0 else "conceded"] = 1
            res["goal_s"] = t / FPS
            break
        players.append(np.concatenate([lt, rt]))
        balls.append(b)
        if o["ball_owned_team"] == 1 - att_id and np.isnan(res["lost_ball_s"]):
            res["lost_ball_s"] = t / FPS
        if video is not None:
            fr = cv2.cvtColor(o["frame"], cv2.COLOR_RGB2BGR)
            video.append(fr)
        if t == steps:
            break
        acts = []
        for side, br in brains.items():
            if tz is not None and side != att_side:                       # TiZero defends (its own mirrored view on the right)
                acts += tz.act(obs[:11] if side == "left" else obs[11:])
                continue
            own, opp, bl = br.view(lt, rt, b)
            pown, popp, pbl = br.view(*prev)
            if t % SNAP_EVERY == 0:
                rown = np.array(o["left_team_roles"] if side == "left" else o["right_team_roles"])
                ropp = np.array(o["right_team_roles"] if side == "left" else o["left_team_roles"])
                br.snapshot(own, opp, bl, (own - pown) * FPS, (opp - popp) * FPS, (bl - pbl) * FPS, rown, ropp)
            if t % THINK_EVERY == 0:
                br.think()
            acts += br.act(o, own, opp, bl, lt if side == "left" else rt)
        prev = (lt, rt, b)
        obs, _, done, _ = env.step(acts)
        if done:
            break
    env.close()
    s = brains[att_side].stats
    bx = np.array(balls)[:, 0]
    bxy = np.array(balls)
    bxy = bxy if att_side == "left" else np.c_[L - bxy[:, 0], W - bxy[:, 1]]       # attacking towards x = 105
    keep = len(bxy) if np.isnan(res["lost_ball_s"]) else max(1, int(res["lost_ball_s"] * FPS))
    try:
        import value_models
        xt_path = value_models.xt(bxy[:keep])
        res["max_xt"] = round(float(xt_path.max()), 4)
        res["xt_gain"] = round(float(xt_path.max() - xt_path[0]), 4)
        P = np.array(players)[:keep]                                            # (T, 22, 2)
        if att_side == "left":
            A, D = P[:, :11], P[:, 11:]
        else:
            A, D = np.stack([L - P[:, 11:, 0], W - P[:, 11:, 1]], -1), np.stack([L - P[:, :11, 0], W - P[:, :11, 1]], -1)
        th = [value_models.threat(bxy[t], A[t], D[t]) for t in range(0, keep, 5)]
        res["max_threat"] = round(float(max(th)), 4) if th and value_models._load()["threat"] else np.nan
    except (OSError, ImportError, TypeError):
        res["max_xt"] = res["xt_gain"] = res["max_threat"] = np.nan
    res.update({"shots": s["shots"], "xg": round(s["xg"], 3), "passes": s["passes"],
                "furthest_m": round(float(bx.max() if att_side == "left" else L - bx.min()), 1),
                "end_ball_to_goal_m": round(float(np.hypot(balls[-1][0] - goal_x, balls[-1][1] - W / 2)), 1),
                "seconds": round(len(balls) / FPS, 1)})
    return res, np.array(players).transpose(1, 0, 2), np.array(balls), mirrored


def _init(dataset):
    import torch
    torch.set_num_threads(1)
    import team_agent
    _M["models"] = team_agent.load_models(dataset)


def _job(j):
    seg, start_s, scen, shift, seed, seconds, move, *rest = j
    try:
        return play(seg, start_s, scen, shift, seed, seconds, _M["models"], move=move, defender=rest[0] if rest else "ours")[0]
    except Exception as e:                                                # one broken run must not stop the rest
        return {"scenario": scen, "seed": seed, "error": repr(e)}


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def summarise(df, shift, move=None):
    """Per scenario: means, and the paired difference to 'no change' (same seed) with its 95% interval."""
    df = df[df.get("error").isna()] if "error" in df else df
    base = df[df["scenario"] == "baseline"].set_index("seed")
    rows = []
    for scen, g in df.groupby("scenario", sort=False):
        g = g.set_index("seed")
        r = {"scenario": scen, "what_changed": describe(scen, shift, move), "runs": len(g),
             "goal_rate": g["goal"].mean(), "shots": g["shots"].mean(), "xg": g["xg"].mean(), "max_xt": g["max_xt"].mean() if "max_xt" in g else np.nan,
             "max_threat": g["max_threat"].mean() if "max_threat" in g else np.nan,
             "furthest_m": g["furthest_m"].mean(),
             "lost_ball_rate": g["lost_ball_s"].notna().mean()}
        if scen != "baseline":
            common = g.index.intersection(base.index)
            for k in ("goal", "shots", "xg", "max_xt", "max_threat", "furthest_m"):
                if k not in g:
                    continue
                d = g.loc[common, k] - base.loc[common, k]
                r[f"d_{k}"], r[f"d_{k}_ci95"] = d.mean(), ci(d)
        rows.append(r)
    return pd.DataFrame(rows)


def chart(key):
    """Bar chart of the summary: each change against 'no change' (needs matplotlib: system Python)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sm = pd.read_csv(f"{OUT}/{key}_summary.csv")
    base = sm[sm["scenario"] == "baseline"].iloc[0]
    s = sm[sm["scenario"] != "baseline"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    y = np.arange(len(s))
    for a_, col, title, ref in ((ax[0], "goal", "Goals per run (change vs no change)", base["goal_rate"]),
                                (ax[1], "xg", "xG per run (change vs no change)", base["xg"]),
                                (ax[2], "furthest_m", "Furthest the ball got (m, change vs no change)", base["furthest_m"])):
        d, e = s[f"d_{col}"].to_numpy(), s[f"d_{col}_ci95"].to_numpy()
        a_.barh(y, d, xerr=e, capsize=4, color=["#2a78d6" if v >= 0 else "#e34948" for v in d])
        a_.axvline(0, color="0.4", lw=0.8)
        a_.set_yticks(y, s["what_changed"] if a_ is ax[0] else [""] * len(s))
        a_.set_title(f"{title}\nno change = {ref:.2f}", fontsize=10)
        a_.invert_yaxis()
    fig.suptitle(f"What-if from the real moment ({key}): {int(base['runs'])} GRF runs per scenario, bars = change, lines = 95% interval")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{key}_chart.png", dpi=120)
    print(f"wrote {OUT}/{key}_chart.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default="data/real/processed/J03WOH_segment_3500.npz")
    ap.add_argument("--start-s", type=float, default=3610.0, help="match time (s) the what-if starts from (3610: home attack 2.6 s before the real goal)")
    ap.add_argument("--seconds", type=float, default=12.0, help="how long each run plays")
    ap.add_argument("--scenarios", nargs="+",
                    default=["baseline", "line_deeper", "line_higher", "early_pass"] + [f"att_as_{t}" for t in TEAMS])
    ap.add_argument("--shift", type=float, default=3.0, help="metres the back four move (line_deeper / line_higher)")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--dataset", default="pff")
    ap.add_argument("--video-seed", type=int, default=0)
    ap.add_argument("--video-only", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--defender", choices=["ours", "tizero"], default="ours",
                    help="who plays the defending team: our team model (default) or TiZero; files get _tzdef")
    ap.add_argument("--report", action="store_true", help="only the chart of an earlier run (system Python, matplotlib)")
    ap.add_argument("--move", nargs=3, metavar=("PLAYER", "FORWARD", "LEFT"),
                    help="coach what-if: move one player (part of the name or side:shirt) FORWARD and LEFT metres; runs baseline + move")
    ap.add_argument("--list-players", action="store_true", help="print every player's name, shirt and spot at --start-s, then stop")
    a = ap.parse_args()
    if not os.path.isabs(a.segment) and not os.path.exists(a.segment):     # started from the scripts folder
        a.segment = os.path.join(ROOT, a.segment)
    os.makedirs(OUT, exist_ok=True)
    base = os.path.basename(a.segment).replace("_segment_", "_").replace(".npz", "")
    key = f"{base}_{int(a.start_s)}"
    z = np.load(a.segment, allow_pickle=True)
    if a.list_players:
        pos, ball, _, att, _, _ = moment(a.segment, a.start_s, "baseline", a.shift)
        print(f"At {a.start_s:g} s ({att} team has the ball; metres, home attacking to the right, ball at {ball[-1].round(1)}):")
        for k in range(22):
            side, j = ("home", k) if k < 11 else ("away", k - 11)
            print(f"  {side}:{int(z[f'jersey_{side}'][j]):<3d} {str(z[f'name_{side}'][j]):22s} x {pos[k, -1, 0]:5.1f}  y {pos[k, -1, 1]:5.1f}"
                  f"  {np.hypot(*(pos[k, -1] - ball[-1])):5.1f} m from the ball")
        return
    move = None
    if a.move:
        k, name = find_player(z, a.move[0])
        move = (k, name, float(a.move[1]), float(a.move[2]))
        a.scenarios = ["baseline", "move"]
        key += "_move_" + name.split(" ")[-1].replace(".", "") + f"_{a.move[1]}_{a.move[2]}"
        pos, ball, *_ = moment(a.segment, a.start_s, "move", a.shift, move)
        print(f"coach what-if: {move_text(move)} -> starts at x {pos[k, -1, 0]:.1f}, y {pos[k, -1, 1]:.1f} "
              f"({np.hypot(*(pos[k, -1] - ball[-1])):.1f} m from the ball)", flush=True)
    if a.defender == "tizero":
        key += "_tzdef"
    if a.report:
        chart(key)
        return
    if "baseline" not in a.scenarios:
        a.scenarios = ["baseline"] + a.scenarios

    if not a.video_only:
        jobs = [(a.segment, a.start_s, s, a.shift, seed, a.seconds, move, a.defender) for s in a.scenarios for seed in range(a.seeds)]
        print(f"{len(jobs)} runs ({len(a.scenarios)} scenarios x {a.seeds} seeds, {a.seconds:g} s each), {a.procs} at a time", flush=True)
        t0, rows = time.time(), []
        with Pool(a.procs, initializer=_init, initargs=(a.dataset,)) as pool:
            for k, r in enumerate(pool.imap_unordered(_job, jobs)):
                rows.append(r)
                if (k + 1) % 10 == 0 or k + 1 == len(jobs):
                    print(f"  {k + 1}/{len(jobs)} runs ({(time.time() - t0) / 60:.1f} min)", flush=True)
        df = pd.DataFrame(rows).sort_values(["scenario", "seed"])
        df.to_csv(f"{OUT}/{key}_runs.csv", index=False)
        sm = summarise(df, a.shift, move)
        sm.round(3).to_csv(f"{OUT}/{key}_summary.csv", index=False)
        print(f"\nWHAT-IF from {a.start_s:g} s ({a.seeds} runs each; d_ = change against 'no change' with the same seed, +- 95% interval)\n")
        cols = ["what_changed", "goal_rate", "shots", "xg", "max_xt", "d_goal", "d_goal_ci95", "d_xg", "d_xg_ci95", "d_max_xt", "d_max_xt_ci95",
                "d_furthest_m", "d_furthest_m_ci95"]
        print(sm[[c for c in cols if c in sm]].round(2).to_string(index=False))
        print(f"\nwrote {OUT}/{key}_runs.csv and _summary.csv")

    if a.no_video:
        return
    # videos: every scenario with the same seed, GRF's own picture, next to the 'no change' run
    import grf_names
    grf_names.setup_from_args(["--segment", a.segment])     # the real players' names in GRF's picture
    import cv2
    import team_agent
    models = team_agent.load_models(a.dataset)
    t_seg = z["time_s"]
    real = np.concatenate([z["home"], z["away"]], axis=1).transpose(1, 0, 2)
    _, _, _, _, _, i0 = moment(a.segment, a.start_s, "baseline", a.shift)
    frames, tracks = {}, {}
    for scen in a.scenarios:
        fr = []
        res, pl, bl, mirrored = play(a.segment, a.start_s, scen, a.shift, a.video_seed, a.seconds, models, video=fr, move=move)
        frames[scen] = fr
        if mirrored:
            pl, bl = flip(pl), flip(bl)
        n = pl.shape[1]
        sl = slice(i0, min(i0 + n, len(t_seg)))
        m = sl.stop - sl.start
        path = f"{OUT}/grf_whatif_{base}_{scen}.npz" if scen != "move" else f"{OUT}/grf_whatif_{key}.npz"
        np.savez_compressed(path, grf=pl[:, :m], ball_grf=bl[:m], real=real[:, sl], ball_real=z["ball"][sl], time_s=t_seg[sl],
                            live=z["live"][sl], scenario=scen, what_changed=describe(scen, a.shift, move), result=str(res))
        tracks[scen] = path
        print(f"  {scen:18s} seed {a.video_seed}: goal {res['goal']}  shots {res['shots']}  xG {res['xg']}  -> {path}", flush=True)

    def banner(img, text, colour):
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2, cv2.LINE_AA)
        return img

    for scen in a.scenarios:
        if scen == "baseline":
            continue
        n = max(len(frames["baseline"]), len(frames[scen]))
        out = f"{OUT}/{key}_{scen}_grf.mp4"
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 394))
        for k in range(n):
            pair = []
            for s, label, col in (("baseline", "NO CHANGE (team models from the real moment)", (230, 230, 230)),
                                  (scen, "WHAT-IF: " + describe(scen, a.shift, move), (80, 200, 255))):
                f = frames[s][min(k, len(frames[s]) - 1)]
                f = cv2.resize(f, (640, 360))
                top = np.zeros((34, 640, 3), np.uint8)
                pair.append(np.vstack([banner(top, label, col), f]))
            img = np.hstack(pair)
            cv2.putText(img, f"{k / FPS:4.1f} s", (600, 384), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            vw.write(img)
        vw.release()
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
