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
  att_as_<Team>    the attacking team plays with that team's model and style (e.g. att_as_Croatia)

Scores per run (attacking team's point of view, within --seconds): goal, shots, xG proxy, furthest point the ball reached,
seconds until the defending team wins the ball.

Input : data/real/processed/<match>_segment_<start>.npz (export_segment.py), the team models (team_policy.py, team_decisions.py)
Output: outputs/whatif/<match>_<t>_runs.csv, _summary.csv
        --video: per scenario a GRF video next to its 'no change' twin (<scenario>_grf.mp4) and a run file
                 grf_whatif_<match>_<start>_<scenario>.npz for make_match_video.py and export_viewer.py
Needs : the GRF environment with torch (.venv)

Usage :  .venv\\Scripts\\python.exe scripts\\whatif.py --segment data\\real\\processed\\J03WOH_segment_3500.npz --start-s 3610 --seeds 20
         .venv\\Scripts\\python.exe scripts\\whatif.py ... --video-only --video-seed 0       (only the videos and run files)
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


def describe(name, shift):
    if name == "baseline":
        return "no change"
    if name == "line_deeper":
        return f"back four {shift:g} m deeper"
    if name == "line_higher":
        return f"back four {shift:g} m higher"
    if name == "early_pass":
        return "carrier passes forward at once"
    return f"attack plays as {name[7:]}"


def moment(seg, start_s, scen, shift):
    """The real moment as GRF sees it (home team on the left, attacking right) with the scenario's change applied.
    Returns positions (22, 3, 2) at start-2 s, start-1 s, start; ball (3, 2); velocities (22+1, 3, 2); attacking side; mirrored."""
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
    if scen in ("line_deeper", "line_higher"):
        d0 = 11 if att == "home" else 0                                    # the defending team's slots
        own_goal_x = L if att == "home" else 0.0
        out = np.arange(d0 + 1, d0 + 11)                                   # outfield players
        back = out[np.argsort(abs(pos[out, -1, 0] - own_goal_x))[:4]]      # the four nearest to their own goal
        sign = np.sign(own_goal_x - L / 2) * (1 if scen == "line_deeper" else -1)
        for arr in (pos, prev):
            arr[back, :, 0] = np.clip(arr[back, :, 0] + sign * shift, 0.5, L - 0.5)
    vel = np.concatenate([(pos - prev) * FPS, ((ball - bprev) * FPS)[None]], axis=0)
    return pos, ball, vel, att, mirrored, i


def play(seg, start_s, scen, shift, seed, seconds, models, video=None):
    """One GRF run of a scenario. Returns the scores and the tracks (22, T, 2), ball (T, 2) in GRF orientation."""
    import gfootball.env as football_env
    from grf_replay import register_scenario, to_metres
    from team_agent import Brain, SNAP_EVERY, THINK_EVERY, xg_proxy
    import cv2

    cfg, net, dec = models
    pos, ball, vel, att, mirrored, _ = moment(seg, start_s, scen, shift)
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
    brains[att_side].early_pass = scen == "early_pass"
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
    seg, start_s, scen, shift, seed, seconds = j
    try:
        return play(seg, start_s, scen, shift, seed, seconds, _M["models"])[0]
    except Exception as e:                                                # one broken run must not stop the rest
        return {"scenario": scen, "seed": seed, "error": repr(e)}


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def summarise(df, shift):
    """Per scenario: means, and the paired difference to 'no change' (same seed) with its 95% interval."""
    df = df[df.get("error").isna()] if "error" in df else df
    base = df[df["scenario"] == "baseline"].set_index("seed")
    rows = []
    for scen, g in df.groupby("scenario", sort=False):
        g = g.set_index("seed")
        r = {"scenario": scen, "what_changed": describe(scen, shift), "runs": len(g),
             "goal_rate": g["goal"].mean(), "shots": g["shots"].mean(), "xg": g["xg"].mean(), "furthest_m": g["furthest_m"].mean(),
             "lost_ball_rate": g["lost_ball_s"].notna().mean()}
        if scen != "baseline":
            common = g.index.intersection(base.index)
            for k in ("goal", "shots", "xg", "furthest_m"):
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
                                (ax[1], "xg", "xG proxy per run (change vs no change)", base["xg"]),
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
    ap.add_argument("--report", action="store_true", help="only the chart of an earlier run (system Python, matplotlib)")
    a = ap.parse_args()
    if not os.path.isabs(a.segment) and not os.path.exists(a.segment):     # started from the scripts folder
        a.segment = os.path.join(ROOT, a.segment)
    os.makedirs(OUT, exist_ok=True)
    base = os.path.basename(a.segment).replace("_segment_", "_").replace(".npz", "")
    key = f"{base}_{int(a.start_s)}"
    if a.report:
        chart(key)
        return
    if "baseline" not in a.scenarios:
        a.scenarios = ["baseline"] + a.scenarios

    if not a.video_only:
        jobs = [(a.segment, a.start_s, s, a.shift, seed, a.seconds) for s in a.scenarios for seed in range(a.seeds)]
        print(f"{len(jobs)} runs ({len(a.scenarios)} scenarios x {a.seeds} seeds, {a.seconds:g} s each), {a.procs} at a time", flush=True)
        t0, rows = time.time(), []
        with Pool(a.procs, initializer=_init, initargs=(a.dataset,)) as pool:
            for k, r in enumerate(pool.imap_unordered(_job, jobs)):
                rows.append(r)
                if (k + 1) % 10 == 0 or k + 1 == len(jobs):
                    print(f"  {k + 1}/{len(jobs)} runs ({(time.time() - t0) / 60:.1f} min)", flush=True)
        df = pd.DataFrame(rows).sort_values(["scenario", "seed"])
        df.to_csv(f"{OUT}/{key}_runs.csv", index=False)
        sm = summarise(df, a.shift)
        sm.round(3).to_csv(f"{OUT}/{key}_summary.csv", index=False)
        print(f"\nWHAT-IF from {a.start_s:g} s ({a.seeds} runs each; d_ = change against 'no change' with the same seed, +- 95% interval)\n")
        cols = ["what_changed", "goal_rate", "shots", "xg", "furthest_m", "d_goal", "d_goal_ci95", "d_xg", "d_xg_ci95", "d_furthest_m", "d_furthest_m_ci95"]
        print(sm[[c for c in cols if c in sm]].round(2).to_string(index=False))
        print(f"\nwrote {OUT}/{key}_runs.csv and _summary.csv")

    if a.no_video:
        return
    # videos: every scenario with the same seed, GRF's own picture, next to the 'no change' run
    import cv2
    import team_agent
    models = team_agent.load_models(a.dataset)
    z = np.load(a.segment, allow_pickle=True)
    t_seg = z["time_s"]
    real = np.concatenate([z["home"], z["away"]], axis=1).transpose(1, 0, 2)
    _, _, _, _, _, i0 = moment(a.segment, a.start_s, "baseline", a.shift)
    frames, tracks = {}, {}
    for scen in a.scenarios:
        fr = []
        res, pl, bl, mirrored = play(a.segment, a.start_s, scen, a.shift, a.video_seed, a.seconds, models, video=fr)
        frames[scen] = fr
        if mirrored:
            pl, bl = flip(pl), flip(bl)
        n = pl.shape[1]
        sl = slice(i0, min(i0 + n, len(t_seg)))
        m = sl.stop - sl.start
        path = f"{OUT}/grf_whatif_{base}_{scen}.npz"
        np.savez_compressed(path, grf=pl[:, :m], ball_grf=bl[:m], real=real[:, sl], ball_real=z["ball"][sl], time_s=t_seg[sl],
                            live=z["live"][sl], scenario=scen, what_changed=describe(scen, a.shift), result=str(res))
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
                                  (scen, "WHAT-IF: " + describe(scen, a.shift), (80, 200, 255))):
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
