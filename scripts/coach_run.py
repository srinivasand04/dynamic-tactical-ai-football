"""Coach run: the coach tells ONE player to run from where he is to a new spot, and the team plays him when he gets there.

Unlike coach_board.py (the player is put on the new spot at the start, so you never see him go there), here he starts on his
REAL spot and has to run: the movement is the instruction. Then the team models play on in GRF (whatif.py):
    1 run     his target is the coach's spot until he is within 2.5 m of it (or 5 s pass); he sprints when it is far
    2 ball    the team's ball carrier plays him (the coach's "we find you there") as soon as he is level with the defending
              line (a through ball, so he is not offside) or has arrived; --no-pass leaves the pass to the decision model
    3 play    after that the models play as normal; while his team keeps the ball he stays within 5 m of the spot
              (his zone), and the instruction ends when the ball is lost
Each seed is played twice: 'no change' (the models play the real moment) and 'coach run', same seed, so the difference is
the effect of the instruction (95% interval over the seeds).

Output: outputs/coach_run/<key>_runs.csv, _summary.json, and viewer/data/coach_run.json (real 12 s + the typical coach run
        + its no-change twin, for viewer/coach_run.html and scripts/record_coach_run.py)
Usage : .venv\\Scripts\\python.exe scripts\\coach_run.py                                  (Klaus in behind the left-back, J03WOH 1621 s)
        .venv\\Scripts\\python.exe scripts\\coach_run.py --who Klaus --to 88 20 --start-s 3554 --seeds 20
        (--to X Y: metres, his team attacking to the right, Y grows to his left)
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

os.environ.setdefault("GRF_PACE", "1")      # real running pace (grf_follow.py): without it GRF players run ~2x real speed and the
                                             # movement model, fed those speeds, keeps sending runners on (off the pitch)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import whatif as WI  # noqa: E402

ROOT = WI.ROOT
OUT = os.path.join(ROOT, "outputs", "coach_run")
VIEW = os.path.join(ROOT, "viewer", "data", "coach_run.json")
L, W, FPS = WI.L, WI.W, WI.FPS
ARRIVE_M, MAX_RUN_S, ZONE_M = 2.5, 5.0, 5.0
_R = {"slot": None, "to": None, "pass": True, "brains": []}               # this process's instruction


# ---------------------------------------------------------------- the instruction, inside the team brain

def _patch():
    import team_agent
    from team_agent import face_action, FACE, SHORT, LONG
    if getattr(team_agent.Brain, "_run_patched", False):
        return
    init, act = team_agent.Brain.__init__, team_agent.Brain.act

    def init_run(self, *a, **k):
        init(self, *a, **k)
        self.run_k, self.run_state, self.run_t = None, "off", 0
        self.run_log = {"arrive_s": np.nan, "pass_s": np.nan, "pass_from": None, "end_s": np.nan, "end_why": ""}
        _R["brains"].append(self)

    def act_run(self, o, own, opp, ball, own_abs):
        if self.targets is not None:                                       # both runs: targets stay on the pitch
            self.targets = np.clip(self.targets, [0.5, 0.5], [L - 0.5, W - 0.5])
        k = self.run_k
        if k is None:
            return act(self, o, own, opp, ball, own_abs)
        self.run_t += 1
        t = self.run_t / FPS
        side_id = 0 if self.side == "left" else 1
        ours = o["ball_owned_team"] == side_id
        lost = o["ball_owned_team"] == 1 - side_id
        to = self.run_to
        if self.run_state in ("run", "zone") and (lost or o["game_mode"] != 0):
            self.run_state = "off"
            self.run_log.update(end_s=t, end_why="ball lost" if lost else "stoppage")
        if self.run_state == "run":
            d = float(np.hypot(*(own[k] - to)))
            if d < ARRIVE_M or t >= MAX_RUN_S:
                self.run_state = "zone"
                self.run_log["arrive_s"] = t
        if self.targets is None:
            self.targets = own.copy()
        if self.run_state == "run":
            self.targets[k] = to
        elif self.run_state == "zone":
            off = self.targets[k] - to
            r = float(np.hypot(*off))
            if r > ZONE_M:
                self.targets[k] = to + off * ZONE_M / r
        # the ball: as soon as he is level with the second-last defender (or has arrived), the carrier plays him
        if self.run_pass and np.isnan(self.run_log["pass_s"]) and self.run_state in ("run", "zone") and ours:
            c = o["ball_owned_player"]
            line = np.sort(opp[1:, 0])[-1] if len(opp) > 1 else L                          # deepest outfield opponent
            level = own[k][0] >= min(line, L - 16.5) - 1.0
            if c is not None and c >= 0 and c != k and not self.plan and (level or self.run_state == "zone"):
                far = np.hypot(*(own[k] - own[c])) > 25
                lead = own[k] + (to - own[k]) * 0.35                                       # aim a little ahead of him
                self.plan = [face_action(own[c], lead)] * FACE + [LONG if far else SHORT]
                self.stats["passes"] += 1
                self.pending_pass, self.hold, self.carrier = True, 0, c
                self.run_log.update(pass_s=t, pass_from=int(c))
        return act(self, o, own, opp, ball, own_abs)

    team_agent.Brain.__init__, team_agent.Brain.act = init_run, act_run
    team_agent.Brain._run_patched = True


def play(seg, start_s, scen, seed, seconds, models):
    """One GRF run; scen 'baseline' or 'run'. Returns scores, tracks (22, T, 2), ball (T, 2) in GRF orientation."""
    _patch()
    import team_agent
    _R["brains"] = []
    init = team_agent.Brain.__init__

    def init_on(self, *a, **k):                                            # switch the instruction on in the right brain
        init(self, *a, **k)
        s = _R["slot"]
        if scen == "run" and s is not None and (self.side == "left") == (s < 11):
            self.run_k, self.run_state = s % 11, "run"
            self.run_to = np.asarray(_R["to"], float)
            self.run_pass = _R["pass"]
    team_agent.Brain.__init__ = init_on
    try:
        res, pl, bl, mirrored = WI.play(seg, start_s, "baseline", 3.0, seed, seconds, models)
    finally:
        team_agent.Brain.__init__ = init
    res["scenario"] = scen
    br = [b for b in _R["brains"] if b.run_k is not None]
    log = br[0].run_log if br else {}
    res.update({k: v for k, v in log.items() if k != "pass_from"})
    s = _R["slot"]
    if s is None:                                                           # a search's shared 'no change' run: no runner
        return res, pl, bl, mirrored
    pk = pl[s]                                                              # GRF frame: home attacks +x (moment() turned it)
    to_abs = np.asarray(_R["to"], float) if s < 11 else WI.flip(np.asarray(_R["to"], float))
    res["closest_to_spot_m"] = round(float(np.min(np.hypot(*(pk - to_abs).T))), 1)
    res["run_m"] = round(float(np.sum(np.hypot(*np.diff(pk[: int(4 * FPS)], axis=0).T))), 1)
    T = min(pl.shape[1], len(bl))
    near = np.hypot(*(pl[:, :T] - bl[None, :T]).transpose(2, 0, 1))      # (22, T)
    got = (near.argmin(0) == s) & (near[s] < 1.2)
    res["touched"] = int(got.any())
    res["touch_s"] = round(float(np.argmax(got) / FPS), 1) if got.any() else np.nan
    return res, pl, bl, mirrored


def _init(dataset, slot, to, pas):
    import torch
    torch.set_num_threads(1)
    import grf_follow
    grf_follow.PACE = True
    import team_agent
    WI._M["models"] = team_agent.load_models(dataset)
    _R.update(slot=slot, to=to, **{"pass": pas})


def _job(j):
    seg, start_s, scen, seed, seconds = j
    try:
        return play(seg, start_s, scen, seed, seconds, WI._M["models"])[0]
    except Exception as e:
        return {"scenario": scen, "seed": seed, "error": repr(e)}


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 else (float("nan"), float("nan"))


def match_clock(match, t):
    """Tracking time -> the match clock (mm:ss), from the period starts of the match file (second half starts at 45:00)."""
    f = os.path.join(ROOT, "data", "real", "processed", f"{match}.csv.gz")
    c0, at = 0.0, 0.0
    if os.path.exists(f):
        st = pd.read_csv(f, usecols=["time_s", "period"]).groupby("period")["time_s"].min()
        for per, t0 in st.items():
            if t0 <= t:
                c0, at = float(t0), {1: 0, 2: 45, 3: 90, 4: 105}.get(int(per), 0) * 60.0
    c = t - c0 + at
    return f"{int(c // 60)}:{int(c % 60):02d}"


def real_part(z, i, seconds, slot, to):
    """The real 12 s from the moment (GRF orientation) and what the player actually did."""
    P = np.concatenate([z["home"], z["away"]], axis=1).astype(float)
    B = z["ball"].astype(float)
    if float(np.nanmean(z["home"][i, 0, 0])) > L / 2:
        P, B = WI.flip(P), WI.flip(B)
    n = int(seconds * FPS) + 1
    pl, bl = P[i: i + n].transpose(1, 0, 2), B[i: i + n]
    to_abs = np.asarray(to, float) if slot < 11 else WI.flip(np.asarray(to, float))
    pk = pl[slot]
    own = z["owner"][i: i + n]
    att = "home" if str(z["owner"][i]) != "away" else "away"
    lost = [j for j in range(n) if str(own[j]) not in (att, "None", "", "nan")]
    return pl, bl, {"closest_to_spot_m": round(float(np.nanmin(np.hypot(*(pk - to_abs).T))), 1),
                    "run_m": round(float(np.nansum(np.hypot(*np.diff(pk[: int(4 * FPS)], axis=0).T))), 1),
                    "lost_ball_s": round(lost[0] / FPS, 1) if lost else None}


# ---------------------------------------------------------------- player model: how often does he make this run anyway?

def runs_in_behind(match_ids, fps=10):
    """Per player (dfl_id): minutes and runs in behind = while his team has the ball, he crosses the deepest outfield
    opponent's line in the opponents' half, moving forward at 4 m/s or more (one run per 3 s). Real tracking, all matches."""
    rows = []
    for m in match_ids:
        f = os.path.join(ROOT, "data", "real", "processed", f"{m}.csv.gz")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, usecols=["frame", "team", "role", "dfl_id", "x_m", "ball_owner_team", "game_mode", "period"])
        for (per, team), g in d.groupby(["period", "team"]):
            opp = d[(d.period == per) & (d.team != team)]
            gk = g[g.role == "GK"].x_m.mean()
            sgn = 1.0 if gk < L / 2 else -1.0                                  # attack direction this half
            X = g.pivot_table(index="frame", columns="dfl_id", values="x_m")
            ox = opp[opp.role != "GK"].groupby("frame").x_m.agg("max" if sgn > 0 else "min").reindex(X.index)
            fr = g.drop_duplicates("frame").set_index("frame").reindex(X.index)
            ours = (fr.ball_owner_team == team) & (fr.game_mode == "normal")
            ax = (X.values - (0 if sgn > 0 else L)) * sgn                       # metres, attacking towards +x
            line = ((ox.values - (0 if sgn > 0 else L)) * sgn)[:, None]
            v = np.vstack([np.full((1, ax.shape[1]), np.nan), np.diff(ax, axis=0) * fps])
            cross = (ax >= line) & (np.vstack([np.full((1, ax.shape[1]), False), ax[:-1] < line[:-1]]))                 & (v >= 4.0) & (line > L / 2) & ours.values[:, None]
            for j, pid in enumerate(X.columns):
                ts = np.flatnonzero(cross[:, j])
                n, last = 0, -1e9
                for t in ts:
                    if t - last >= 3 * fps:
                        n, last = n + 1, t
                role = g[g.dfl_id == pid].role.mode()
                rows.append({"dfl_id": pid, "match": m, "role": role.iloc[0] if len(role) else "", "minutes": np.isfinite(ax[:, j]).sum() / fps / 60, "runs": n})
    R = pd.DataFrame(rows).groupby("dfl_id").agg(minutes=("minutes", "sum"), runs=("runs", "sum"), matches=("match", "nunique"),
                                                 role=("role", lambda r: r.mode().iloc[0]))
    R["per90"] = R.runs / R.minutes * 90
    return R


def step_habits(name, match=None):
    """Adds his player model (movement model + runs in behind vs other wide midfielders) to viewer/data/coach_run.json."""
    import settings
    d = json.load(open(VIEW, encoding="utf-8"))
    pm = json.load(open(os.path.join(ROOT, "viewer", "data", "player_models.json"), encoding="utf-8"))
    pid, mv = next(((k, v) for k, v in pm["movement"].items() if v["name"] == name), (None, None))
    if pid is None:                                                         # no movement model: find his id in the DFL player lists
        import glob
        for f in sorted(glob.glob(os.path.join(ROOT, "data", "real", "raw", f"{match or '*'}_players.csv"))):
            pl = pd.read_csv(f, encoding="utf-8")
            hit = pl[pl["name"] == name]
            if len(hit):
                pid = str(hit["player_id"].iloc[0])
                break
    matches = [m for m in settings.matches() if os.path.exists(os.path.join(ROOT, "data", "real", "processed", f"{m}.csv.gz"))]         if hasattr(settings, "matches") else ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
    R = runs_in_behind(matches)
    wide = R[R.role.isin(["LM", "RM", "LW", "RW"]) & (R.minutes >= 90)]
    me = R.loc[pid] if pid in R.index else None
    out = {"name": name, "dfl_id": pid, "movement": mv,
           "behind90": round(float(me.per90), 2) if me is not None else None, "behind_runs": int(me.runs) if me is not None else None,
           "minutes": round(float(me.minutes)) if me is not None else None, "matches": int(me.matches) if me is not None else None,
           "wide_median90": round(float(wide.per90.median()), 2), "wide_n": int(len(wide)),
           "pct": round(float((wide.per90 < me.per90).mean() * 100)) if me is not None else None,
           "top_wide": [[str(i), round(float(r.per90), 2)] for i, r in wide.sort_values("per90", ascending=False).head(3).iterrows()]}
    d["player"] = out
    json.dump(d, open(VIEW, "w", encoding="utf-8"), separators=(",", ":"))
    print(f"{name}: {out['behind_runs']} runs in behind in {out['minutes']} min ({out['behind90']}/90), "
          f"wide midfielders median {out['wide_median90']}/90 (n={out['wide_n']}), percentile {out['pct']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default="data/real/processed/J03WOH_segment_1589.npz")
    ap.add_argument("--start-s", type=float, default=1621.0, help="1621: Hoffmann has the ball, 12 m free; in the real match it goes wide to Klaus and the attack dies")
    ap.add_argument("--who", default="Klaus", help="part of the name, or side:shirt")
    ap.add_argument("--to", nargs=2, type=float, default=[95.0, 22.0], metavar=("X", "Y"),
                    help="the coach's spot, metres, his team attacking to the right (Y grows to his left)")
    ap.add_argument("--no-pass", action="store_true", help="no 'we play you': the decision model alone chooses passes")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0, help="first seed (use new seeds to check a scenario picked by coach_search.py)")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--dataset", default="pff")
    ap.add_argument("--habits-only", action="store_true", help="only add his player model (runs in behind, real tracking) to the viewer file")
    ap.add_argument("--out", default="coach_run", help="viewer file name: viewer/data/<out>.json (coach_run.html?data=<out>)")
    ap.add_argument("--label", default="run in behind", help="the instruction in a few words, for the video")
    ap.add_argument("--show", choices=["median", "goal"], default="median",
                    help="video run: the median one (by threat) of the runs where he got the ball, or one that scored")
    a = ap.parse_args()
    global VIEW
    VIEW = os.path.join(ROOT, "viewer", "data", f"{a.out}.json")
    seg = a.segment if os.path.isabs(a.segment) else os.path.join(ROOT, a.segment)
    z = np.load(seg, allow_pickle=True)
    slot, name = WI.find_player(z, a.who)
    if a.habits_only:
        return step_habits(name, os.path.basename(seg).split("_")[0])
    side = "home" if slot < 11 else "away"
    pos, ball, _, att, mirrored, i = WI.moment(seg, a.start_s, "baseline", 3.0)
    to = np.array(a.to, float)
    start = pos[slot, -1] if slot < 11 else WI.flip(pos[slot, -1])        # his team's view
    key = f"{os.path.basename(seg).replace('_segment_', '_').replace('.npz', '')}_{int(a.start_s)}_{name.split(' ')[-1].replace('.', '')}"
    os.makedirs(OUT, exist_ok=True)
    print(f"coach run: {name} ({side}) from ({start[0]:.0f}, {start[1]:.0f}) to ({to[0]:.0f}, {to[1]:.0f}), "
          f"{np.hypot(*(to - start)):.0f} m; {att} team has the ball; {'we play him' if not a.no_pass else 'no forced pass'}", flush=True)

    t0 = time.time()
    jobs = [(seg, a.start_s, s, seed, a.seconds) for seed in range(a.seed0, a.seed0 + a.seeds) for s in ("baseline", "run")]
    with Pool(a.procs, initializer=_init, initargs=(a.dataset, slot, tuple(to), not a.no_pass)) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"{key}_runs.csv"), index=False)
    if "error" in df and df["error"].notna().any():
        print(df[df["error"].notna()][["scenario", "seed", "error"]].head().to_string())
    print(f"{len(df)} runs in {time.time() - t0:.0f}s", flush=True)

    b = df[df.scenario == "baseline"].set_index("seed")
    r = df[df.scenario == "run"].set_index("seed")
    seeds = b.index.intersection(r.index)
    metrics = [("max_threat", "Max threat"), ("xg", "xG"), ("shots", "Shots"), ("goal", "Goals"),
               ("touched", f"{name.split(' ')[-1]} on the ball"), ("max_xt", "Max xT")]
    summ = []
    for m, lab in metrics:
        if m not in r:
            continue
        mb, mr = b.loc[seeds, m].astype(float), r.loc[seeds, m].astype(float)
        d, h = ci(mr.values - mb.values)
        verdict = "no clear effect" if not np.isfinite(h) or abs(d) <= h else ("up" if d > 0 else "down")
        summ.append({"metric": m, "what": lab, "no_change": round(float(mb.mean()), 4), "coach_run": round(float(mr.mean()), 4),
                     "diff": round(d, 4), "ci95": round(h, 4), "verdict": verdict})
        print(f"  {lab:32s} no change {mb.mean():7.3f}   coach run {mr.mean():7.3f}   diff {d:+.3f} +- {h:.3f}  {verdict}")
    lost_b, lost_r = b.loc[seeds, "lost_ball_s"].fillna(a.seconds), r.loc[seeds, "lost_ball_s"].fillna(a.seconds)
    run_stats = {"arrived": float(r["arrive_s"].notna().mean()), "arrive_s": float(r["arrive_s"].median()),
                 "passed": float(r["pass_s"].notna().mean()), "pass_s": float(r["pass_s"].median()),
                 "touched": float(r["touched"].mean()), "closest_m": float(r["closest_to_spot_m"].median()),
                 "keep_ball_s": [round(float(lost_b.mean()), 1), round(float(lost_r.mean()), 1)]}
    print(f"  arrived {run_stats['arrived']:.0%} (median {run_stats['arrive_s']:.1f} s), played {run_stats['passed']:.0%} "
          f"(median {run_stats['pass_s']:.1f} s), on the ball {run_stats['touched']:.0%}; "
          f"ball kept {lost_b.mean():.1f} s -> {lost_r.mean():.1f} s")

    # the run shown in the video: the coach run with the median max threat (not the best one), and its no-change twin
    rr = r.loc[seeds].copy()
    if a.show == "goal" and rr["goal"].sum() > 0:                           # a run that scored (the video says how often that happens)
        show = int(rr[rr["goal"] == 1].index[0])
        why = f"first of the {int(rr['goal'].sum())} runs (of {len(rr)}) that scored"
    else:
        if rr["touched"].sum() > 0 and not a.no_pass:
            rr = rr[rr["touched"] == 1]                                     # among runs where the plan happened at all
        med = rr["max_threat"].median()
        show = int((rr["max_threat"] - med).abs().sort_values(kind="stable").index[0])
        why = f"median max threat {med:.3f} of {len(rr)} runs where he got the ball"
    print(f"video run: seed {show} ({why})")
    _init(a.dataset, slot, tuple(to), not a.no_pass)
    res_r, pl_r, bl_r, _ = play(seg, a.start_s, "run", show, a.seconds, WI._M["models"])
    res_b, pl_b, bl_b, _ = play(seg, a.start_s, "baseline", show, a.seconds, WI._M["models"])
    pl_real, bl_real, real = real_part(z, i, a.seconds, slot, to)

    names = [str(n) for n in z["name_home"]] + [str(n) for n in z["name_away"]]
    names_all = names
    shirts = [int(x) for x in z["jersey_home"]] + [int(x) for x in z["jersey_away"]]
    roles = [str(x) for x in z["role_home"]] + [str(x) for x in z["role_away"]]
    r2 = lambda arr: np.round(np.nan_to_num(arr, nan=-99.0), 2).tolist()
    to_abs = to if slot < 11 else WI.flip(to)
    reg = pd.read_csv(os.path.join(ROOT, "data", "real", "matches.csv"), encoding="utf-8").set_index("match_id")
    mid = key.split("_")[0]
    home, away = (str(reg.at[mid, "home"]), str(reg.at[mid, "away"])) if mid in reg.index else ("Home", "Away")
    P0 = np.concatenate([z["home"], z["away"]], axis=1).astype(float)[i]
    car = int(np.nanargmin(np.hypot(*(P0[(0 if att == "home" else 11):(11 if att == "home" else 22)] - z["ball"][i][:2]).T))) + (0 if att == "home" else 11)
    free = float(np.nanmin(np.hypot(*(P0[(11 if att == "home" else 0):(22 if att == "home" else 11)] - z["ball"][i][:2]).T)))
    out = {"match": mid, "segment": os.path.basename(seg)[:-4], "home": home, "away": away, "label": a.label, "show": a.show, "show_why": why,
           "carrier": names_all[car], "carrier_free_m": round(free, 1), "start_s": a.start_s, "clock": match_clock(mid, a.start_s),
           "fps": FPS, "names": names, "shirts": shirts, "roles": roles, "slot": slot, "who": name, "side": side, "att": att,
           "from": pos[slot, -1].round(2).tolist(), "to": np.round(to_abs, 2).tolist(), "pass": not a.no_pass,
           "instruction": f"{name.split(' ')[-1]}: {a.label} from ({start[0]:.0f}, {start[1]:.0f}) to ({to[0]:.0f}, {to[1]:.0f}); we play you there",
           "real": {"p": r2(pl_real), "b": r2(bl_real), **real},
           "run": {"p": r2(pl_r), "b": r2(bl_r), "seed": show, **{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in res_r.items()
                                                               if k in ("arrive_s", "pass_s", "touch_s", "end_s", "end_why", "shots", "xg", "goal", "goal_s", "max_threat", "closest_to_spot_m", "run_m", "lost_ball_s", "touched")}},
           "twin": {"p": r2(pl_b), "b": r2(bl_b), **{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in res_b.items()
                                                     if k in ("shots", "xg", "goal", "goal_s", "max_threat", "lost_ball_s")}},
           "summary": summ, "run_stats": run_stats, "seeds": int(len(seeds))}
    json.dump(out, open(VIEW, "w", encoding="utf-8"), separators=(",", ":"))
    json.dump({k: out[k] for k in ("instruction", "summary", "run_stats", "seeds")} | {"video_seed": show, "real": real},
              open(os.path.join(OUT, f"{key}_summary.json"), "w"), indent=1)
    print(f"wrote {VIEW} ({os.path.getsize(VIEW) // 1024} KB); real: {name.split(' ')[-1]} came within {real['closest_to_spot_m']} m of the spot, "
          f"ran {real['run_m']} m in 4 s, ball lost at {real['lost_ball_s']} s")
    step_habits(name, mid)


if __name__ == "__main__":
    main()
