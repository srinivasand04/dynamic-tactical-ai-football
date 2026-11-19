"""Coach board: the coach drags players in the viewer, presses Play, and GRF plays the moment out from there.

Three boards (who may be dragged), split by where the team's outfield players stand at that moment:
    forward    the 3 most advanced outfield players
    midfield   the next 3
    defence    the 4 deepest outfield players + the keeper
Two ways to play it:
    once       players start at the new spots, then the team model plays on as normal (it may pull them back)
    hold       the coach's instruction lasts: every time the team model picks targets, the moved players keep the same
               offset from what the model wants (a "stay 6 m higher" instruction for the whole run)

The board file (JSON, written by the viewer or by hand):
    {"match": "J03WOH", "t": 3610.0, "board": "defence", "mode": "hold", "team": "away",
     "moves": [{"side": "away", "shirt": 4, "x": 80.0, "y": 30.0}, ...]}        metres, same frame as the viewer
Each run: 'no change' + 'board', same seeds, 12 s, scored with the threat model (max from 1 s after the start while the team
has the ball, so the dragged starting picture itself does not count), xG, shots and time on the ball.
Result: a viewer file with the real moment + 'no change' + 'your board' ghosts and a result card.

Usage : .venv\\Scripts\\python.exe scripts\\coach_board.py --board outputs\\coach_board\\board.json
        .venv\\Scripts\\python.exe scripts\\coach_board.py --units J03WOH 3610 away     (who is on each board)
The viewer server (scripts\\viewer_server.py) runs this for you when you press Play on the board.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import whatif as WI  # noqa: E402

ROOT = WI.ROOT
OUT = os.path.join(ROOT, "outputs", "coach_board")
L, W = WI.L, WI.W
BOARDS = ("forward", "midfield", "defence")
_B = {"delta": {}, "hold": False}          # this run's moves (GRF frame, slot -> [dx, dy]); filled in every process
HOLD_SCENS = {"board"}                     # scenarios whose moved players hold their offset (other scripts may add theirs)


# ---------------------------------------------------------------- which players are on each board

def units(pos_now, team_slots, own_goal_x):
    """pos_now (22, 2) in metres; team_slots: the 11 slots of the team (keeper first). Returns {board: [slots]}."""
    keeper, out = team_slots[0], np.array(team_slots[1:])
    depth = np.abs(pos_now[out, 0] - own_goal_x)
    order = out[np.argsort(depth)]                                         # deepest first
    return {"defence": [int(keeper)] + [int(s) for s in order[:4]], "midfield": [int(s) for s in order[4:7]],
            "forward": [int(s) for s in order[7:]]}


def find_segment(match, t):
    """A processed segment of the match with 2 s of data before t and 12 s after."""
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "real", "processed", f"{match}_segment_*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "name_home" in z.files and z["time_s"][0] + 2.2 <= t <= z["time_s"][-1] - 12:
            return f
    return None


def ensure_segment(match, t):
    seg = find_segment(match, t)
    if seg is None:                                                        # cut one (3 min) that starts 20 s before t
        py = os.path.join(ROOT, ".venv-analysis", "Scripts", "python.exe")
        subprocess.run([py, os.path.join(ROOT, "scripts", "export_segment.py"), "--match", match, "--start-s", str(int(t - 20)),
                        "--minutes", "3"], cwd=ROOT, check=True)
        seg = find_segment(match, t)
    assert seg, f"no segment of {match} around {t} s"
    return seg


# ---------------------------------------------------------------- the moment with the board applied (patched into whatif)

_moment = WI.moment


def moment(seg, start_s, scen, shift, move=None):
    pos, ball, vel, att, mirrored, i = _moment(seg, start_s, "baseline" if scen == "board" else scen, shift, move)
    if scen == "board":
        for k, d in _B["delta"].items():
            pos[int(k)] = np.clip(pos[int(k)] + np.asarray(d), [0.5, 0.5], [L - 0.5, W - 0.5])   # all 3 frames: same speed
    return pos, ball, vel, att, mirrored, i


WI.moment = moment
_describe = WI.describe
WI.describe = lambda s, shift, move=None: "your board" if s == "board" else _describe(s, shift, move)


def _patch_hold():
    """'hold': each think, the moved players keep the coach's offset on top of the team model's target."""
    import team_agent
    if getattr(team_agent.Brain, "_board_patched", False):
        return
    think = team_agent.Brain.think

    def think_hold(self):
        think(self)
        if self.targets is None or not _B["hold"] or not getattr(self, "board_on", False):
            return
        for k, d in _B["delta"].items():
            k = int(k)
            own = k if self.side == "left" else k - 11
            if 0 <= own < 11:
                dd = np.asarray(d) if self.side == "left" else -np.asarray(d)        # the right brain sees the pitch turned round
                self.targets[own] = np.clip(self.targets[own] + dd, [0.5, 0.5], [L - 0.5, W - 0.5])
    team_agent.Brain.think = think_hold
    team_agent.Brain._board_patched = True


_play = WI.play


def play(seg, start_s, scen, shift, seed, seconds, models, video=None, move=None):
    """whatif.play, with the 'hold' instruction switched on for the board run only."""
    import team_agent
    _patch_hold()
    init = team_agent.Brain.__init__

    def init_on(self, *a, **k):
        init(self, *a, **k)
        self.board_on = scen in HOLD_SCENS
    team_agent.Brain.__init__ = init_on
    try:
        res, pl, bl, mirrored = _play(seg, start_s, scen, shift, seed, seconds, models, video=video, move=move)
    finally:
        team_agent.Brain.__init__ = init
    att = _moment(seg, start_s, "baseline", shift)[3]
    res["threat_play"], res["on_ball"] = threat_from(pl, bl, att)
    res.update(extra_metrics(pl, bl, att))
    return res, pl, bl, mirrored


def extra_metrics(pl, bl, att, from_s=1.0, hold_m=3.0, regain_s=1.0):
    """Metrics over the whole run, not one instant (less noisy than a maximum):
        threat_auc   mean threat of the attacking team over the run from 1 s (0 while it does not have the ball), pp-ready 0-1
        box_ctrl     share of the defending penalty area nearer to a defender than to an attacker (1 m grid, every 1 s)
        t_regain     first time (s) the defence holds the ball for 1 s (nearest player within 3 m); the run length if never
    pl (22, T, 2), bl (T, 2) in GRF orientation (home on the left)."""
    import value_models
    T = min(pl.shape[1], len(bl))
    P, B = pl[:, :T].transpose(1, 0, 2), bl[:T]
    if att == "home":
        A, D = P[:, :11], P[:, 11:]
    else:
        A, D, B = WI.flip(P[:, 11:]), WI.flip(P[:, :11]), WI.flip(B)
    gx, gy = np.meshgrid(np.arange(L - 16.0, L, 1.0), np.arange(W / 2 - 19.66, W / 2 + 20.16, 1.0))
    box = np.c_[gx.ravel(), gy.ravel()]
    own, th, ctrl, run, regain = 1, [], [], 0, np.nan
    start = int(from_s * WI.FPS)
    for t in range(T):
        da, dd = np.nanmin(np.hypot(*(A[t] - B[t]).T)), np.nanmin(np.hypot(*(D[t] - B[t]).T))
        if min(da, dd) < hold_m:
            own = 1 if da <= dd else 0
        if t < start:
            continue
        run = run + 1 if own == 0 else 0
        if np.isnan(regain) and run >= regain_s * WI.FPS:
            regain = (t - run + 1) / WI.FPS
        if t % 5 == 0:
            x = value_models.threat(B[t], A[t], D[t]) if own else 0.0
            th.append(float(x) if x is not None and np.isfinite(x) else 0.0)
        if t % 10 == 0:
            a_ok, d_ok = A[t][np.isfinite(A[t][:, 0])], D[t][np.isfinite(D[t][:, 0])]
            if len(a_ok) and len(d_ok):
                na = np.min(np.hypot(*(box[:, None] - a_ok[None]).transpose(2, 0, 1)), 1)
                nd = np.min(np.hypot(*(box[:, None] - d_ok[None]).transpose(2, 0, 1)), 1)
                ctrl.append(float((nd < na).mean()))
    return {"threat_auc": round(float(np.mean(th)), 4) if th else np.nan,
            "box_ctrl": round(float(np.mean(ctrl)), 4) if ctrl else np.nan,
            "t_regain": round(float(regain if np.isfinite(regain) else T / WI.FPS), 1)}


def threat_from(pl, bl, att, from_s=1.0, hold_m=3.0):
    """The run judged after the dragged starting picture: from 1 s on, who has the ball is the team of the player nearest
    to it (within 3 m; otherwise the last team that had it). Returns (max threat of the attacking team while it has the
    ball, share of the time it has the ball). GRF's own 'ball lost' fires on the first brief touch, so it is not used.
    pl (22, T, 2), bl (T, 2) in GRF orientation (home on the left)."""
    import value_models
    T = min(pl.shape[1], len(bl))
    P, B = pl[:, :T].transpose(1, 0, 2), bl[:T]
    if att == "home":
        A, D = P[:, :11], P[:, 11:]
    else:
        A, D, B = WI.flip(P[:, 11:]), WI.flip(P[:, :11]), WI.flip(B)
    own, ours, th = 1, [], []
    for t in range(T):
        da, dd = np.nanmin(np.hypot(*(A[t] - B[t]).T)), np.nanmin(np.hypot(*(D[t] - B[t]).T))
        if min(da, dd) < hold_m:
            own = 1 if da <= dd else 0
        if t >= from_s * WI.FPS:
            ours.append(own)
            if own and t % 5 == 0:
                x = value_models.threat(B[t], A[t], D[t])
                if x is not None and np.isfinite(x):
                    th.append(x)
    return (round(float(max(th)), 4) if th else 0.0), (round(float(np.mean(ours)), 3) if ours else np.nan)


WI.play = play


def _init(dataset, board):
    _B.update(board)
    WI._init(dataset)


# ---------------------------------------------------------------- one board

def board_deltas(b, seg):
    """Board moves (viewer frame, by shirt) -> GRF-frame offsets per segment slot, checked against the chosen board."""
    z = np.load(seg, allow_pickle=True)
    pos, ball, _, att, mirrored, i = _moment(seg, b["t"], "baseline", 3.0)
    now_real = np.concatenate([z["home"][i], z["away"][i]]).astype(float)
    team = b.get("team") or b["moves"][0]["side"]
    slots = list(range(11)) if team == "home" else list(range(11, 22))
    own_goal_x = L if np.nanmean(now_real[slots[0], 0]) > L / 2 else 0.0
    allowed = units(now_real, slots, own_goal_x)[b["board"]]
    delta, names = {}, []
    for mv in b["moves"]:
        k, name = WI.find_player(z, f"{mv['side']}:{int(mv['shirt'])}")
        if k not in allowed:
            print(f"  note: {name} is not on the {b['board']} board at this moment (moved anyway)")
        d = np.array([mv["x"], mv["y"]], float) - now_real[k]                   # viewer / real frame
        if mirrored:
            d = -d                                                             # GRF frame: home on the left
        if np.hypot(*d) > 0.3:
            delta[k] = d.round(2).tolist()
            names.append(f"{name.split(' ')[-1]} {np.hypot(*d):.0f} m")
    return delta, names, att, team


def card(df, att, team):
    """Result card: 'no change' vs 'your board', same seeds, from the point of view of the team on the board."""
    df = df[df["error"].isna()] if "error" in df else df
    base, brd = df[df.scenario == "baseline"].set_index("seed"), df[df.scenario == "board"].set_index("seed")
    common = base.index.intersection(brd.index)
    rows = []
    for k, label in (("threat_play", "Threat (from 1 s)"), ("xg", "xG"), ("shots", "Shots per run"), ("on_ball", "Time on the ball")):
        a, b_ = base.loc[common, k].astype(float), brd.loc[common, k].astype(float)
        d = b_ - a
        rows.append({"what": label, "no_change": float(a.mean()), "board": float(b_.mean()), "diff": float(d.mean()), "ci95": float(WI.ci(d))})
    attacking = team == att                                    # the team on the board has the ball: up is good; else down is good
    for r in rows:
        good = r["diff"] > 0 if attacking else r["diff"] < 0
        r["verdict"] = "no clear effect" if not np.isfinite(r["ci95"]) or abs(r["diff"]) <= r["ci95"] else ("better" if good else "worse")
    return {"runs": int(len(common)), "team_on_board": team, "attacking": bool(attacking), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", help="board JSON (from the viewer)")
    ap.add_argument("--units", nargs=3, metavar=("MATCH", "T", "TEAM"), help="print who is on each board, then stop")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--dataset", default="pff")
    a = ap.parse_args()
    if a.units:
        m, t, team = a.units[0], float(a.units[1]), a.units[2]
        seg = ensure_segment(m, t)
        z = np.load(seg, allow_pickle=True)
        i = int(np.argmin(abs(z["time_s"] - t)))
        now = np.concatenate([z["home"][i], z["away"][i]]).astype(float)
        slots = list(range(11)) if team == "home" else list(range(11, 22))
        gx = L if now[slots[0], 0] > L / 2 else 0.0
        names = [str(n) for n in z["name_home"]] + [str(n) for n in z["name_away"]]
        for bd, ss in units(now, slots, gx).items():
            print(f"{bd:9s}: " + ", ".join(names[s] for s in ss))
        return

    b = json.load(open(a.board, encoding="utf-8"))
    assert b.get("board") in BOARDS and b.get("moves"), "board file needs 'board' (forward/midfield/defence) and 'moves'"
    os.makedirs(OUT, exist_ok=True)
    bid = b.get("id") or time.strftime("%Y%m%d_%H%M%S")
    seg = ensure_segment(b["match"], float(b["t"]))
    delta, names, att, team = board_deltas(b, seg)
    assert delta, "no player was moved"
    board = {"delta": {str(k): v for k, v in delta.items()}, "hold": b.get("mode", "once") == "hold"}
    _B.update(board)
    what = f"{b['board']} board ({'hold' if board['hold'] else 'move once'}): " + ", ".join(names)
    print(f"{what}\nsegment {os.path.basename(seg)}, start {b['t']} s, {team} on the board, {att} has the ball", flush=True)

    jobs = [(seg, float(b["t"]), s, 3.0, seed, a.seconds, None) for s in ("baseline", "board") for seed in range(a.seeds)]
    t0, rows = time.time(), []
    with Pool(a.procs, initializer=_init, initargs=(a.dataset, board)) as pool:
        for k, r in enumerate(pool.imap_unordered(WI._job, jobs)):
            rows.append(r)
            print(f"  {k + 1}/{len(jobs)} runs ({(time.time() - t0) / 60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"{bid}_runs.csv"), index=False)
    res = card(df, att, team)
    res.update({"id": bid, "what": what, "board": b})
    print(pd.DataFrame(res["rows"]).round(3).to_string(index=False), flush=True)

    # ghosts: one run of each with seed 0, in the viewer's frame
    import team_agent
    models = team_agent.load_models(a.dataset)
    z = np.load(seg, allow_pickle=True)
    real = np.concatenate([z["home"], z["away"]], axis=1).transpose(1, 0, 2)
    i0 = _moment(seg, float(b["t"]), "baseline", 3.0)[5]
    t_seg = z["time_s"]
    segkey = os.path.basename(seg).replace("_segment_", "_").replace(".npz", "")
    layers = []
    for scen, label in (("baseline", "No change"), ("board", "Your board")):
        _, pl, bl, mirrored = WI.play(seg, float(b["t"]), scen, 3.0, 0, a.seconds, models)
        if mirrored:
            pl, bl = WI.flip(pl), WI.flip(bl)
        n = min(pl.shape[1], len(t_seg) - i0)
        sl = slice(i0, i0 + n)
        path = os.path.join(OUT, f"grf_board_{segkey}_{bid}_{scen}.npz")
        np.savez_compressed(path, grf=pl[:, :n], ball_grf=bl[:n], real=real[:, sl], ball_real=z["ball"][sl], time_s=t_seg[sl],
                            live=z["live"][sl], scenario=scen, what_changed=what if scen == "board" else "no change")
        layers += ["--layer", f"{label}={os.path.relpath(path, ROOT)}"]
    vf = f"board_{bid}.json"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "export_viewer.py"), "--match", b["match"], "--start-s", str(float(b["t"]) - 3),
                    "--minutes", str((a.seconds + 6) / 60), *layers, "--out", vf], cwd=ROOT, check=True)
    p = os.path.join(ROOT, "viewer", "data", vf)
    d = json.load(open(p, encoding="utf-8"))
    d["card"] = res
    json.dump(d, open(p, "w", encoding="utf-8"), separators=(",", ":"))
    json.dump(res, open(os.path.join(OUT, f"{bid}_result.json"), "w", encoding="utf-8"), indent=1)
    print(f"\nOpen: http://localhost:8765/?file={vf}")


if __name__ == "__main__":
    main()
