"""Phase 2, step 1: how good do video positions need to be? (noise test, no video needed)

A video pipeline (detector or VLM + tracker + pitch mapping) gives worse positions than real tracking data. Here we take the
real tracking of each held-out Bundesliga match, damage it the way a video pipeline would, rebuild the Delaunay graphs from the
damaged positions, and score the movement model (trained on clean data) against the TRUE next movement.

Damage types (each alone, as a sweep, then combined profiles):
  view      broadcast camera: only players within W/2 metres (along the pitch) of the ball are seen
  drop      k visible players missed per Key State (missed detections)
  jitter    position error, sigma metres (Gaussian, every node); velocity error sigma * VEL_PER_M m/s
  ball      the ball is not found in a share p of Key States
  swap      identity switches: per Key State, with chance p two visible teammates swap identities from then on (a tracker ID switch)

Scores per held-out match (then the mean over the 7 matches):
  player_m      model error (m) on players that are visible in the damaged input, against their true movement
  ball_m        model error (m) on the ball
  keepvel_m     the "keep your velocity" baseline on the same damaged input (a pipeline without any model)
  edge_recall   share of the true graph's player links that the damaged graph still has (graph fidelity)
  seen          share of the 22 players visible

The combined profiles are ASSUMPTIONS about what each pipeline delivers; Phase 2 measures the real values on video.

Input : data/real/processed/<match>_graphs_delaunay.npz, outputs/noise_test/models/ours_<match>.pt (train_gnn.py --save-model)
Output: outputs/noise_test/noise_runs.csv, noise_summary.csv, noise_chart.png

Usage :  .venv-analysis\\Scripts\\python.exe scripts\\noise_test.py
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
import settings
import train_gnn as TG
from build_graphs import make_edges, N, BALL, TEAM

OUT = "outputs/noise_test"
MATCHES = ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]
VEL_PER_M = 1.0
G = settings.S["graph"]

SWEEPS = {
    "view": [("view_m", w) for w in (100, 70, 60, 50, 40, 30)],
    "drop": [("drop", k) for k in (0, 1, 2, 4, 6, 8)],
    "jitter": [("jitter_m", s) for s in (0, 0.25, 0.5, 1, 2, 3)],
    "ball": [("ball_miss", p) for p in (0, 0.1, 0.25, 0.5)],
    "swap": [("swap", p) for p in (0, 0.02, 0.05, 0.1, 0.25)],
}
PROFILES = {   # assumptions, to be replaced by measured values in Phase 2
    "clean (true tracking)": {},
    "good specialised CV": dict(view_m=60, jitter_m=0.5, ball_miss=0.1, swap=0.02),
    "detector + homography": dict(view_m=50, jitter_m=1.0, ball_miss=0.2, swap=0.05, drop=1),
    "VLM only (rough)": dict(view_m=50, jitter_m=2.0, ball_miss=0.4, swap=0.1, drop=2),
}


def damage(g, cfg, rng):
    """Damaged copy of a graph file (dict) and, per Key State, the slot -> true player map."""
    xy, vel, role = g["xy"].copy(), g["vel"].copy(), g["role"].copy()
    K = len(xy)
    present = ~np.isnan(xy[..., 0])
    perm = np.tile(np.arange(N), (K, 1))
    if cfg.get("swap", 0) > 0:                                            # persistent ID switches between visible teammates
        cur = np.arange(N)
        for k in range(K):
            if rng.random() < cfg["swap"]:
                t = rng.integers(2)
                cand = [s for s in range(11 * t, 11 * t + 11) if present[k, cur[s]]]
                if len(cand) >= 2:
                    i, j = rng.choice(cand, 2, replace=False)
                    cur[[i, j]] = cur[[j, i]]
            perm[k] = cur
        xy = np.take_along_axis(xy, perm[..., None], 1)
        vel = np.take_along_axis(vel, perm[..., None], 1)
        present = ~np.isnan(xy[..., 0])
    seen = present.copy()
    bx = pd.Series(xy[:, BALL, 0]).ffill().bfill().to_numpy()
    if cfg.get("view_m", 200) < 105:
        seen[:, :22] &= np.abs(xy[:, :22, 0] - bx[:, None]) <= cfg["view_m"] / 2
    if cfg.get("drop", 0) > 0:
        for k in range(K):
            vis = np.flatnonzero(seen[k, :22])
            if len(vis):
                seen[k, rng.choice(vis, min(cfg["drop"], len(vis)), replace=False)] = False
    if cfg.get("ball_miss", 0) > 0:
        seen[:, BALL] &= rng.random(K) >= cfg["ball_miss"]
    if cfg.get("jitter_m", 0) > 0:
        s = cfg["jitter_m"]
        xy = xy + rng.normal(0, s, xy.shape)
        vel = vel + rng.normal(0, s * VEL_PER_M, vel.shape)
    xy[~seen] = np.nan
    vel[~seen] = 0
    adj_team, adj_opp, adj_ball, D, present2 = make_edges(xy, "delaunay", G["opp_radius_m"], G["ball_k"], G["team_k"])
    out = {k: g[k] for k in g.files}
    out.update(xy=xy, vel=vel, role=role, present=present2, dist_ball=np.where(present2, D[:, BALL, :], np.nan),
               adj_team=adj_team, adj_opp=adj_opp, adj_ball=adj_ball)
    return out, perm


def edge_recall(g, dmg, perm):
    """Share of true player-player links (team + opponent) that the damaged graph also has, comparing true players."""
    true = g["adj_team"] | g["adj_opp"]
    dm = dmg["adj_team"] | dmg["adj_opp"]
    K = len(true)
    back = np.empty_like(perm)
    back[np.arange(K)[:, None], perm] = np.arange(N)[None]               # true player -> slot
    dm_true = dm[np.arange(K)[:, None, None], back[:, :, None], back[:, None, :]]
    t = true.sum()
    return float((true & dm_true).sum() / t) if t else np.nan


def score(model, match, g, cfg, seed):
    rng = np.random.default_rng(seed)
    dmg, perm = damage(g, cfg, rng)
    clean = TG.load(match)
    with tempfile.TemporaryDirectory() as tmp:
        np.savez(os.path.join(tmp, "x_graphs_delaunay.npz"), **dmg)
        TG.PATH["_dmg"] = os.path.join(tmp, "x")
        d = TG.load("_dmg")
    K = len(perm)
    rows = np.arange(K)[:, None]
    d["disp"] = clean["disp"][rows, perm]                                 # true movement of the player each slot now follows
    ok = clean["disp_ok"][rows, perm]
    ok[:, :22] &= d["present"][:, :22]                                    # score players the pipeline actually sees
    if cfg.get("swap", 0) > 0:
        ok[:-1] &= (perm[1:] == perm[:-1])                                # skip the instant of the switch itself
    d["disp_ok"] = ok
    d["vel_ms"] = d["vel_ms"]                                             # keep-velocity baseline uses the damaged velocity
    (p, b), _, (pv, bv) = TG.evaluate(model, d)
    seen = float(d["present"][:, :22].mean() / max(clean["present"][:, :22].mean(), 1e-9))
    return dict(player_m=p, ball_m=b, keepvel_m=pv, keepvel_ball_m=bv, edge_recall=edge_recall(g, dmg, perm), seen=seen)


def chart(sm, prof, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"view": "Camera view width (m)", "drop": "Players missed per moment", "jitter": "Position error sigma (m)",
             "ball": "Share of moments ball not found", "swap": "Chance of an ID switch per moment"}
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    clean = prof.loc[prof["case"] == "clean (true tracking)", "player_m"].iloc[0]
    for a, (sw, title) in zip(ax.flat, names.items()):
        s = sm[sm["sweep"] == sw]
        a.errorbar(s["level"], s["player_m"], yerr=s["player_m_ci95"], marker="o", color="#1d7a4e", label="our model (players)", capsize=3)
        a.plot(s["level"], s["keepvel_m"], marker="s", ls="--", color="#999", label="keep velocity, no model")
        a.axhline(clean, color="#b86e00", lw=0.8, ls=":", label=f"clean tracking ({clean:.2f} m)")
        a.set_title(title)
        a.set_ylabel("movement error (m), lower is better")
        if sw == "view":
            a.invert_xaxis()
        a2 = a.twinx()
        a2.plot(s["level"], s["edge_recall"], color="#5b6ad0", alpha=0.6, marker=".", label="graph links kept")
        a2.set_ylim(0, 1.05)
        a2.set_ylabel("share of true graph links kept", color="#5b6ad0")
        a.legend(fontsize=8, loc="upper left")
    a = ax.flat[5]
    a.clear()
    y = np.arange(len(prof))
    a.barh(y, prof["player_m"], xerr=prof["player_m_ci95"], color=["#b86e00", "#1d7a4e", "#7da88f", "#c8402f"][:len(prof)], capsize=3)
    a.barh(y + 0.3, prof["keepvel_m"], height=0.25, color="#bbb", label="keep velocity, no model")
    a.set_yticks(y, prof["case"])
    a.invert_yaxis()
    for i, (v, e) in enumerate(zip(prof["player_m"], prof["edge_recall"])):
        a.text(v + 0.03, i - 0.1, f"{v:.2f} m, links kept {e:.0%}", fontsize=8)
    a.set_title("Assumed pipeline profiles (to be measured in Phase 2)")
    a.set_xlabel("movement error (m)")
    a.legend(fontsize=8)
    fig.suptitle("Noise test: movement model on damaged tracking, 7 held-out Bundesliga matches (mean and 95% interval over matches)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="+", default=MATCHES)
    ap.add_argument("--report", action="store_true", help="only rebuild the summary and chart")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    runs_p = f"{OUT}/noise_runs.csv"
    if not a.report:
        torch.set_num_threads(6)
        cases = [(sw, lvl, {k: v}) for sw, lv in SWEEPS.items() for k, v in lv for lvl in [v]]
        cases += [("profile", name, cfg) for name, cfg in PROFILES.items()]
        rows = []
        for m in a.matches:
            mp = f"{OUT}/models/ours_{m}.pt"
            if not os.path.exists(mp):
                print(f"skip {m}: no model {mp}")
                continue
            data0 = TG.load(m)
            model = TG.Net(data0["feat"].shape[2])
            model.load_state_dict(torch.load(mp))
            g = np.load(settings.base(m) + "_graphs_delaunay.npz")
            for i, (sw, lvl, cfg) in enumerate(cases):
                r = score(model, m, g, cfg, seed=1000 + i)
                rows.append({"match": m, "sweep": sw, "level": lvl, **r})
                print(f"{m} {sw:8s} {str(lvl):24s} players {r['player_m']:.3f} m  ball {r['ball_m']:.2f} m  keep-vel {r['keepvel_m']:.3f}  links kept {r['edge_recall']:.2f}  seen {r['seen']:.2f}", flush=True)
            pd.DataFrame(rows).to_csv(runs_p, index=False)
    df = pd.read_csv(runs_p)
    keys = ["player_m", "ball_m", "keepvel_m", "edge_recall", "seen"]
    agg = df.groupby(["sweep", "level"], sort=False).agg(**{k: (k, "mean") for k in keys}, player_m_ci95=("player_m", ci), matches=("match", "nunique")).reset_index()
    agg.round(4).to_csv(f"{OUT}/noise_summary.csv", index=False)
    sm = agg[agg["sweep"] != "profile"].copy()
    sm["level"] = sm["level"].astype(float)
    prof = agg[agg["sweep"] == "profile"].rename(columns={"level": "case"})
    pd.set_option("display.width", 200)
    print(agg.round(3).to_string(index=False))
    try:
        chart(sm, prof, f"{OUT}/noise_chart.png")
        print(f"wrote {OUT}/noise_chart.png")
    except ImportError:
        print("chart needs matplotlib")


if __name__ == "__main__":
    main()
