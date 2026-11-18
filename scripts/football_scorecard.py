"""Football Scorecard: rate the movement models on football sense, not only metres of error.

Every movement model in the tuning ladder saved its held-out predictions (outputs/tuning/preds/, 7 Bundesliga matches,
leave-one-match-out). Each prediction is where all 22 players and the ball will be at the next Key State (~1.1 s ahead).
Here the predicted picture is compared with the real one on six football questions:

    shape     is the team shape right?  width, depth and back-line height of both teams (error in m)
    space     does each team own the same parts of the pitch?  nearest-player map on a 1 m grid (% of pitch wrong)
    passes    are the same teammates open?  team on the ball, teammate open = no opponent within 5 m (% wrong)
    danger    is the moment as dangerous?  threat model (value_models, chance of a shot in 10 s), |pred - real| in %-points
    realistic can a human run it?  % of predicted player moves faster than 9 m/s (real data: almost none)
    runner    did it pick the runner?  the attacking outfield player with the longest real forward run is in the model's top 3

Scores 0-100 = how much of the 'everyone stands still' mistake the model removes (skill score; 0 = no better than
standing still, 100 = perfect). 'realistic' is the % of moves within human speed. Baselines included: stand still,
keep running (last movement repeated). Ensemble = mean of the 3 seeds. Same yardstick for every model, so models can be
compared and quantified: per-match scores give a 95% interval and 'better than current in x/7 matches'.

No training: saved predictions are re-read. New files only.
Output: outputs/scorecard/scorecard.csv (one row per model), scorecard_matches.csv (per match), scorecard_chart.png
Usage : .venv-analysis\\Scripts\\python.exe scripts\\football_scorecard.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings  # noqa: E402
import value_models as V  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDS = os.path.join(ROOT, "outputs", "tuning", "preds")
OUT = os.path.join(ROOT, "outputs", "scorecard")
SUFFIX = settings.graph_suffix()
START = 4                      # train_gnn.EVAL_START: predictions are for Key States 4 .. K-2
STEP = 5                       # score every 5th Key State (space/danger are the slow parts)
OPEN_M, MAX_SPEED, HOLD_M = 5.0, 9.0, 3.0
GX, GY = np.meshgrid(np.arange(0.5, 105, 1.0), np.arange(0.5, 68, 1.0))
GRID = np.stack([GX.ravel(), GY.ravel()], 1)
THREAT = json.load(open(os.path.join(ROOT, "outputs", "value_models", "threat_model.json")))
CURRENT = "ours__window_3____current"
ENSEMBLE = [CURRENT, "ours_seed_1", "ours_seed_2"]
SCORES = ["shape", "space", "passes", "danger", "runner"]


def threat(pos, team):
    """Chance the team on the ball shoots within 10 s (value_models threat logistic), team attacking away from its keeper."""
    a0, d0 = (0, 11) if team == 0 else (11, 0)
    flip = pos[a0, 0] > 52.5 if np.isfinite(pos[a0, 0]) else False
    f = (lambda p: np.stack([105 - p[..., 0], 68 - p[..., 1]], -1)) if flip else (lambda p: p)
    x = V.state_features(f(pos[22]), f(pos[a0:a0 + 11]), f(pos[d0:d0 + 11]))
    z = THREAT["weights"][0] + sum(w * (x[n[2:]] - m) / s for w, n, m, s in
                                   zip(THREAT["weights"][1:], THREAT["features"], THREAT["mean"], THREAT["std"]))
    return 1 / (1 + np.exp(-z))


def shape(pos, sign):
    """width, depth, back-line height (distance of the 4 deepest outfield players from their own goal) for one team."""
    out = pos[1:11] if sign[0] else pos[12:22]
    out = out[np.isfinite(out[:, 0])]
    if len(out) < 6:
        return np.full(3, np.nan)
    x = out[:, 0] if sign[1] > 0 else 105 - out[:, 0]          # distance from own goal line
    return np.array([np.ptp(out[:, 1]), np.ptp(x), np.sort(x)[:4].mean()])


def owner_map(pos):
    p = pos[:22]
    ok = np.isfinite(p[:, 0])
    d = np.hypot(*(GRID[:, None] - p[None, ok]).transpose(2, 0, 1))
    return np.where(ok)[0][d.argmin(1)] < 11


def open_mates(pos, team):
    a0, d0 = (0, 11) if team == 0 else (11, 0)
    att, dfs = pos[a0 + 1:a0 + 11], pos[d0:d0 + 11]
    dfs = dfs[np.isfinite(dfs[:, 0])]
    d = np.hypot(*(att[:, None] - dfs[None]).transpose(2, 0, 1)).min(1)
    return d > OPEN_M, np.isfinite(att[:, 0])


def frame_metrics(now, pred, real, ok, team, sign, dt):
    """All six questions for one Key State. now/pred/real: (23, 2) in metres."""
    pred = np.where(ok[:, None], pred, real)           # players missing in the target are not judged
    r = {}
    r["error"] = float(np.linalg.norm(pred[:22] - real[:22], axis=1)[ok[:22]].mean())
    s_p = np.array([shape(pred, (True, sign)), shape(pred, (False, -sign))])
    s_r = np.array([shape(real, (True, sign)), shape(real, (False, -sign))])
    r["shape"] = float(np.nanmean(np.abs(s_p - s_r)))
    r["space"] = float((owner_map(pred) != owner_map(real)).mean() * 100)
    if team is not None:
        op, okp = open_mates(pred, team)
        orl, okr = open_mates(real, team)
        m = okp & okr
        r["passes"] = float((op[m] != orl[m]).mean() * 100) if m.any() else np.nan
        r["danger"] = float(abs(threat(pred, team) - threat(real, team)) * 100)
        a0 = 1 if team == 0 else 12
        fwd = (sign if team == 0 else -sign)
        run_r = (real[a0:a0 + 10, 0] - now[a0:a0 + 10, 0]) * fwd
        run_p = (pred[a0:a0 + 10, 0] - now[a0:a0 + 10, 0]) * fwd
        good = ok[a0:a0 + 10] & np.isfinite(run_r)
        if good.sum() >= 5 and np.nanmax(np.where(good, run_r, -99)) > 1.0:
            best = np.nanargmax(np.where(good, run_r, -99))
            r["runner"] = 0.0 if best in np.argsort(-np.where(good, run_p, -99))[:3] else 100.0   # % missed
        else:
            r["runner"] = np.nan
    else:
        r["passes"] = r["danger"] = r["runner"] = np.nan
    spd = np.linalg.norm(pred[:22] - now[:22], axis=1) / max(dt, 0.2)
    r["realistic"] = float((spd[ok[:22]] <= MAX_SPEED).mean() * 100)
    return r


def match_metrics(m, models):
    """models: {name: pred moves (n, 23, 2)}. Returns per-model mean of each metric for this match."""
    g = np.load(settings.base(m) + f"_graphs{SUFFIX}.npz")
    xy, t = g["xy"], g["time_s"]
    z = np.load(os.path.join(PREDS, f"{CURRENT}_{m}.npz"))
    tgt, ok = z["target"], z["ok"]
    n = len(tgt)
    sign = 1.0 if np.nanmedian(xy[:, 0, 0]) < 52.5 else -1.0         # home attacks +x if its keeper is on the left
    team, out = None, {k: [] for k in models}
    for i in range(n):
        k = START + i
        now = xy[k]
        d = np.hypot(*(now[:22] - now[22]).T)
        if np.isfinite(d).any() and np.nanmin(d) < HOLD_M:
            team = 0 if np.nanargmin(d) < 11 else 1
        if i % STEP or not ok[i].any() or not np.isfinite(now[22, 0]):
            continue
        real = now + tgt[i]
        dt = t[k + 1] - t[k]
        for name, mv in models.items():
            out[name].append(frame_metrics(now, now + mv[i], real, ok[i], team, sign, dt))
    return {name: pd.DataFrame(v).mean() for name, v in out.items()}


def load_models(m):
    slugs = sorted({os.path.basename(p)[:-len(f"_{m}.npz")] for p in glob.glob(os.path.join(PREDS, f"*_{m}.npz"))})
    models = {s: np.load(os.path.join(PREDS, f"{s}_{m}.npz"))["pred"] for s in slugs}
    if all(s in models for s in ENSEMBLE):
        models["3-seed ensemble"] = np.mean([models[s] for s in ENSEMBLE], axis=0)
    g = np.load(settings.base(m) + f"_graphs{SUFFIX}.npz")
    n = len(models[CURRENT])
    models["baseline: stand still"] = np.zeros_like(models[CURRENT])
    prev = g["dxy_prev"][START:START + n]
    gap = g["gap_s"]
    ratio = (gap[START + 1:START + n + 1] / np.maximum(gap[START:START + n], 0.2))[:, None, None]
    models["baseline: keep running"] = np.nan_to_num(prev * ratio)
    return models


def nice(s):
    return s.replace("ours__window_3____current", "current (ours)").replace("___", " + ").replace("__", " ").replace("_", " ").strip()


def chart(sm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = SCORES + ["realistic", "overall"]
    d = sm.set_index("model")[cols]
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(d) + 1.6))
    im = ax.imshow(d.to_numpy(float), cmap="RdYlGn", vmin=-20, vmax=100, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(d)), d.index)
    for i in range(len(d)):
        for j in range(len(cols)):
            ax.text(j, i, f"{d.iat[i, j]:.0f}", ha="center", va="center", fontsize=9)
    ax.set_title("Football Scorecard (0 = no better than standing still, 100 = perfect; realistic = % moves at human speed)")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for m in settings.available(settings.REAL):
        if not os.path.exists(os.path.join(PREDS, f"{CURRENT}_{m}.npz")):
            continue
        res = match_metrics(m, load_models(m))
        for name, s in res.items():
            rows.append({"match": m, "model": nice(name), **s.to_dict()})
        print(f"  {m} scored ({len(res)} models)", flush=True)
    raw = pd.DataFrame(rows)
    still = raw[raw.model == "baseline: stand still"].set_index("match")
    for c in SCORES:                                       # skill vs standing still, per match
        raw[f"{c}_score"] = 100 * (1 - raw[c] / raw["match"].map(still[c]))
    raw["overall"] = raw[[f"{c}_score" for c in SCORES]].mean(1)
    raw.round(4).to_csv(os.path.join(OUT, "scorecard_matches.csv"), index=False)

    cur = raw[raw.model == "current (ours)"].set_index("match")["overall"]
    out = []
    for name, g in raw.groupby("model", sort=False):
        r = {"model": name}
        for c in SCORES:
            r[c] = g[f"{c}_score"].mean()
        r["realistic"] = g["realistic"].mean()
        r["overall"] = g["overall"].mean()
        r["overall_ci95"] = 1.96 * g["overall"].std(ddof=1) / np.sqrt(len(g))
        d = g.set_index("match")["overall"] - cur
        r["better_than_current_in"] = f"{int((d > 0).sum())}/{len(d)}"
        r["error_m"] = g["error"].mean()
        r["danger_gap_pp"] = g["danger"].mean()
        r["space_wrong_pct"] = g["space"].mean()
        out.append(r)
    sm = pd.DataFrame(out).sort_values("overall", ascending=False)
    sm.round(2).to_csv(os.path.join(OUT, "scorecard.csv"), index=False)
    pd.set_option("display.width", 250)
    print(sm.round(1).to_string(index=False))
    try:
        chart(sm, os.path.join(OUT, "scorecard_chart.png"))
        print("wrote outputs/scorecard/scorecard.csv, scorecard_matches.csv, scorecard_chart.png")
    except ImportError:
        print("chart needs matplotlib (.venv-analysis)")


if __name__ == "__main__":
    main()
