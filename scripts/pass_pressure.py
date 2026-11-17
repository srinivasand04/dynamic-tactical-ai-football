"""Weak spot 5: does PRESSURE ON THE RECEIVER make the pass models better? Plus: is pass completion calibrated?

The pass models already know the nearest opponent to each teammate (now and when the ball arrives) and the time margin of
the lane. New numbers here, per teammate, from the same positions and velocities (graph of bo_graph.py):
    n_opp5_f   opponents within 5 m of where the teammate will be when the ball arrives          (crowding)
    closing    speed of his nearest opponent (at arrival) towards him, m/s; > 0 = closing down   (pressure coming)
    n_lane     opponents within 3 m of the pass line                                               (bodies in the lane)

Test (same rule as the tuning ladder, leave one match out, 7 real matches):
    choice      linear (current, 13 numbers)  vs  linear + pressure     top-1 / top-3 / log loss
    completion  logistic (current)            vs  logistic + pressure   AUC / Brier, and calibration (ECE, slope)
    A new model wins only if it is better on average, the 95% interval excludes zero and it is better in at least 6 of 7.

Reads data/real/processed/<m>_passes.npz and <m>_passfeat.npz; writes only outputs/tuning/pass_pressure*.csv.
outputs/pass_model.json and the _passfeat files are not changed.
Usage:  .venv-analysis\\Scripts\\python.exe scripts\\pass_pressure.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bo_graph  # noqa: E402
import fit_pass_model as F  # noqa: E402
import pass_features as P  # noqa: E402
import settings  # noqa: E402
from value_calibration import reliability, slope_intercept  # noqa: E402

OUT = os.path.join(settings.ROOT if hasattr(settings, "ROOT") else os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "tuning")
PRESS = ["n_opp5_f", "closing", "n_lane"]


def pressure(xy, vel, present, team, passer):
    c = P.candidates(xy, vel, present, team, passer)
    g = bo_graph.build(xy, present, team, None, bo=passer)
    if c is None or g is None:
        return None
    opps = np.array(g["opps"])
    a, posf = c["a"], c["posf"]
    tau = np.clip(np.linalg.norm(xy[c["slots"]] - a, axis=1) / P.VB, 0, P.TAU_MAX)
    oposf = np.clip(xy[opps][None] + vel[opps][None] * (tau[:, None, None] * P.DAMP), 0, [P.L, P.W])
    d = np.linalg.norm(posf[:, None] - oposf, axis=2)                       # (mates, opps)
    n5 = (d < 5).sum(1)
    near = d.argmin(1)
    to_me = posf - oposf[np.arange(len(posf)), near]
    closing = (vel[opps][near] * to_me).sum(1) / (np.linalg.norm(to_me, axis=1) + 1e-6)
    seg = posf - a                                                          # distance of every opponent (now) to the pass line
    o = xy[opps]
    u = np.clip(((o[None] - a) * seg[:, None]).sum(2) / ((seg ** 2).sum(1)[:, None] + 1e-9), 0, 1)
    dl = np.linalg.norm(a + u[..., None] * seg[:, None] - o[None], axis=2)
    n_lane = (dl < 3).sum(1)
    X = np.stack([n5 / 3, np.clip(closing, -8, 8) / 4, n_lane / 3], axis=1)
    return c["slots"], X


def extras(m, feat):
    z = np.load(settings.base(m) + "_passes.npz", allow_pickle=True)
    where = {(int(f), int(p)): i for i, (f, p) in enumerate(zip(z["frame"], z["passer"]))}
    E = np.zeros(feat["mask"].shape + (len(PRESS),))
    miss = 0
    for r, (f, p) in enumerate(zip(feat["frame"], feat["passer"])):
        i = where.get((int(f), int(p)))
        out = None if i is None else pressure(z["xy"][i], z["vel"][i], z["present"][i], z["team"], int(p))
        if out is None:
            miss += 1
            continue
        n = min(len(out[0]), P.N_MATES)
        E[r, :n] = out[1][:n]
    return E, miss


def rule(d, higher_better=True):
    d = np.asarray(d) if higher_better else -np.asarray(d)
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    wins = int((d > 0).sum())
    return d.mean(), ci, wins, bool(d.mean() > 0 and d.mean() - ci > 0 and wins >= len(d) - 1)


def main():
    matches = [m for m in settings.available(settings.REAL) if os.path.exists(settings.base(m) + "_passfeat.npz")]
    data = F.load(matches)
    for m in matches:
        E, miss = extras(m, data[m])
        data[m]["I"] = np.concatenate([data[m]["I"], E], axis=-1)
        print(f"  {m}: {len(E)} passes, pressure added ({miss} without a graph, left at 0)", flush=True)
    for k, c in enumerate(PRESS):
        F.COL[c] = len(P.COLUMNS) + k
    cur, new = F.FULL, F.FULL + PRESS
    ccur, cnew = F.COMPLETION_COLS, F.COMPLETION_COLS + PRESS

    rows, oof = [], {"current": ([], []), "+ pressure": ([], [])}
    for m in matches:
        tr = [x for x in matches if x != m]
        I, M, T = (F.cat(data, tr, k, True) for k in ("I", "mask", "target"))
        Ite, Mte, Tte = (data[m][k][data[m]["complete"]] for k in ("I", "mask", "target"))
        Xtr, ytr = F.chosen(F.cat(data, tr, "I"), F.cat(data, tr, "target")), F.cat(data, tr, "complete").astype(float)
        te, y = F.chosen(data[m]["I"], data[m]["target"]), data[m]["complete"].astype(int)
        for name, cols, cc in (("current", cur, ccur), ("+ pressure", new, cnew)):
            w = F.fit_choice(I, M, T, cols)
            ch = F.choice_metrics(F.choice_scores(w, Ite, Mte, cols), Tte)
            lw, lb = F.fit_logistic(Xtr[:, F.idx(cc)], ytr)
            p = F.sigmoid(te[:, F.idx(cc)] @ lw + lb)
            oof[name][0].append(y)
            oof[name][1].append(p)
            rows.append({"fold": m, "model": name, **ch, "auc": F.auc(y, p), "brier": float(((p - y) ** 2).mean())})
        print(f"  held out {m}: top-1 {rows[-2]['top1_%']:.1f}% -> {rows[-1]['top1_%']:.1f}%   AUC {rows[-2]['auc']:.3f} -> {rows[-1]['auc']:.3f}", flush=True)
    runs = pd.DataFrame(rows)

    a, b = runs[runs.model == "current"].set_index("fold"), runs[runs.model == "+ pressure"].set_index("fold")
    lad = []
    for key, hb in (("top1_%", True), ("top3_%", True), ("log_loss", False), ("auc", True), ("brier", False)):
        mean, ci, wins, ok = rule(b[key] - a[key], hb)
        lad.append({"metric": key, "current": a[key].mean(), "+ pressure": b[key].mean(), "change": b[key].mean() - a[key].mean(),
                    "ci95": ci, "better_in_folds": f"{wins}/{len(a)}", "wins_rule": ok})
    lad = pd.DataFrame(lad)
    cal = []
    for name, (ys, ps) in oof.items():
        yy, pp = np.concatenate(ys).astype(float), np.concatenate(ps)
        t, ece = reliability(yy, pp, [0, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
        icpt, slope = slope_intercept(yy, pp)
        cal.append({"completion model": name, "passes": len(yy), "ece": ece, "slope": slope, "intercept": icpt,
                    "predicted_complete": pp.sum(), "real_complete": int(yy.sum())})
    cal = pd.DataFrame(cal)
    os.makedirs(OUT, exist_ok=True)
    runs.round(4).to_csv(os.path.join(OUT, "pass_pressure_runs.csv"), index=False)
    lad.round(4).to_csv(os.path.join(OUT, "pass_pressure.csv"), index=False)
    cal.round(4).to_csv(os.path.join(OUT, "pass_calibration.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nPressure on the receiver (leave one match out):")
    print(lad.round(4).to_string(index=False))
    print("\nPass completion calibration (held-out, pooled):")
    print(cal.round(4).to_string(index=False))
    print("\nwrote outputs/tuning/pass_pressure.csv, pass_pressure_runs.csv, pass_calibration.csv")


if __name__ == "__main__":
    main()
