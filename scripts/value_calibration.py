"""Weak spot 1: are the xG and threat numbers *calibrated*? (does "0.20" really mean 1 in 5?)

value_models.py already checks ranking (AUC) and totals. This adds the calibration check, on matches the model never saw:
    reliability table   predictions cut into bins; per bin: mean prediction vs real share (with a 95% interval)
    ECE                 expected calibration error = weighted mean |prediction - real share| over the bins
    slope / intercept   logistic fit  real ~ a + b * logit(prediction);  perfect = slope 1, intercept 0
                        slope < 1: predictions too extreme;  slope > 1: too timid
    totals              predicted vs real goals (or shots), with a Poisson 95% band on the real count

Checked (the models the what-if and GRF actually use):
    xG      the old distance proxy, distance + angle, all features (ours)    World Cup held-out groups + Bundesliga (never seen)
    threat  distance to goal (rank only, not a probability), state logistic (ours), target "this team shoots within 10 s"

Reads the cached tables of value_models.py, writes only new files (outputs/value_models/calibration_*), changes no model.
Usage:  .venv-analysis\\Scripts\\python.exe scripts\\value_calibration.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import value_models as V  # noqa: E402

OUT = V.OUT
BINS = [0, 0.03, 0.06, 0.10, 0.15, 0.25, 0.40, 1.0]


def slope_intercept(y, p):
    z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    X = np.c_[np.ones(len(z)), z]
    w = np.array([0.0, 1.0])
    for _ in range(50):
        q = 1 / (1 + np.exp(-X @ w))
        H = (X * (q * (1 - q))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (q - y))
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return float(w[0]), float(w[1])


def reliability(y, p, bins=BINS):
    b = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
    rows = []
    for k in range(len(bins) - 1):
        m = b == k
        n = int(m.sum())
        if not n:
            continue
        r = float(y[m].mean())
        half = 1.96 * np.sqrt(max(r * (1 - r), 1 / n) / n)
        rows.append({"bin": f"{bins[k]:.2f}-{bins[k + 1]:.2f}", "n": n, "predicted": float(p[m].mean()), "real": r,
                     "real_lo": max(0.0, r - half), "real_hi": min(1.0, r + half)})
    t = pd.DataFrame(rows)
    ece = float((t["n"] * (t["predicted"] - t["real"]).abs()).sum() / t["n"].sum())
    return t, ece


def summary(name, data, y, p):
    a, b = slope_intercept(y, p)
    t, ece = reliability(y, p)
    real = int(y.sum())
    lo, hi = real - 1.96 * np.sqrt(real), real + 1.96 * np.sqrt(real)
    s = {"model": name, "data": data, "n": len(y), "auc": V.auc(y, p), "brier": float(np.mean((p - y) ** 2)), "ece": ece,
         "slope": b, "intercept": a, "predicted_total": float(p.sum()), "real_total": real,
         "total_ok": bool(lo <= p.sum() <= hi)}
    t.insert(0, "data", data)
    t.insert(0, "model", name)
    return s, t


def main():
    shots = pd.read_csv(os.path.join(OUT, "pff_shots.csv"))
    moves = pd.read_csv(os.path.join(OUT, "pff_moves.csv.gz"))
    dfl = pd.read_csv(os.path.join(OUT, "dfl_shots.csv"))
    open_ = V.add_derived(shots[~shots["penalty"]].reset_index(drop=True))
    dfl_open = V.add_derived(dfl[~dfl["penalty"]].reset_index(drop=True))
    g = open_["match"].map(V.groups(open_["match"])).to_numpy()
    y, dy = open_["goal"].to_numpy(float), dfl_open["goal"].to_numpy(float)

    rows, tables, curves = [], [], {}
    xg = {"distance proxy (old)": None, "distance + angle": V.SETS["distance + angle"], "all features (ours)": V.FEATS}
    for name, cols in xg.items():
        if cols is None:
            p, dp = V.proxy(open_["dist"].to_numpy()), V.proxy(dfl_open["dist"].to_numpy())
        else:
            p = np.zeros(len(open_))
            for k in range(V.N_GROUPS):
                p[g == k] = V.Logit(cols).fit(open_[g != k]).predict(open_[g == k])
            dp = V.Logit(cols).fit(open_).predict(dfl_open)
        for data, yy, pp in (("World Cup (held-out matches)", y, p), ("Bundesliga (never seen)", dy, dp)):
            s, t = summary(name, data, yy, pp)
            rows.append(s)
            tables.append(t)
            curves[(name, data)] = t
        print(f"  xG {name}: done", flush=True)

    ev = V.threat_labels(moves)
    gev = ev["match"].map(V.groups(moves["match"])).to_numpy()
    ty = ev["shot10"].to_numpy(float)
    tp = np.zeros(len(ev))
    for k in range(V.N_GROUPS):
        tp[gev == k] = V.Logit(V.SCOLS, y="shot10").fit(ev[gev != k]).predict(ev[gev == k])
    s, t = summary("threat: state logistic (ours)", "World Cup (held-out matches)", ty, tp)
    rows.append(s)
    tables.append(t)
    curves[("threat: state logistic (ours)", "World Cup (held-out matches)")] = t
    print("  threat: done", flush=True)

    summ, rel = pd.DataFrame(rows), pd.concat(tables, ignore_index=True)
    summ.to_csv(os.path.join(OUT, "calibration_summary.csv"), index=False)
    rel.to_csv(os.path.join(OUT, "calibration_bins.csv"), index=False)
    chart(curves, summ, os.path.join(OUT, "calibration_chart.png"))

    pd.set_option("display.width", 220)
    print("\nCalibration (held-out):")
    print(summ.round(3).to_string(index=False))
    print("\nwrote outputs/value_models/calibration_summary.csv, calibration_bins.csv, calibration_chart.png")


def chart(curves, summ, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.2))
    col = {"distance proxy (old)": "#b5b3ab", "distance + angle": "#8fb1e0", "all features (ours)": "#2a7ad5",
           "threat: state logistic (ours)": "#e0483b"}
    panels = [("World Cup (held-out matches)", "xG, World Cup (matches never seen in training)", False),
              ("Bundesliga (never seen)", "xG, Bundesliga (other league, other data)", False),
              ("World Cup (held-out matches)", "Threat: this team shoots within 10 s", True)]
    for a, (data, title, thr) in zip(ax, panels):
        top = 0.6 if not thr else 0.5
        a.plot([0, top], [0, top], ls="--", c="#777", lw=1, label="perfect")
        for (name, d), t in curves.items():
            if d != data or (name.startswith("threat") != thr):
                continue
            r = summ[(summ["model"] == name) & (summ["data"] == d)].iloc[0]
            a.errorbar(t["predicted"], t["real"], yerr=[t["real"] - t["real_lo"], t["real_hi"] - t["real"]], marker="o",
                       c=col[name], capsize=3, lw=2 if "ours" in name else 1.2,
                       label=f"{name}: ECE {r['ece']:.3f}, slope {r['slope']:.2f}")
        a.set_xlim(0, top)
        a.set_ylim(0, top)
        a.set_xlabel("predicted chance")
        a.set_ylabel("real share")
        a.set_title(title)
        a.legend(fontsize=9, loc="upper left")
        a.grid(alpha=0.3)
    fig.suptitle("Do the numbers mean what they say?  Points on the dashed line = calibrated (bars: 95% interval)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
