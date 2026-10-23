"""Group Key States into sequences using geometry only (no possession, no dead-ball flag, no events).

Version 2 (tuned):
  * time-based window: Key States are irregular in time, so each Key State's features are held until
    the next Key State and compared over the previous W seconds vs the next W seconds
  * shared scale: all matches are standardised with the SAME pooled mean / std, and share ONE threshold,
    so the number of sequences can be compared between the real match and the simulator
  * a boundary is a peak of the difference that exceeds (pooled mean + c * pooled std) of the scores

Input : common-format CSVs + their adaptive Key-State CSVs (from extract_keystates.py)
Output: <name>_sequences.csv    one row per sequence
        <name>_ks_features.csv  one row per Key State, with its features and sequence id

Usage :  python scripts\\find_sequences.py --csv data\\real\\processed\\J03WMX.csv.gz data\\simulated\\match_001.csv data\\simulated\\match_002.csv
         python scripts\\find_sequences.py --csv <same list> --sweep
"""
import argparse
import os
import numpy as np
import pandas as pd
import settings
from scipy.signal import find_peaks

DT = 0.5   # seconds, grid for the sliding comparison

FEATURES = ["log_gap", "ball_x_m", "ball_y_m", "ball_speed", "mean_speed",
            "home_cx", "away_cx", "home_width", "home_depth", "away_width", "away_depth"]


def base_name(csv):
    return csv.replace(".csv.gz", "").replace(".csv", "")


def key_state_features(csv, ks):
    cols = ["frame", "team", "role", "x_m", "y_m", "vx_ms", "vy_ms", "ball_x_m", "ball_y_m"]
    d = pd.read_csv(csv, usecols=cols)
    d = d[d["frame"].isin(ks["frame"])]
    d["speed"] = np.hypot(d["vx_ms"], d["vy_ms"])

    ball = d.drop_duplicates("frame").set_index("frame")[["ball_x_m", "ball_y_m"]]
    speed = d.groupby("frame")["speed"].mean().rename("mean_speed")
    out = d[d["role"] != "GK"]
    g = out.groupby(["frame", "team"]).agg(cx=("x_m", "mean"), xmin=("x_m", "min"), xmax=("x_m", "max"),
                                           ymin=("y_m", "min"), ymax=("y_m", "max"))
    g["width"] = g["ymax"] - g["ymin"]
    g["depth"] = g["xmax"] - g["xmin"]
    g = g[["cx", "width", "depth"]].unstack("team")
    g.columns = [f"{team}_{m}" for m, team in g.columns]

    f = ks[["ks_id", "frame", "time_s", "gap_s"]].set_index("frame").join([ball, speed, g]).reset_index()
    step = np.hypot(f["ball_x_m"].diff(), f["ball_y_m"].diff())
    f["ball_speed"] = (step / f["gap_s"]).fillna(0)
    f["log_gap"] = np.log1p(f["gap_s"].fillna(f["gap_s"].median()))
    return f


def standardised(frames_by_match):
    """Z-scores with one pooled mean / std for all matches."""
    X = {m: f[FEATURES].fillna(f[FEATURES].median()) for m, f in frames_by_match.items()}
    pooled = pd.concat(X.values())
    mu, sd = pooled.mean(), pooled.std()
    return {m: ((x - mu) / sd).to_numpy() for m, x in X.items()}


def grid_score(f, Z, W):
    """Difference between the mean of the previous W s and the next W s, on a regular time grid."""
    t_ks = f["time_s"].to_numpy()
    grid = np.arange(0, t_ks[-1], DT)
    idx = np.clip(np.searchsorted(t_ks, grid, side="right") - 1, 0, len(t_ks) - 1)
    Zg = Z[idx]                                    # each Key State's features held until the next one
    w = int(round(W / DT))
    cs = np.vstack([np.zeros((1, Zg.shape[1])), np.cumsum(Zg, axis=0)])
    n = len(grid)
    score = np.zeros(n)
    j = np.arange(w, n - w + 1)
    left = (cs[j] - cs[j - w]) / w
    right = (cs[j + w] - cs[j]) / w
    score[j] = np.linalg.norm(left - right, axis=1) / np.sqrt(Zg.shape[1])
    return grid, score


def boundaries(grid, score, thr, W):
    peaks, _ = find_peaks(score, height=thr, distance=int(round(W / DT)))
    return grid[peaks]


def build_sequences(t_end, tb):
    starts = np.r_[0.0, tb]
    ends = np.r_[tb, t_end]
    seq = pd.DataFrame({"seq_id": np.arange(len(starts)), "start_time_s": starts, "end_time_s": ends})
    seq["duration_s"] = seq["end_time_s"] - seq["start_time_s"]
    return seq


def describe(seq):
    d = seq["duration_s"]
    return {"sequences": len(seq), "dur_min_s": round(d.min(), 1), "dur_median_s": round(d.median(), 1),
            "dur_mean_s": round(d.mean(), 1), "dur_max_s": round(d.max(), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=None, help="default: every match in settings.json that has Key States")
    ap.add_argument("--W", type=float, default=settings.get("sequences", "window_s", default=5.0))       # seconds on each side of a candidate boundary
    ap.add_argument("--c", type=float, default=settings.get("sequences", "peak_c", default=0.5))       # peak must exceed pooled mean + c * pooled std
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    if not a.csv:
        a.csv = [settings.csv(m) for m in settings.available(settings.REAL + settings.SIM) if os.path.exists(settings.base(m) + "_adaptive_ks.csv")]

    feats = {}
    for csv in a.csv:
        ks = pd.read_csv(base_name(csv) + "_adaptive_ks.csv")
        feats[csv] = key_state_features(csv, ks)
    Z = standardised(feats)

    def run(W, c):
        scores = {m: grid_score(feats[m], Z[m], W) for m in a.csv}
        pooled = np.concatenate([s[s > 0] for _, s in scores.values()])
        thr = pooled.mean() + c * pooled.std()
        res = {}
        for m, (grid, s) in scores.items():
            tb = boundaries(grid, s, thr, W)
            res[m] = (build_sequences(feats[m]["time_s"].iloc[-1], tb), tb, grid, s)
        return res, thr

    if a.sweep:
        rows = []
        for W in (3, 5, 8, 12):
            for c in (0.5, 1.0, 1.5):
                res, thr = run(W, c)
                for m in a.csv:
                    seq = res[m][0]
                    dsc = describe(seq)
                    rows.append({"W": W, "c": c, "match": base_name(m).split("\\")[-1].split("/")[-1],
                                 "per_min": round(len(seq) / (feats[m]["time_s"].iloc[-1] / 60), 2), **dsc})
        print(pd.DataFrame(rows).to_string(index=False))
        return

    res, thr = run(a.W, a.c)
    print(f"shared threshold {thr:.3f} (W = {a.W} s, c = {a.c})")
    summary = []
    for m in a.csv:
        seq, tb, grid, s = res[m]
        f = feats[m]
        f["seq_id"] = np.searchsorted(tb, f["time_s"].to_numpy(), side="right")
        f.round(3).to_csv(base_name(m) + "_ks_features.csv", index=False)
        seq.round(2).to_csv(base_name(m) + "_sequences.csv", index=False)
        summary.append({"match": base_name(m).split("\\")[-1].split("/")[-1],
                        "per_min": round(len(seq) / (f["time_s"].iloc[-1] / 60), 2), **describe(seq)})
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()

