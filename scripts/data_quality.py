"""Quality check of tracking data, the same numbers for every dataset in settings.json (so a new source can be judged against DFL).

Per match: length, frames with all 22 players, share of positions that were really seen (datasets with a visibility flag),
share of frames with a seen ball, player speeds (median, 99.9th percentile, share above 12 m/s = impossible), share of positions
outside the pitch, longest unbroken stretch, Key States (count, median gap) when Key States exist.

Input : the common-format file of every match of the chosen datasets (settings.json)
Output: outputs/data_quality.csv (per match) and a printed summary per dataset

Usage :  python scripts\\data_quality.py                    (every dataset that has matches on disk)
         python scripts\\data_quality.py --dataset pff dfl
"""
import argparse
import os
import numpy as np
import pandas as pd
import settings

L, W = 105.0, 68.0


def one(m):
    d = pd.read_csv(settings.csv(m))
    f = d.drop_duplicates("frame").sort_values("frame")
    fr = f["frame"].to_numpy()
    br = np.flatnonzero(np.diff(fr) > 1)
    seg = np.diff(np.r_[-1, br, len(fr) - 1])                                   # frames in every unbroken stretch
    sp = np.hypot(d["vx_ms"], d["vy_ms"]).dropna()
    row = {"match": m, "minutes": round(len(fr) / 600, 1), "span_min": round((fr[-1] - fr[0]) / 600, 1),
           "coverage_%": round(100 * len(fr) / (fr[-1] - fr[0] + 1), 1), "stretches": len(seg),
           "longest_stretch_s": round(seg.max() / 10, 1), "median_stretch_s": round(float(np.median(seg)) / 10, 1),
           "player_visible_%": round(100 * d["visible"].mean(), 1) if "visible" in d else np.nan,
           "ball_visible_%": round(100 * f["ball_visible"].mean(), 1) if "ball_visible" in f else np.nan,
           "speed_median_ms": round(float(sp.median()), 2), "speed_p999_ms": round(float(sp.quantile(0.999)), 1),
           "impossible_speed_%": round(100 * float((sp > 12).mean()), 3),
           "off_pitch_%": round(100 * float(((d["x_m"] < -1) | (d["x_m"] > L + 1) | (d["y_m"] < -1) | (d["y_m"] > W + 1)).mean()), 2),
           "dead_ball_%": round(100 * float((f["game_mode"] == "dead").mean()), 1)}
    ks = settings.base(m) + "_adaptive_ks.csv"
    if os.path.exists(ks):
        k = pd.read_csv(ks)
        row.update({"key_states": len(k), "ks_gap_median_s": round(float(k["gap_s"].median()), 2),
                    "ks_gap_p99_s": round(float(k["gap_s"].quantile(0.99)), 1)})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=None)
    a = ap.parse_args()
    names = a.dataset or [n for n in settings._DATASETS if settings._DATASETS[n]["kind"] == "real"]
    rows = []
    for n in names:
        for m in settings.dataset_matches(n):
            r = one(m)
            r["dataset"] = n
            rows.append(r)
            print(f"  {n} {m} done", flush=True)
    t = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    t.to_csv("outputs/data_quality.csv", index=False)
    cols = [c for c in t.columns if c not in ("match", "dataset")]
    print("\nMEAN PER DATASET\n")
    print(t.groupby("dataset")[cols].mean().round(2).T.to_string())
    print("\nwrote outputs/data_quality.csv")


if __name__ == "__main__":
    main()
