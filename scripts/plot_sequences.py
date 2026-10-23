"""Show where the sequence boundaries fall: ball position over time, boundaries as vertical lines.

Grey shading = ball dead / set piece (shown for description only; the boundaries never used it).
All three panels show the first 300 s so real and simulated matches are on the same time scale.

Usage:  python scripts\\plot_sequences.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SHOW_S = 300
PANELS = [
    ("Real: Köln 1–2 Bayern (first 5 min)", "data/real/processed/J03WMX.csv.gz"),
    ("Simulated match 1", "data/simulated/match_001.csv"),
    ("Simulated match 2", "data/simulated/match_002.csv"),
]


def base_name(csv):
    return csv.replace(".csv.gz", "").replace(".csv", "")


fig, axes = plt.subplots(len(PANELS), 1, figsize=(11, 7), sharex=True)
for ax, (title, csv) in zip(axes, PANELS):
    d = pd.read_csv(csv, usecols=["frame", "time_s", "ball_x_m", "game_mode"]).drop_duplicates("frame")
    d = d[d["time_s"] <= SHOW_S]
    seq = pd.read_csv(base_name(csv) + "_sequences.csv")
    feat = pd.read_csv(base_name(csv) + "_ks_features.csv")

    # description only: shade the periods where the ball is not in normal play
    off = (d["game_mode"] != "normal").to_numpy()
    t = d["time_s"].to_numpy()
    start = None
    for i, flag in enumerate(off):
        if flag and start is None:
            start = t[i]
        if (not flag or i == len(off) - 1) and start is not None:
            ax.axvspan(start, t[i], color="#D5DBDF", lw=0)
            start = None

    ax.plot(d["time_s"], d["ball_x_m"], color="#14532D", lw=0.9)
    for tb in seq["start_time_s"].iloc[1:]:
        if tb <= SHOW_S:
            ax.axvline(tb, color="#C0392B", lw=1.2)
    ks = feat[feat["time_s"] <= SHOW_S]
    ax.scatter(ks["time_s"], [-6] * len(ks), s=4, color="#2F6DB5", clip_on=False)
    n_seq = int((seq["start_time_s"] <= SHOW_S).sum())
    ax.set_title(f"{title}: {n_seq} sequences in these 5 min", loc="left", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel("ball x (m)")
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)

axes[-1].set_xlabel("time (s)   |   red lines = sequence boundaries, blue dots = Key States, grey = ball not in normal play")
plt.tight_layout()
plt.savefig("outputs/sequence_boundaries_v2.png", dpi=150)
print("saved outputs/sequence_boundaries_v2.png")

