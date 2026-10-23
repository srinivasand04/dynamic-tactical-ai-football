"""Draw the graphs: one Key State as a graph (real and simulated), and four consecutive Key States showing time.

Input : <name>_graphs.npz from build_graphs.py
Output: outputs/graph_snapshot.png   one Key State as a graph, real beside simulated
        outputs/graph_time.png       four consecutive Key States; faded dots = previous Key State, arrows = movement

Usage :  python scripts\\plot_graph.py
         python scripts\\plot_graph.py --real-time 600 --sim-time 100
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

REAL = "data/real/processed/J03WMX_graphs.npz"
SIM = "data/simulated/match_001_graphs.npz"
L, W = 105, 68
HOME, AWAY, BALLC = "#1B7F4B", "#2F6DB5", "#E8833A"


def pitch(ax):
    ax.add_patch(Rectangle((0, 0), L, W, fill=False, ec="#6B7A89", lw=1))
    ax.plot([L / 2, L / 2], [0, W], color="#6B7A89", lw=1)
    ax.add_patch(Circle((L / 2, W / 2), 9.15, fill=False, ec="#6B7A89", lw=1))
    for x0 in (0, L - 16.5):
        ax.add_patch(Rectangle((x0, W / 2 - 20.15), 16.5, 40.3, fill=False, ec="#6B7A89", lw=1))
    ax.set_xlim(-2, L + 2)
    ax.set_ylim(-2, W + 2)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_graph(ax, g, k, title):
    pitch(ax)
    xy, team, present = g["xy"][k], g["team"], g["present"][k]
    for name, adj, color, lw, alpha in (("team", g["adj_team"][k], "#9AA5AF", 0.5, 0.35),
                                        ("opp", g["adj_opp"][k], "#C0392B", 1.5, 0.9),
                                        ("ball", g["adj_ball"][k], BALLC, 1.8, 0.9)):
        ii, jj = np.nonzero(np.triu(adj))
        for i, j in zip(ii, jj):
            ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color=color, lw=lw, alpha=alpha, zorder=2)
    for i in range(22):
        if present[i]:
            ax.scatter(*xy[i], s=70, color=HOME if team[i] == 0 else AWAY, ec="white", zorder=3)
    ax.scatter(*xy[22], s=90, color=BALLC, ec="black", zorder=4)
    ax.set_title(title, fontsize=10, loc="left")


def nearest(g, t):
    return int(np.argmin(np.abs(g["time_s"] - t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-time", type=float, default=600)
    ap.add_argument("--sim-time", type=float, default=100)
    a = ap.parse_args()
    real, sim = np.load(REAL), np.load(SIM)
    kr, ks = nearest(real, a.real_time), nearest(sim, a.sim_time)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    draw_graph(axes[0], real, kr, f"Real: Köln–Bayern, Key State {int(real['ks_id'][kr])} at {real['time_s'][kr]:.1f} s")
    draw_graph(axes[1], sim, ks, f"Simulated: match_001, Key State {int(sim['ks_id'][ks])} at {sim['time_s'][ks]:.1f} s")
    fig.text(0.5, 0.02, "dots: green = home, blue = away, orange = ball    |    lines: grey = teammates, red = opponents within 8 m, orange = ball's 3 nearest players",
             ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig("outputs/graph_snapshot.png", dpi=150)

    fig, axes = plt.subplots(2, 4, figsize=(18, 6.4))
    for row, (g, k0, label) in enumerate(((real, kr, "Real"), (sim, ks, "Simulated"))):
        for c in range(4):
            k = k0 + c
            ax = axes[row, c]
            pitch(ax)
            xy, prev, team, present = g["xy"][k], g["xy"][k - 1], g["team"], g["present"][k]
            for i in range(22):
                if present[i] and not np.isnan(prev[i, 0]):
                    col = HOME if team[i] == 0 else AWAY
                    ax.scatter(*prev[i], s=40, color=col, alpha=0.25, zorder=2)
                    ax.annotate("", xy=xy[i], xytext=prev[i], arrowprops=dict(arrowstyle="->", color=col, lw=0.9, alpha=0.8), zorder=3)
                    ax.scatter(*xy[i], s=55, color=col, ec="white", zorder=4)
            ax.scatter(*prev[22], s=50, color=BALLC, alpha=0.3, zorder=2)
            ax.annotate("", xy=xy[22], xytext=prev[22], arrowprops=dict(arrowstyle="->", color=BALLC, lw=1.4), zorder=3)
            ax.scatter(*xy[22], s=80, color=BALLC, ec="black", zorder=5)
            ax.set_title(f"{label}: t = {g['time_s'][k]:.1f} s  (+{g['gap_s'][k]:.1f} s)", fontsize=10, loc="left")
    fig.text(0.5, 0.01, "faded dot = position at the previous Key State, arrow = movement since then; a temporal edge links each player to itself across Key States",
             ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig("outputs/graph_time.png", dpi=130)
    print("saved outputs/graph_snapshot.png and outputs/graph_time.png")


if __name__ == "__main__":
    main()
