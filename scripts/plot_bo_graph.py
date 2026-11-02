"""Draw the ball-owner graph (bo_graph.py) on the pitch for a few consecutive Key States, real and simulated.

Picture: ring = ball owner. Green lines to teammates: thicker and darker = a higher weight (number written on the line).
Red dashed line with a minus number = the teammate is offside (his weight is multiplied by -1, the size is kept).
Orange X = the opponent who blocks the passing lane (only drawn when the lane score is below 0.7).
Red thin dashed lines = opponents within the threshold of the ball owner. Dashed vertical line = the offside line.
Opponents not connected to the ball owner are faded.

Usage :  python scripts\\plot_bo_graph.py                                  (real J03WMX at 600 s, simulated match_001 at 100 s)
         python scripts\\plot_bo_graph.py --match J03WN1 --time 1500 --sim hard_001 --sim-time 60
         python scripts\\plot_bo_graph.py --mean product                    (plain product instead of the geometric mean)
         python scripts\\plot_bo_graph.py --learned                         (numbers learned by fit_bo_weights.py)
Output:  outputs/bo_graph.png and outputs/bo_graph_edges.csv (every edge drawn)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle

import bo_graph
import settings

L, W = 105, 68
HOME, AWAY = "#2a78d6", "#eb6834"


def pitch(ax):
    ax.add_patch(Rectangle((0, 0), L, W, fill=False, ec="#898781", lw=1))
    ax.plot([L / 2, L / 2], [0, W], color="#898781", lw=1)
    ax.add_patch(Circle((L / 2, W / 2), 9.15, fill=False, ec="#898781", lw=1))
    for x0 in (0, L - 16.5):
        ax.add_patch(Rectangle((x0, W / 2 - 20.15), 16.5, 40.3, fill=False, ec="#898781", lw=1))
    ax.set_xlim(-2, L + 2)
    ax.set_ylim(-3, W + 2)
    ax.set_aspect("equal")
    ax.axis("off")


def load(match):
    g = np.load(settings.graphs(match))
    c = pd.read_csv(settings.csv(match), usecols=["frame", "ball_owner_team", "game_mode"]).drop_duplicates("frame").set_index("frame").loc[g["frame"]]
    return g, c["ball_owner_team"].to_numpy(), (c["game_mode"] == "normal").to_numpy()


def pick(g, owner, live, t0, n, params):
    out = []
    for k in np.flatnonzero(g["time_s"] >= t0):
        if live[k]:
            b = bo_graph.build(g["xy"][k], g["present"][k], g["team"], owner[k], **params)
            if b:
                out.append((k, b))
        if len(out) == n:
            break
    return out


def label(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=7, ha="center", va="center", color=color, zorder=7,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))


def draw(ax, g, k, b, match, top=0):
    pitch(ax)
    xy, team, side = g["xy"][k], g["team"], b["side"]
    col = lambda t: HOME if t == 0 else AWAY
    linked = {e["to"] for e in b["edges"] if e["type"] == "opponent"}
    x_line = b["offside_line"] if side == 0 else L - b["offside_line"]
    ax.plot([x_line, x_line], [0, W], color="#ba7517", lw=1, ls=(0, (6, 4)), zorder=1)
    for i in b["mates"] + b["opps"]:
        faded = i in b["opps"] and i not in linked
        ax.scatter(*xy[i], s=55, color=col(team[i]), ec="white", zorder=4, alpha=0.3 if faded else 1)
    ax.scatter(*xy[b["bo"]], s=220, facecolors="none", edgecolors="black", linewidths=2, zorder=5)
    ax.scatter(*xy[bo_graph.BALL], s=30, color="#222222", zorder=6)
    ranked = sorted([e for e in b["edges"] if e["type"] == "teammate"], key=lambda e: -e["weight_abs"])
    shown = {id(e) for e in (ranked[:top] if top else ranked)}
    shown |= {id(e) for e in ranked if e["offside"]}                      # an offside teammate is always drawn
    for e in b["edges"]:
        if e["type"] == "teammate" and id(e) not in shown:
            continue
        (x0, y0), (x1, y1) = xy[e["from"]], xy[e["to"]]
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        if e["type"] == "opponent":
            ax.plot([x0, x1], [y0, y1], color="#c0392b", lw=1.2, ls=(0, (4, 3)), zorder=3)
            continue
        w = e["weight_abs"]
        if e["offside"]:
            ax.plot([x0, x1], [y0, y1], color="#c0392b", lw=0.8 + 3.5 * w, ls=(0, (7, 4)), zorder=3, alpha=0.9)
            label(ax, xm, ym, f"{e['weight']:.2f}", "#c0392b")
        else:
            ax.plot([x0, x1], [y0, y1], color=plt.cm.YlGn(0.35 + 0.65 * min(w, 1)), lw=0.8 + 3.5 * w, zorder=2, alpha=0.95)
            label(ax, xm, ym, f"{e['weight']:.2f}", "#14532d")
        if e["lane"] < 0.7 and e["blocker"] is not None:
            bx, by = xy[e["blocker"]]
            ax.plot([bx], [by], marker="x", color="#e8833a", ms=9, mew=2, zorder=8)
    owner = "home" if side == 0 else "away"
    ax.set_title(f"{match}  t = {g['time_s'][k]:.1f} s  ball owner: {owner}", loc="left", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="J03WMX")
    ap.add_argument("--time", type=float, default=600)
    ap.add_argument("--sim", default="match_001")
    ap.add_argument("--sim-time", type=float, default=100)
    ap.add_argument("--n", type=int, default=3, help="Key States per row")
    ap.add_argument("--top", type=int, default=0, help="draw only the best N teammate edges (offside ones are always drawn); 0 = all")
    ap.add_argument("--out", default="outputs/bo_graph.png")
    ap.add_argument("--mean", choices=["geometric", "product"], default="geometric")
    ap.add_argument("--thr", type=float, default=bo_graph.PARAMS["thr_m"], help="opponents within this many metres of the ball owner are connected")
    ap.add_argument("--learned", action="store_true", help="use the numbers learned from real passes (fit_bo_weights.py)")
    a = ap.parse_args()
    params = dict(mean=a.mean, thr_m=a.thr, **(bo_graph.learned() if a.learned else {}))

    rows = [(a.match, a.time)] + ([(a.sim, a.sim_time)] if a.sim else [])
    fig, axes = plt.subplots(len(rows), a.n, figsize=((10 if a.n == 1 else 5.6) * a.n, (6.8 if a.n == 1 else 3.8) * len(rows)), squeeze=False)
    out = []
    for r, (m, t0) in enumerate(rows):
        g, owner, live = load(m)
        found = pick(g, owner, live, t0, a.n, params)
        for c in range(a.n):
            if c >= len(found):
                axes[r, c].axis("off")
                continue
            k, b = found[c]
            draw(axes[r, c], g, k, b, m, a.top)
            for e in b["edges"]:
                out.append({"match": m, "ks_id": int(g["ks_id"][k]), "time_s": round(float(g["time_s"][k]), 1),
                            **{kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in e.items() if kk != "blocker"}})
    fig.text(0.5, 0.004, "ring = ball owner | green: weight to a teammate (thicker = better) | red dashed + minus = offside, size kept\n"
                         "orange X = blocking opponent | thin red dashed = opponents within the threshold | vertical dashed = offside line",
             ha="center", fontsize=7.5)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(a.out, dpi=150)
    df = pd.DataFrame(out)
    df.to_csv("outputs/bo_graph_edges.csv", index=False)
    print(f"saved {a.out} and outputs/bo_graph_edges.csv")
    t = df[df["type"] == "teammate"]
    print(t.groupby("match")[["free", "reach", "goal", "lane", "weight_abs"]].mean().round(2).to_string())
    print("offside teammates per graph:", t.groupby("match")["offside"].sum().to_dict())


if __name__ == "__main__":
    main()
