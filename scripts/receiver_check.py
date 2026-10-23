"""Receiver check: does the ball-owner graph rank the REAL receiver of a pass at the top?

For every completed open-play pass in the real matches (extract_passes.py), the graph is built with the real passer as the
ball owner. Each teammate gets a weight; we look at the place (rank) the real receiver gets among the teammates.

Methods compared (each one ranks the same teammates)
    chance           pick a teammate at random                               (top-1 = 1 / number of teammates)
    nearest          the teammate closest to the passer                      (reach alone)
    free / goal / lane   one ingredient alone
    weight (hand)    the full BO weight with the starting numbers in bo_graph.PARAMS
    weight (learned) the full BO weight with the numbers learned by fit_bo_weights.py (only if that file exists)

Measures
    top1_%   the receiver is the teammate ranked first
    top3_%   the receiver is in the first three
    rank     average place of the receiver (1 = best; about 5.5 for chance with 10 teammates)
Also printed: how often the receiver of a completed pass is called offside by our rule (should be close to 0;
if not, the offside rule or the moment of the pass is off).

Output : the table, outputs/receiver_check.csv, outputs/receiver_check.png
Usage  :  python scripts\\receiver_check.py
          python scripts\\receiver_check.py --set-pieces        (also use passes from set pieces)
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bo_graph
import settings


def passes(m, set_pieces=False):
    """The usable passes of a match: completed, receiver known, a teammate on the pitch."""
    z = np.load(settings.base(m) + "_passes.npz", allow_pickle=True)
    keep = z["complete"] & (z["receiver"] >= 0)
    if not set_pieces:
        keep &= ~z["set_piece"]
    rows = []
    for i in np.flatnonzero(keep):
        p, r = int(z["passer"][i]), int(z["receiver"][i])
        if r != p and z["team"][r] == z["team"][p] and z["present"][i, r]:
            rows.append(i)
    return z, rows


def rank_of(scores, target):
    """1 = the target has the highest score; ties share the place in the middle."""
    s = scores[target]
    return 1 + sum(v > s for v in scores.values()) + 0.5 * (sum(v == s for v in scores.values()) - 1)


def evaluate(matches, methods, set_pieces=False):
    out = []
    for m in matches:
        z, rows = passes(m, set_pieces)
        for i in rows:
            p, r = int(z["passer"][i]), int(z["receiver"][i])
            row = {"match": m, "teammates": 0}
            for name, params in methods.items():
                g = bo_graph.build(z["xy"][i], z["present"][i], z["team"], None, bo=p, **params)
                if g is None:
                    break
                e = {x["to"]: x for x in g["edges"] if x["type"] == "teammate"}
                if r not in e:
                    break
                row["teammates"] = len(e)
                row["offside_flag"] = e[r]["offside"]
                row[name] = rank_of({k: v["weight"] for k, v in e.items()}, r)
                if name == "weight (hand)":
                    row["nearest"] = rank_of({k: -v["metres"] for k, v in e.items()}, r)
                    for part in ("free", "goal", "lane"):
                        row[part] = rank_of({k: v[part] for k, v in e.items()}, r)
            else:
                row["chance"] = (row["teammates"] + 1) / 2
                out.append(row)
    return pd.DataFrame(out)


def summary(df, names):
    rows = []
    for name in names:
        r = df[name]
        top1 = 100 / df["teammates"] if name == "chance" else 100 * (r <= 1)
        top3 = 300 / df["teammates"] if name == "chance" else 100 * (r <= 3)
        rows.append({"method": name, "top1_%": top1.mean(), "top3_%": top3.mean(), "rank": r.mean()})
    return pd.DataFrame(rows).set_index("method")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-pieces", action="store_true")
    a = ap.parse_args()
    matches = [m for m in settings.available(settings.REAL) if os.path.exists(settings.base(m) + "_passes.npz")]
    if not matches:
        raise SystemExit("no _passes.npz files: run scripts/extract_passes.py first")
    methods = {"weight (hand)": {}}
    if os.path.exists(bo_graph.LEARNED):
        methods["weight (learned)"] = bo_graph.learned()
    df = evaluate(matches, methods, a.set_pieces)
    names = ["chance", "nearest", "free", "goal", "lane"] + list(methods)

    print(f"{len(df)} completed passes with a known receiver, {len(matches)} real matches, "
          f"on average {df['teammates'].mean():.1f} teammates to choose from\n")
    s = summary(df, names)
    print("ALL MATCHES\n")
    print(s.round(1).to_string())
    print("\nTOP-1 % PER MATCH\n")
    per = pd.DataFrame({m: summary(g, names)["top1_%"] for m, g in df.groupby("match")}).T
    print(per.round(1).to_string())
    print(f"\nreceivers of completed passes that our rule calls offside: {100 * df['offside_flag'].mean():.1f}%")

    os.makedirs("outputs", exist_ok=True)
    s.round(2).to_csv("outputs/receiver_check.csv")
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(names) + 1.2))
    colors = ["#2a78d6" if n.startswith("weight") else "#b4b2a9" for n in names]
    y = np.arange(len(names))[::-1]
    ax.barh(y, s["top1_%"], color=colors, height=0.6)
    for yy, v in zip(y, s["top1_%"]):
        ax.text(v + 0.8, yy, f"{v:.0f}%", va="center", fontsize=9, color="#3d3d3a")
    ax.axvline(s.loc["chance", "top1_%"], color="#73726c", lw=1, ls=(0, (4, 3)))
    ax.set_yticks(y, names)
    ax.set_xlabel("real receiver ranked first (%)")
    ax.set_title("Is the real receiver the teammate the graph ranks first?  (dashed = chance)", loc="left", fontsize=10)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/receiver_check.png", dpi=150)
    print("\nsaved outputs/receiver_check.csv and outputs/receiver_check.png")


if __name__ == "__main__":
    main()
