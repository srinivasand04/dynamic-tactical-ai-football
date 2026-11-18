"""Weak spot 4: where does the movement model go wrong, and why did MORE data make it worse?

A. Error by role. The tuning ladder saved every held-out prediction (outputs/tuning/preds/, 7 matches per version).
   Here the error (metres, predicted vs real movement to the next Key State) is split by the player's role
   (keeper, centre-backs, full-backs, defensive/central mid, wide mid, attacking mid, striker, ball), for:
   the current model, the 3-seed ensemble (the best), GAT, and the model trained with 20 more matches.
   No training: the saved predictions are re-read.

B. Are the datasets alike? For the 7 Bundesliga matches (test set), the 12 World Cup and the 8 A-League matches used by
   the 'more data' version: time between Key States, how far players move between them, player speed.
   If the extra datasets move on a different clock or scale, mixing them without normalising teaches the model the
   wrong step size: that is the case for 'normalise each dataset, then mix'.

Output: outputs/tuning/movement_by_role.csv, dataset_compare.csv, movement_by_role.png (new files only)
Usage : .venv-analysis\\Scripts\\python.exe scripts\\movement_by_role.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "tuning")
PREDS = os.path.join(OUT, "preds")
SUFFIX = settings.graph_suffix()
GROUP = {0: "keeper", 1: "centre-back", 2: "full-back", 3: "full-back", 4: "central mid", 5: "central mid",
         6: "wide mid", 7: "wide mid", 8: "attacking mid", 9: "striker"}
ORDER = ["keeper", "centre-back", "full-back", "central mid", "wide mid", "attacking mid", "striker", "ball"]
VERSIONS = {"current": ["ours__window_3____current"], "3-seed ensemble (best)": ["ours__window_3____current", "ours_seed_1", "ours_seed_2"],
            "GAT": ["gat"], "+ 20 more matches": ["ours___20_more_matches__12_world_cup__8_a_league"]}
WINDOW_START = 4            # train_gnn.EVAL_START: the predictions are for Key States 4 .. K-2


def roles(m, n):
    g = np.load(settings.base(m) + f"_graphs{SUFFIX}.npz")
    r = g["role"][WINDOW_START:WINDOW_START + n]
    lab = np.vectorize(lambda v: GROUP.get(int(v), "unknown"))(r).astype(object)
    lab[:, 22] = "ball"
    return lab


def by_role():
    folds = settings.available(settings.REAL)
    rows = []
    for name, slugs in VERSIONS.items():
        for m in folds:
            zs = [np.load(os.path.join(PREDS, f"{s}_{m}.npz")) for s in slugs]
            pred = np.mean([z["pred"] for z in zs], axis=0)
            tgt, ok = zs[0]["target"], zs[0]["ok"]
            err = np.linalg.norm(pred - tgt, axis=2)
            lab = roles(m, len(err))
            for g in ORDER:
                sel = ok & (lab == g)
                if sel.any():
                    rows.append({"version": name, "match": m, "role": g, "n": int(sel.sum()), "err_m": float(err[sel].mean())})
            pl = ok.copy()
            pl[:, 22] = False
            rows.append({"version": name, "match": m, "role": "all players", "n": int(pl.sum()), "err_m": float(err[pl].mean())})
    df = pd.DataFrame(rows)
    t = df.groupby(["role", "version"])["err_m"].mean().unstack()[list(VERSIONS)].reindex(["all players"] + ORDER)
    cur = df[df.version == "current"].set_index(["match", "role"])["err_m"]
    best = df[df.version == "3-seed ensemble (best)"].set_index(["match", "role"])["err_m"]
    more = df[df.version == "+ 20 more matches"].set_index(["match", "role"])["err_m"]
    t["ensemble_gain_m"] = (best - cur).groupby(level=1).mean()
    t["more_data_loss_m"] = (more - cur).groupby(level=1).mean()
    t["more_data_worse_in"] = (more - cur).gt(0).groupby(level=1).sum().astype(int).astype(str) + "/" + str(len(settings.available(settings.REAL)))
    return t


def dataset_compare():
    sets = {"Bundesliga (test set)": settings.available(settings.REAL)}
    for name, n, label in (("pff", 12, "World Cup (extra)"), ("skillcorner", 8, "A-League (extra)")):
        sets[label] = [m for m in settings.dataset_matches(name) if os.path.exists(settings.base(m) + f"_graphs{SUFFIX}.npz")][:n]
    rows = []
    for label, ms in sets.items():
        gaps, steps, speed, pres = [], [], [], []
        for m in ms:
            g = np.load(settings.base(m) + f"_graphs{SUFFIX}.npz")
            xy, present, t = g["xy"], g["present"], g["time_s"]
            gap = np.diff(t)
            mv = np.linalg.norm(xy[1:, :22] - xy[:-1, :22], axis=2)
            ok = present[1:, :22] & present[:-1, :22] & np.isfinite(mv)
            gaps.append(gap)
            steps.append(mv[ok])
            v = np.linalg.norm(g["vel"][:, :22], axis=2)
            speed.append(v[present[:, :22] & np.isfinite(v)])
            pres.append(present[:, :22].sum(1))
        gaps, steps, speed, pres = (np.concatenate(x) for x in (gaps, steps, speed, pres))
        rows.append({"dataset": label, "matches": len(ms), "key_states": len(gaps) + len(ms),
                     "gap_s_median": np.median(gaps), "gap_s_p90": np.percentile(gaps, 90),
                     "move_per_step_m_median": np.median(steps), "move_per_step_m_p90": np.percentile(steps, 90),
                     "speed_ms_median": np.median(speed), "speed_ms_p90": np.percentile(speed, 90),
                     "players_seen_median": np.median(pres)})
    return pd.DataFrame(rows)


def chart(t, dc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(18, 6.5), gridspec_kw={"width_ratios": [2, 1]})
    a = ax[0]
    x = np.arange(len(t))
    cols = ["#b5b3ab", "#2a7ad5", "#8fb1e0", "#e0483b"]
    for i, (v, c) in enumerate(zip(VERSIONS, cols)):
        a.bar(x + (i - 1.5) * 0.2, t[v], 0.2, color=c, label=v)
    a.set_xticks(x, t.index, rotation=20)
    a.set_ylabel("mean error (m), 7 held-out matches")
    a.set_title("Movement error by role")
    a.legend()
    a.grid(axis="y", alpha=0.3)
    a = ax[1]
    y = np.arange(len(dc))
    a.barh(y - 0.2, dc["gap_s_median"], 0.4, color="#2a7ad5", label="time between Key States (s, median)")
    a.barh(y + 0.2, dc["move_per_step_m_median"], 0.4, color="#e0483b", label="player move per step (m, median)")
    a.set_yticks(y, dc["dataset"])
    a.set_title("Are the datasets alike?")
    a.legend(fontsize=9)
    a.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=105)


def main():
    t = by_role()
    dc = dataset_compare()
    t.round(4).to_csv(os.path.join(OUT, "movement_by_role.csv"))
    dc.round(4).to_csv(os.path.join(OUT, "dataset_compare.csv"), index=False)
    chart(t, dc, os.path.join(OUT, "movement_by_role.png"))
    pd.set_option("display.width", 220)
    print("A. Error by role (metres, mean of 7 held-out matches):")
    print(t.round(3).to_string())
    print("\nB. Datasets:")
    print(dc.round(3).to_string(index=False))
    print("\nwrote outputs/tuning/movement_by_role.csv, dataset_compare.csv, movement_by_role.png")


if __name__ == "__main__":
    main()
