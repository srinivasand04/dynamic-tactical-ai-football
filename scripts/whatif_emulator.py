"""What-if emulator: predict a GRF what-if result in milliseconds instead of running GRF (about 6 s x seeds per run).

Trained on the Tactics Tournament (28 real moments x 36 attack/defence plan pairs x 6 seeds, hold): for each moment and
plan pair, the change in danger over time (threat_auc, x100) against 'no change + no change' with the same seeds.
Input: the moment's start picture (who is where, attack towards x = 105: the 13 threat-model features, ball position,
spread and line heights of both teams) + the two plans (one-hot).
Model: the table (the plan pair's average effect on the training moments) + a small neural net that learns only the
moment's own correction (1 hidden layer, dropout, strong weight decay, 5 nets averaged): with nothing to learn it falls
back to the table. Leave-one-moment-out: every moment is predicted by nets that never saw it.

Is it useful? Compared on held-out moments with
    'zero'        always predict no effect
    'table'       the pair's average effect on the other 27 moments (what the tournament table says)
The emulator only helps if it beats 'table' (it must know something about THIS moment). Measures: error (pp), rank
correlation with GRF over the 35 plan pairs of a moment, and how often its best defence for a moment is GRF's best
(or within the 95% interval of it).

Output: outputs/tournament/emulator_cv.csv, emulator_summary.csv, emulator.pt, emulator_chart.png
Usage : .venv-analysis\\Scripts\\python.exe scripts\\whatif_emulator.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tactics_tournament import ATTACK, DEFENCE, NAME, OUT, ROOT  # noqa: E402

L, W = 105.0, 68.0
METRIC = "threat_auc"


def moment_features(seg, start_s):
    """The start picture of a moment as numbers (attacking team towards x = 105)."""
    import value_models as V
    import whatif as WI
    pos, ball, _, att, _, _ = WI.moment(seg, start_s, "baseline", 3.0)
    P, B = pos[:, -1], ball[-1]
    if att == "home":
        A, D = P[:11], P[11:]
    else:
        A, D, B = WI.flip(P[11:]), WI.flip(P[:11]), WI.flip(B)
    f = V.state_features(B, A, D)
    Ao, Do = A[1:], D[1:]
    f.update({"ball_x": B[0], "ball_y": B[1], "att_width": np.ptp(Ao[:, 1]), "def_width": np.ptp(Do[:, 1]),
              "att_front": Ao[:, 0].max(), "def_line": np.sort(Do[:, 0])[-4:].mean(), "def_depth": np.ptp(Do[:, 0]),
              "att_in_half": float((Ao[:, 0] > L / 2).sum()), "keeper_x": D[0, 0]})
    return f


MIN_SEEDS = int(os.environ.get("EMU_MIN_SEEDS", "3"))


def dataset():
    runs = pd.read_csv(os.path.join(OUT, "runs.csv"))
    runs = runs[runs["error"].isna()] if "error" in runs else runs
    base = runs[runs.scenario == "none+none"].set_index(["moment", "seed"])[METRIC]
    rows = []
    for (m, a, d), g in runs.groupby(["moment", "attack", "defence"]):
        if (a, d) == ("none", "none"):
            continue
        y = g.set_index(["moment", "seed"])[METRIC]
        c = y.index.intersection(base.index)
        if len(c) < MIN_SEEDS:
            continue
        diff = (y.loc[c] - base.loc[c]) * 100
        rows.append({"moment": m, "attack": a, "defence": d, "y": diff.mean(), "y_se": diff.std(ddof=1) / np.sqrt(len(c)) if len(c) > 1 else 1.0, "seeds": len(c)})
    df = pd.DataFrame(rows)
    mo = pd.read_csv(os.path.join(ROOT, "outputs", "whatif", "pairs_moments.csv")).set_index("moment")
    feats = {m: moment_features(os.path.join(ROOT, mo.loc[m, "segment"]), float(mo.loc[m, "start_s"])) for m in df.moment.unique()}
    F = pd.DataFrame(feats).T
    return df, F


def design(df, F, mu=None, sd=None):
    S = F.loc[df.moment].to_numpy(float)
    if mu is None:
        mu, sd = S.mean(0), S.std(0) + 1e-6
    S = (S - mu) / sd
    A = np.eye(len(ATTACK))[[ATTACK.index(a) for a in df.attack]]
    D = np.eye(len(DEFENCE))[[DEFENCE.index(d) for d in df.defence]]
    return np.c_[S, A, D].astype(np.float32), mu, sd


def train(X, y, w, seed, epochs=300):
    import torch
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 16), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=3e-2)
    Xt, yt, wt = torch.tensor(X), torch.tensor(y, dtype=torch.float32)[:, None], torch.tensor(w, dtype=torch.float32)[:, None]
    for _ in range(epochs):
        opt.zero_grad()
        loss = (wt * (net(Xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    return net


def predict(nets, X):
    import torch
    with torch.no_grad():
        for n in nets:
            n.eval()
        return np.mean([n(torch.tensor(X)).numpy()[:, 0] for n in nets], axis=0)


def spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def main(n_nets=5):
    import torch
    torch.set_num_threads(2)
    df, F = dataset()
    ms = sorted(df.moment.unique())
    print(f"{len(df)} (moment, plan pair) cells from {len(ms)} moments", flush=True)
    w = 1.0 / np.clip(df.y_se.to_numpy() ** 2, 0.05, None)
    w = (w / w.mean()).astype(np.float32)
    df["emulator"], df["table"] = np.nan, np.nan
    for m in ms:
        tr, te = (df.moment != m).to_numpy(), (df.moment == m).to_numpy()
        X, mu, sd = design(df[tr], F)
        Xte, _, _ = design(df[te], F, mu, sd)
        tab = df[tr].groupby(["attack", "defence"]).y.mean()
        t_tr = np.array([tab[(a, d)] for a, d in zip(df.attack[tr], df.defence[tr])], np.float32)
        t_te = np.array([tab.get((a, d), 0.0) for a, d in zip(df.attack[te], df.defence[te])], np.float32)
        nets = [train(X, df.y[tr].to_numpy(np.float32) - t_tr, w[tr], s) for s in range(n_nets)]   # the moment's own part
        df.loc[te, "emulator"] = t_te + predict(nets, Xte)
        df.loc[te, "table"] = t_te
    df["zero"] = 0.0
    df.round(4).to_csv(os.path.join(OUT, "emulator_cv.csv"), index=False)

    out = []
    for k in ("emulator", "table", "zero"):
        err = (df[k] - df.y).abs()
        rho, hit, near = [], [], []
        for m, g in df.groupby("moment"):
            rho.append(spearman(g[k], g.y) if g[k].std() > 0 else 0.0)
            gd = g[g.attack == "none"]                                    # the defence's choice vs an unchanged attack
            if len(gd) >= 3:
                pick = gd.loc[gd[k].idxmin()] if gd[k].std() > 0 else gd.iloc[0]
                best = gd.loc[gd.y.idxmin()]
                hit.append(pick.defence == best.defence)
                near.append(pick.y <= best.y + 1.96 * np.sqrt(pick.y_se ** 2 + best.y_se ** 2))
        out.append({"predictor": k, "mean_abs_error_pp": err.mean(), "rank_corr_in_moment": np.nanmean(rho),
                    "best_defence_hit": np.mean(hit), "best_defence_within_ci": np.mean(near)})
    sm = pd.DataFrame(out)
    # speed: one prediction for all 35 pairs of a moment vs GRF
    X, mu, sd = design(df, F)
    tab = df.groupby(["attack", "defence"]).y.mean()
    t_all = np.array([tab[(a, d)] for a, d in zip(df.attack, df.defence)], np.float32)
    nets = [train(X, df.y.to_numpy(np.float32) - t_all, w, s) for s in range(n_nets)]
    torch.save({"nets": [n.state_dict() for n in nets], "mu": mu.tolist(), "sd": sd.tolist(), "features": list(F.columns),
                "attack": ATTACK, "defence": DEFENCE, "metric": METRIC}, os.path.join(OUT, "emulator.pt"))
    t0 = time.perf_counter()
    for _ in range(100):
        predict(nets, X[:35])
    ms_per = (time.perf_counter() - t0) / 100 * 1000
    sm["ms_per_moment_35_pairs"] = [ms_per, np.nan, np.nan]
    sm.round(4).to_csv(os.path.join(OUT, "emulator_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nheld-out moments (leave-one-moment-out):\n" + sm.round(3).to_string(index=False))
    print(f"\nspeed: emulator {ms_per:.1f} ms for all 35 plan pairs of a moment; GRF about 35 x 6 seeds x 3.3 s (4 cores) = "
          f"{35 * 6 * 3.3 / 60:.0f} min")
    chart(df, sm)


def chart(df, sm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].errorbar(df.y, df.emulator, xerr=1.96 * df.y_se, fmt="o", ms=3, alpha=0.4, color="#e0483b", ecolor="#e8b4ae", label="emulator")
    ax[0].scatter(df.y, df.table, s=8, color="#2a78d6", alpha=0.5, label="table (pair average)")
    lim = [df.y.min() - 1, df.y.max() + 1]
    ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].set_xlabel("GRF: change in danger over time (pp, held-out moment)")
    ax[0].set_ylabel("predicted")
    ax[0].legend()
    ax[0].set_title("Held-out moments: emulator vs GRF")
    s = sm.set_index("predictor")
    x = np.arange(3)
    ax[1].bar(x - 0.2, s.rank_corr_in_moment, 0.4, label="rank corr. within a moment", color="#8a8a8a")
    ax[1].bar(x + 0.2, s.best_defence_within_ci, 0.4, label="best defence within GRF's 95% interval", color="#e0483b")
    ax[1].set_xticks(x, s.index)
    ax[1].legend()
    ax[1].set_title("Does it know THIS moment better than the table?")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "emulator_chart.png"), dpi=110)
    print("wrote outputs/tournament/emulator_cv.csv, emulator_summary.csv, emulator.pt, emulator_chart.png")


if __name__ == "__main__":
    main()
