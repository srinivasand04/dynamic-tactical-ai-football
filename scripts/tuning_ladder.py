"""Phase 1: the model tuning ladder. Each rung changes one thing; every rung is scored on the 7 real matches it never saw.

MOVEMENT (train_gnn.py, Delaunay graphs, leave one match out; error in metres to the next Key State, lower is better)
  no graph (mlp)  ->  GCN  ->  GAT  ->  ours, no time (window 1)  ->  ours (window 3, the current model)  ->  window 5
  ->  no mirroring (ablation)  ->  twice as wide (dim 96)  ->  ensemble: ours with 3 seeds, predictions averaged
PASS (fit_pass_model.py data, leave one match out)
  choice     : chance -> nearest teammate -> linear score (current) -> small neural score (MLP) -> 5 MLPs averaged
  completion : pass length only -> logistic, all numbers (current) -> MLP -> 5 MLPs averaged   (AUC higher, Brier lower is better)

Every training run is a separate process (4 at a time); a run already done (its prediction file exists) is skipped, so the
ladder can be stopped and started again.

Output : outputs/tuning/movement_runs.csv, movement_ladder.csv, pass_ladder.csv, ladder_chart.png, preds/ (per run)
Needs  : .venv-analysis (torch, pandas, matplotlib)
Usage  : .venv-analysis\\Scripts\\python.exe scripts\\tuning_ladder.py                  (everything)
         .venv-analysis\\Scripts\\python.exe scripts\\tuning_ladder.py --part pass      (only the pass models, ~2 min)
         .venv-analysis\\Scripts\\python.exe scripts\\tuning_ladder.py --report         (tables and chart from finished runs)
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import settings

OUT = "outputs/tuning"
PREDS = f"{OUT}/preds"
FOLDS = settings.available(settings.REAL)
# rung name -> train_gnn.py options (all on Delaunay graphs, held-out real match only)
RUNGS = {
    "no graph (mlp)": ["--arch", "mlp"],
    "GCN": ["--arch", "gcn"],
    "GAT": ["--arch", "gat"],
    "ours, no time (window 1)": ["--window", "1"],
    "ours (window 3) = current": [],
    "ours, window 5": ["--window", "5"],
    "ours, no mirroring": ["--no-mirror"],
    "ours, dim 96": ["--dim", "96"],
    "ours seed 1": ["--seed", "1"],
    "ours seed 2": ["--seed", "2"],
}
ENSEMBLE = ["ours (window 3) = current", "ours seed 1", "ours seed 2"]
SHOW = list(RUNGS)[:8] + ["ensemble of 3 seeds"]


def slug(rung):
    return "".join(c if c.isalnum() else "_" for c in rung).strip("_").lower()


def run_one(job):
    rung, fold = job
    pred = f"{PREDS}/{slug(rung)}_{fold}.npz"
    if os.path.exists(pred):
        return rung, fold, "done before"
    t0 = time.time()
    cmd = [sys.executable, "scripts/train_gnn.py", "--holdout", fold, "--graphs", "delaunay", "--no-sims",
           "--tag", f"ladder_{slug(rung)}_{fold}", "--save-preds", pred, *RUNGS[rung]]
    env = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return rung, fold, "FAILED: " + r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "FAILED"
    return rung, fold, f"{(time.time() - t0) / 60:.1f} min"


def errors(pred, target, ok):
    e = np.linalg.norm(pred - target, axis=-1)
    ply, ball = ok.copy(), ok.copy()
    ply[:, 22], ball[:, :22] = False, False
    return float(e[ply].mean()), float(e[ball].mean())


def movement_tables():
    log = pd.read_csv("outputs/run_log.csv")
    rows = []
    for fold in FOLDS:
        kv = None
        for rung in RUNGS:
            p = f"{PREDS}/{slug(rung)}_{fold}.npz"
            if not os.path.exists(p):
                continue
            z = np.load(p)
            ply, ball = errors(z["pred"], z["target"], z["ok"])
            hit = log[log["tag"] == f"ladder_{slug(rung)}_{fold}"].tail(1)
            if len(hit):
                kv = (float(hit["real_player_keepvel_m"].iloc[0]), float(hit["real_ball_keepvel_m"].iloc[0]))
            rows.append({"rung": rung, "fold": fold, "player_m": ply, "ball_m": ball,
                         "epochs": float(hit["epochs_run"].iloc[0]) if len(hit) else np.nan,
                         "params": float(hit["params"].iloc[0]) if len(hit) and "params" in hit else np.nan})
        ens = [np.load(f"{PREDS}/{slug(r)}_{fold}.npz") for r in ENSEMBLE if os.path.exists(f"{PREDS}/{slug(r)}_{fold}.npz")]
        if len(ens) == len(ENSEMBLE):
            ply, ball = errors(np.mean([z["pred"] for z in ens], axis=0), ens[0]["target"], ens[0]["ok"])
            rows.append({"rung": "ensemble of 3 seeds", "fold": fold, "player_m": ply, "ball_m": ball})
        if kv:
            rows.append({"rung": "keep velocity (baseline)", "fold": fold, "player_m": kv[0], "ball_m": kv[1]})
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.round(4).to_csv(f"{OUT}/movement_runs.csv", index=False)
    if df.empty:
        return df, df
    order = ["keep velocity (baseline)"] + SHOW + ["ours seed 1", "ours seed 2"]
    cur = df[df["rung"] == "ours (window 3) = current"].set_index("fold")["player_m"]
    out = []
    for rung in [r for r in order if r in set(df["rung"])]:
        g = df[df["rung"] == rung].set_index("fold")
        common = g.index.intersection(cur.index)
        d = (g.loc[common, "player_m"] - cur.loc[common]).to_numpy()
        out.append({"rung": rung, "folds": len(g), "player_m": g["player_m"].mean(), "ball_m": g["ball_m"].mean(),
                    "vs_current_m": d.mean() if len(d) else np.nan,
                    "vs_current_ci95": 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan,
                    "better_in_folds": f"{int((d < 0).sum())}/{len(d)}" if len(d) and rung != "ours (window 3) = current" else "",
                    "params": g["params"].mean() if "params" in g else np.nan})
    lad = pd.DataFrame(out)
    lad.round(4).to_csv(f"{OUT}/movement_ladder.csv", index=False)
    return df, lad


# ---- pass models ----------------------------------------------------------------------------------------------------------
def pass_ladder(seeds=5):
    import torch
    import fit_pass_model as F

    torch.set_num_threads(4)
    matches = [m for m in FOLDS if os.path.exists(settings.base(m) + "_passfeat.npz")]
    data = F.load(matches)
    cols, ccols = F.FULL, F.COMPLETION_COLS

    def norm_stats(X, M):
        v = X[M] if M is not None else X
        return v.mean(0), v.std(0) + 1e-6

    def fit_choice_mlp(I, M, T, seed, epochs=300):
        torch.manual_seed(seed)
        mu, sd = norm_stats(I, M)
        X = torch.tensor((I - mu) / sd, dtype=torch.float32)
        Mt, Tt = torch.tensor(M), torch.tensor(T)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[-1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
        opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-3)
        for _ in range(epochs):
            opt.zero_grad()
            s = net(X).squeeze(-1).masked_fill(~Mt, -1e9)
            torch.nn.functional.cross_entropy(s, Tt).backward()
            opt.step()
        return lambda Ite, Mte: np.where(Mte, net(torch.tensor((Ite - mu) / sd, dtype=torch.float32)).squeeze(-1).detach().numpy(), -1e9)

    def fit_comp_mlp(X, y, seed, epochs=400):
        torch.manual_seed(seed)
        mu, sd = norm_stats(X, None)
        Xt, yt = torch.tensor((X - mu) / sd, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
        opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-3)
        for _ in range(epochs):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(net(Xt).squeeze(-1), yt).backward()
            opt.step()
        return lambda Xte: torch.sigmoid(net(torch.tensor((Xte - mu) / sd, dtype=torch.float32)).squeeze(-1)).detach().numpy()

    def softmax_ll(s, T):
        return F.choice_metrics(s, T)

    rows = []
    for m in matches:
        tr = [x for x in matches if x != m]
        I, M, T = F.cat(data, tr, "I", True), F.cat(data, tr, "mask", True), F.cat(data, tr, "target", True)
        Ite, Mte, Tte = (data[m][k][data[m]["complete"]] for k in ("I", "mask", "target"))
        n = Mte.sum(1)
        rows.append({"part": "choice", "rung": "chance", "fold": m, "top1_%": 100 * float((1 / n).mean())})
        rows.append({"part": "choice", "rung": "nearest teammate", "fold": m, **softmax_ll(np.where(Mte, -Ite[..., F.COL["dist"]], -1e9), Tte)})
        w = F.fit_choice(I, M, T, cols)
        rows.append({"part": "choice", "rung": "linear (current)", "fold": m, **softmax_ll(F.choice_scores(w, Ite, Mte, cols), Tte)})
        Ic, Ic_te = I[..., F.idx(cols)], Ite[..., F.idx(cols)]
        scores = [fit_choice_mlp(Ic, M, T, s)(Ic_te, Mte) for s in range(seeds)]
        rows.append({"part": "choice", "rung": "MLP", "fold": m, **softmax_ll(scores[0], Tte)})
        lp = [s - np.log(np.exp(s - s.max(1, keepdims=True)).sum(1, keepdims=True)) - s.max(1, keepdims=True) for s in scores]
        ens = np.where(Mte, np.log(np.mean([np.exp(x) for x in lp], axis=0) + 1e-12), -1e9)
        rows.append({"part": "choice", "rung": f"{seeds} MLPs averaged", "fold": m, **softmax_ll(ens, Tte)})

        Xtr_all = F.chosen(F.cat(data, tr, "I"), F.cat(data, tr, "target"))
        ytr = F.cat(data, tr, "complete").astype(float)
        te = F.chosen(data[m]["I"], data[m]["target"])
        y = data[m]["complete"].astype(int)
        cc = F.idx(ccols)
        base = F.fit_logistic(Xtr_all[:, [F.COL["dist"], F.COL["dist2"]]], ytr)
        pb = F.sigmoid(te[:, [F.COL["dist"], F.COL["dist2"]]] @ base[0] + base[1])
        lg = F.fit_logistic(Xtr_all[:, cc], ytr)
        pl = F.sigmoid(te[:, cc] @ lg[0] + lg[1])
        pm = [fit_comp_mlp(Xtr_all[:, cc], ytr, s)(te[:, cc]) for s in range(seeds)]
        for rung, p in (("pass length only", pb), ("logistic (current)", pl), ("MLP", pm[0]), (f"{seeds} MLPs averaged", np.mean(pm, axis=0))):
            rows.append({"part": "completion", "rung": rung, "fold": m, "auc": F.auc(y, p), "brier": float(((p - y) ** 2).mean())})
        print(f"  pass models, held out {m}: choice linear {rows[-8]['top1_%']:.1f}%  MLP ens {rows[-5]['top1_%']:.1f}%  |"
              f"  AUC logistic {rows[-3]['auc']:.3f}  MLP ens {rows[-1]['auc']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.round(4).to_csv(f"{OUT}/pass_runs.csv", index=False)
    return pass_tables(df)


def pass_tables(df):
    out = []
    for part, ref, key in (("choice", "linear (current)", "top1_%"), ("completion", "logistic (current)", "auc")):
        g = df[df["part"] == part]
        refv = g[g["rung"] == ref].set_index("fold")[key]
        for rung in dict.fromkeys(g["rung"]):
            r = g[g["rung"] == rung].set_index("fold")
            d = (r[key] - refv).to_numpy()
            row = {"part": part, "rung": rung, "folds": len(r), key: r[key].mean()}
            for k in ("top3_%", "log_loss", "brier"):
                if k in r and r[k].notna().any():
                    row[k] = r[k].mean()
            row["vs_current"] = d.mean()
            row["vs_current_ci95"] = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            row["better_in_folds"] = f"{int((d > 0).sum())}/{len(d)}" if rung != ref else ""
            out.append(row)
    lad = pd.DataFrame(out)
    lad.round(4).to_csv(f"{OUT}/pass_ladder.csv", index=False)
    return lad


def chart(mov, pas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    if len(mov):
        m = mov[mov["rung"] != "ours seed 1"][lambda d: d["rung"] != "ours seed 2"]
        y = np.arange(len(m))
        col = ["#b4b2a9" if r.startswith("keep") else "#e0473c" if "ensemble" in r else "#2a78d6" if "current" in r else "#8fb3e0" for r in m["rung"]]
        ax[0].barh(y, m["player_m"], color=col, xerr=m["vs_current_ci95"].fillna(0), capsize=3)
        for yy, v in zip(y, m["player_m"]):
            ax[0].text(v + 0.01, yy, f"{v:.3f}", va="center", fontsize=9)
        ax[0].set_yticks(y, m["rung"])
        ax[0].invert_yaxis()
        ax[0].set_xlim(0.9, max(1.6, m["player_m"].max() + 0.1))
        ax[0].set_title("Movement: player error to the next Key State (m)\n7 held-out matches; lines = 95% interval vs current", fontsize=10)
    for k, (part, key, title, lo) in enumerate((("choice", "top1_%", "Pass choice: real receiver ranked first (%)", 0),
                                                ("completion", "auc", "Pass completion: AUC (1 = perfect, 0.5 = coin toss)", 0.5))):
        a = ax[k + 1]
        g = pas[pas["part"] == part]
        y = np.arange(len(g))
        col = ["#b4b2a9" if i < (2 if part == "choice" else 1) else "#2a78d6" if "current" in r else "#e0473c" if "averaged" in r else "#8fb3e0"
               for i, r in enumerate(g["rung"])]
        a.barh(y, g[key], color=col)
        for yy, v in zip(y, g[key]):
            a.text(v + (0.5 if key == "top1_%" else 0.004), yy, f"{v:.1f}" if key == "top1_%" else f"{v:.3f}", va="center", fontsize=9)
        a.set_yticks(y, g["rung"])
        a.invert_yaxis()
        a.set_xlim(lo, (g[key].max() * 1.15) if key == "top1_%" else 0.95)
        a.set_title(title + "\nleave one match out, 7 matches", fontsize=10)
    fig.suptitle("Phase 1, model tuning ladder: each rung changes one thing (grey = baseline, blue = current model, red = ensemble)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/ladder_chart.png", dpi=120)
    print(f"wrote {OUT}/ladder_chart.png")


def report():
    _, mov = movement_tables()
    pas = pass_tables(pd.read_csv(f"{OUT}/pass_runs.csv")) if os.path.exists(f"{OUT}/pass_runs.csv") else pd.DataFrame()
    if len(mov):
        print("\nMOVEMENT LADDER (mean over held-out matches; vs_current < 0 = better than the current model)\n")
        print(mov.round(3).to_string(index=False))
    if len(pas):
        print("\nPASS LADDER (vs_current > 0 = better than the current model)\n")
        print(pas.round(3).to_string(index=False))
    if len(mov) or len(pas):
        chart(mov, pas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["all", "movement", "pass"], default="all")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.makedirs(PREDS, exist_ok=True)
    if a.report:
        report()
        return
    if a.part in ("all", "pass"):
        print("PASS MODELS", flush=True)
        pass_ladder()
    if a.part in ("all", "movement"):
        jobs = [(r, f) for r in RUNGS for f in FOLDS]
        print(f"\nMOVEMENT: {len(jobs)} training runs ({len(RUNGS)} rungs x {len(FOLDS)} held-out matches), {a.procs} at a time", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(a.procs) as ex:
            for k, fut in enumerate(as_completed([ex.submit(run_one, j) for j in jobs])):
                rung, fold, msg = fut.result()
                print(f"  {k + 1:2d}/{len(jobs)}  {rung:28s} {fold}  {msg}   ({(time.time() - t0) / 60:.0f} min)", flush=True)
    report()


if __name__ == "__main__":
    main()
