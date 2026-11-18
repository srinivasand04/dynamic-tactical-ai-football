"""Weak spot 4, the fix to try: pre-train on the 20 extra matches, then fine-tune on the Bundesliga.

Mixing the 20 extra matches (12 World Cup, 8 A-League) into training made the movement model WORSE on every role, in 7 of 7
held-out matches (+0.07 m, movement_by_role.py), although the datasets move on the same clock and scale
(dataset_compare.csv). So the extra data is useful as a start, not as an equal partner:
    1. pre-train once on the 20 extra matches only                       -> outputs/tuning/pretrain_extra.pt
    2. for each of the 7 Bundesliga folds: start from those weights and train on the 6 other Bundesliga matches with a
       lower learning rate (fine-tuning)                                 -> outputs/tuning/preds/finetune_<fold>.npz
    3. score against the current model and the 3-seed ensemble with the ladder's rule (better on average, 95% interval
       excludes zero, better in at least 6 of 7 matches); also try 'ensemble of 3 seeds + fine-tuned'.
Trains only new files; the ladder's predictions and the current model are not changed. A stopped run resumes.

Output: outputs/tuning/finetune_ladder.csv
Usage : .venv-analysis\\Scripts\\python.exe scripts\\movement_finetune.py --procs 2       (8 trainings, about an hour or two)
        .venv-analysis\\Scripts\\python.exe scripts\\movement_finetune.py --report
"""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "tuning")
PREDS = os.path.join(OUT, "preds")
PRE = os.path.join(OUT, "pretrain_extra.pt")
EXTRA = ["pff:12", "skillcorner:8"]
FT_LR = 5e-4
BASE = ["ours__window_3____current", "ours_seed_1", "ours_seed_2"]


def train(args, log):
    env = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3")
    cmd = [sys.executable, "scripts/train_gnn.py", "--graphs", "delaunay", "--no-sims", *args]
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    return r.returncode


def finetune(fold):
    pred = os.path.join(PREDS, f"finetune_{fold}.npz")
    if os.path.exists(pred):
        return fold, "done before"
    t0 = time.time()
    rc = train(["--holdout", fold, "--init-model", PRE, "--lr", str(FT_LR), "--tag", f"finetune_{fold}", "--save-preds", pred],
               os.path.join(OUT, f"finetune_{fold}.log"))
    return fold, "FAILED (see log)" if rc else f"{(time.time() - t0) / 60:.1f} min"


def player_err(z_list):
    pred = np.mean([z["pred"] for z in z_list], axis=0)
    tgt, ok = z_list[0]["target"], z_list[0]["ok"].copy()
    ok[:, 22] = False
    return float(np.linalg.norm(pred - tgt, axis=2)[ok].mean())


def report():
    folds = settings.available(settings.REAL)
    L = lambda s, m: np.load(os.path.join(PREDS, f"{s}_{m}.npz"))
    have = [m for m in folds if os.path.exists(os.path.join(PREDS, f"finetune_{m}.npz"))]
    if not have:
        print("no fine-tuned predictions yet")
        return
    vers = {"current": lambda m: [L(BASE[0], m)], "3-seed ensemble (best)": lambda m: [L(s, m) for s in BASE],
            "+ 20 more matches mixed": lambda m: [L("ours___20_more_matches__12_world_cup__8_a_league", m)],
            "pre-train, then fine-tune": lambda m: [L("finetune", m)],
            "3 seeds + fine-tuned": lambda m: [L(s, m) for s in BASE] + [L("finetune", m)]}
    e = pd.DataFrame({v: {m: player_err(f(m)) for m in have} for v, f in vers.items()})
    rows = []
    for ref in ("current", "3-seed ensemble (best)"):
        for v in e:
            d = (e[v] - e[ref]).to_numpy()
            ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            rows.append({"compared_with": ref, "version": v, "error_m": e[v].mean(), "change_m": d.mean(), "ci95": ci,
                         "better_in": f"{int((d < 0).sum())}/{len(d)}",
                         "wins_rule": bool(d.mean() < 0 and d.mean() + ci < 0 and (d < 0).sum() >= len(d) - 1) if v != ref else ""})
    t = pd.DataFrame(rows)
    t.round(4).to_csv(os.path.join(OUT, "finetune_ladder.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"held-out matches scored: {len(have)}/{len(folds)}")
    print(t.round(4).to_string(index=False))
    print("wrote outputs/tuning/finetune_ladder.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if not a.report:
        if not os.path.exists(PRE):
            print("1/2 pre-training on the 20 extra matches ...", flush=True)
            if train(["--holdout", "none", "--only-extra", "--extra-train", *EXTRA, "--save-model", PRE, "--tag", "pretrain_extra"],
                     os.path.join(OUT, "pretrain_extra.log")):
                raise SystemExit("pre-training failed, see outputs/tuning/pretrain_extra.log")
        print("2/2 fine-tuning on the Bundesliga, one run per held-out match ...", flush=True)
        with Pool(a.procs) as pool:
            for fold, msg in pool.imap_unordered(finetune, settings.available(settings.REAL)):
                print(f"  {fold}: {msg}", flush=True)
    report()


if __name__ == "__main__":
    main()
