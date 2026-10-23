"""Hold each real match out in turn: train the GNN on the others and test on the one it never saw.

For every real match M in settings.json this runs
    python scripts\\train_gnn.py --holdout M --tag fold_M --save-embeddings
so you get, per fold, the error on the unseen real match and on every simulated match, plus embeddings made by a model
that never saw M (used by novelty_holdout.py and learned_distance.py for a clean, unbiased comparison).
Every fold is also appended to outputs/run_log.csv, and the averages are printed at the end.

Usage :  python scripts\\run_folds.py
         python scripts\\run_folds.py --only J03WOH J03WMX      (just these folds)
         python scripts\\run_folds.py --seed 1                  (another random start, to see how much results move)
"""
import argparse
import subprocess
import sys

import pandas as pd

import settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", choices=["ours", "gcn", "gat", "mlp", "pass_score", "pass_raw", "pass_shuffle"], default="ours", help="graph layer to use (everything except ours does not save embeddings; pass_* need build_pass_edges.py)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="anything else to pass to train_gnn.py")
    a = ap.parse_args()
    folds = a.only or settings.available(settings.REAL)
    for m in folds:
        print(f"\n===== fold: hold out {m} =====", flush=True)
        tag = f"fold_{m}" if a.arch == "ours" else f"{a.arch}_{m}"
        extra = ["--save-embeddings"] if a.arch == "ours" else []
        r = subprocess.run([sys.executable, "scripts/train_gnn.py", "--holdout", m, "--tag", tag, "--seed", str(a.seed), "--arch", a.arch, *extra, *a.extra])
        if r.returncode != 0:
            raise SystemExit(f"fold {m} failed")
    log = pd.read_csv("outputs/run_log.csv")
    prefix = "fold_" if a.arch == "ours" else f"{a.arch}_"
    arch_col = log["arch"].fillna("ours") if "arch" in log.columns else "ours"
    mine = log[log["tag"].astype(str).str.startswith(prefix) & (log["seed"] == a.seed) & (arch_col == a.arch)].drop_duplicates("holdout", keep="last")
    cols = ["holdout", "epochs_run", "real_player_model_m", "real_player_keepvel_m", "sim_player_model_m", "sim_player_keepvel_m", "real_ball_model_m", "real_ball_keepvel_m", "sim_ball_model_m", "sim_ball_keepvel_m", "minutes"]
    t = mine[cols].copy()
    t["real_gain_%"] = 100 * (1 - t["real_player_model_m"] / t["real_player_keepvel_m"])
    t["sim_gain_%"] = 100 * (1 - t["sim_player_model_m"] / t["sim_player_keepvel_m"])
    print("\nALL FOLDS (metres; gain = how much smaller the model's error is than the keep-velocity guess)\n")
    print(t.round(2).to_string(index=False))
    print("\nMEAN OVER FOLDS (spread = standard deviation)\n")
    num = t.drop(columns="holdout")
    print(pd.DataFrame({"mean": num.mean(), "spread": num.std()}).round(2).to_string())


if __name__ == "__main__":
    main()
