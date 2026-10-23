"""One table that compares the three graph layers: ours (edge-type aware), plain GCN, and GAT (attention).

Reads outputs/run_log.csv: every fold run of train_gnn.py (made by run_folds.py --arch ours|gcn|gat) with a held-out real
match. For every architecture it averages over the folds that all three have, so the comparison is on the same matches.

Columns
    real_error_m   error of the network on the real match it never saw (lower is better)
    real_gain_%    how much smaller that error is than the 'keep your velocity' guess
    ball_error_m   the same for the ball
    sim_gain_%     gain on the simulated games (all simulated games, never seen in training)
    params         number of weights, to be fair about size
    minutes        training time per fold

Usage :  python scripts\\compare_archs.py
"""
import pandas as pd

import runlog  # noqa: F401  (keeps the import path consistent with the other scripts)

log = pd.read_csv("outputs/run_log.csv")
log = log[log["script"] == "train_gnn"].copy()
log["arch"] = log["arch"].fillna("ours") if "arch" in log.columns else "ours"
log = log[log["holdout"].notna() & (log["holdout"] != "none")]
log = log[~log["tag"].astype(str).str.contains("test|smoke", case=False)]
log["fold_tag"] = log["tag"].astype(str)
log = log.sort_values("time").drop_duplicates(["arch", "holdout", "seed"], keep="last")

folds_per_arch = log.groupby("arch")["holdout"].apply(set)
common = set.intersection(*folds_per_arch.tolist()) if len(folds_per_arch) else set()
print(f"architectures found: {', '.join(folds_per_arch.index)}; folds they all have: {', '.join(sorted(common)) or 'none yet'}\n")
if not common:
    raise SystemExit("run run_folds.py for each architecture first (for example --only J03WOH J03WMX J03WN1)")

d = log[log["holdout"].isin(common)].copy()
d["real_gain_%"] = 100 * (1 - d["real_player_model_m"] / d["real_player_keepvel_m"])
d["sim_gain_%"] = 100 * (1 - d["sim_player_model_m"] / d["sim_player_keepvel_m"])
d["ball_gain_%"] = 100 * (1 - d["real_ball_model_m"] / d["real_ball_keepvel_m"])
t = d.groupby("arch").agg(folds=("holdout", "count"), real_error_m=("real_player_model_m", "mean"), real_gain=("real_gain_%", "mean"),
                          ball_error_m=("real_ball_model_m", "mean"), ball_gain=("ball_gain_%", "mean"), sim_gain=("sim_gain_%", "mean"),
                          params=("params", "mean"), epochs=("epochs_run", "mean"), minutes=("minutes", "mean"))
t = t.rename(columns={"real_gain": "real_gain_%", "ball_gain": "ball_gain_%", "sim_gain": "sim_gain_%"})
print(t.round(2).to_string())

spread = d.groupby("arch")["real_gain_%"].std().round(2)
print("\nspread of the real gain over folds (standard deviation, %):", spread.to_dict())
print("\nper fold, real error in metres:")
print(d.pivot(index="holdout", columns="arch", values="real_player_model_m").round(3).to_string())
t.round(3).to_csv("outputs/compare_archs.csv")
print("\nsaved outputs/compare_archs.csv")
