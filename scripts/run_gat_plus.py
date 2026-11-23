"""GAT-family comparison (Nov 2026): GAT, GATv2, typed GAT, distance GAT, typed+distance GATv2 (radius and Delaunay graphs).

Runs train_gnn.py on the same 3 held-out real matches as compare_archs.py (J03WOH, J03WMX, J03WN1), seed 0,
several runs at once (each limited to a few CPU threads). Logs go to outputs/gat_plus/logs/.
Tags: gatplus_<name>_<fold>, so compare_archs.py and run_folds.py results are not touched.

Usage : python scripts/run_gat_plus.py [--procs 6] [--threads 2]
        python scripts/run_gat_plus.py --report          (table -> outputs/gat_plus/compare_gat_plus.csv)
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "gat_plus")
FOLDS = ["J03WOH", "J03WMX", "J03WN1"]
VARIANTS = [  # name, arch, graphs  (most important first)
    ("gatv2_td", "gatv2_td", "radius"),
    ("gat", "gat", "radius"),
    ("gat_typed", "gat_typed", "radius"),
    ("gatv2", "gatv2", "radius"),
    ("gat_dist", "gat_dist", "radius"),
    ("gatv2_td_delaunay", "gatv2_td", "delaunay"),
]


def run(procs, threads):
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    jobs = [(n, a, g, f) for n, a, g in VARIANTS for f in FOLDS]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    running, t0, done = [], time.time(), 0
    while jobs or running:
        while jobs and len(running) < procs:
            n, a, g, f = jobs.pop(0)
            log = open(os.path.join(OUT, "logs", f"{n}_{f}.log"), "w")
            p = subprocess.Popen([sys.executable, "scripts/train_gnn.py", "--holdout", f, "--arch", a, "--graphs", g, "--seed", "0",
                                  "--no-sims", "--tag", f"gatplus_{n}_{f}"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((p, n, f, log))
            print(f"started {n} {f}", flush=True)
        time.sleep(10)
        for item in running[:]:
            p, n, f, log = item
            if p.poll() is not None:
                log.close(); running.remove(item); done += 1
                print(f"finished {n} {f} (exit {p.returncode}) · {done}/{len(VARIANTS) * len(FOLDS)} · {(time.time() - t0) / 60:.1f} min",
                      flush=True)


def report():
    import pandas as pd
    log = pd.read_csv(os.path.join(ROOT, "outputs", "run_log.csv"))
    log = log[(log["script"] == "train_gnn") & log["tag"].astype(str).str.startswith("gatplus_")].copy()
    log["name"] = log["tag"].str.replace("gatplus_", "", regex=False).str.rsplit("_", n=1).str[0]
    log = log.sort_values("time").drop_duplicates(["name", "holdout"], keep="last")
    log["real_gain_%"] = 100 * (1 - log["real_player_model_m"] / log["real_player_keepvel_m"])
    log["ball_gain_%"] = 100 * (1 - log["real_ball_model_m"] / log["real_ball_keepvel_m"])
    t = log.groupby("name").agg(folds=("holdout", "count"), real_error_m=("real_player_model_m", "mean"),
                                real_gain=("real_gain_%", "mean"), ball_error_m=("real_ball_model_m", "mean"),
                                params=("params", "mean"), epochs=("epochs_run", "mean"), minutes=("minutes", "mean"))
    t = t.rename(columns={"real_gain": "real_gain_%"}).sort_values("real_error_m")
    print(t.round(3).to_string())
    print("\nper fold, real player error (m):")
    print(log.pivot(index="holdout", columns="name", values="real_player_model_m").round(3).to_string())
    ref = os.path.join(ROOT, "outputs", "compare_archs.csv")
    if os.path.exists(ref):
        print("\nreference (earlier run, compare_archs.csv):")
        print(pd.read_csv(ref)[["arch", "real_error_m", "real_gain_%", "ball_error_m", "params"]].round(3).to_string(index=False))
    os.makedirs(OUT, exist_ok=True)
    t.round(3).to_csv(os.path.join(OUT, "compare_gat_plus.csv"))
    print(f"\nsaved {os.path.relpath(os.path.join(OUT, 'compare_gat_plus.csv'), ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a.procs, a.threads)
