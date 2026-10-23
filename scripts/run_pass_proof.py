"""The firmer test of the pass edges: three networks, every real match held out in turn, several random starts.

    A  ours            the radius-graph network (team, opponent, ball edges)
    B  pass_raw        A + the pass edges of the ball-owner graph, with the real numbers on them
    C  pass_shuffle    A + the same pass edges, but the numbers are shuffled between teammates and Key States of a match
                       (same size and shape as B, no information: if B beats C, the gain comes from the pass information and
                       not just from having a bigger network)

Every run is one train_gnn.py call (a fold = one held-out real match, a seed = one random start). Runs that are already in
outputs/run_log.csv are skipped, so you can stop (Ctrl+C) and start again any time, and the existing 'ours' folds are reused.
The loop is seed by seed, so after the first seed you already have a complete (if thin) comparison.

When it finishes (or with --report) prove_pass_edges.py prints the paired comparison with confidence intervals.

Time: about 10 minutes per run; 7 folds x 3 seeds x 3 networks = 63 runs, minus what exists (about 9 hours in total).
      Start with   --seeds 0   (about 3 hours), add more seeds later.

Usage :  python scripts\\run_pass_proof.py --seeds 0
         python scripts\\run_pass_proof.py --seeds 0 1 2
         python scripts\\run_pass_proof.py --seeds 0 --folds J03WOH J03WMX      (a part)
         python scripts\\run_pass_proof.py --report                             (only the comparison)
"""
import argparse
import subprocess
import sys
import time

import pandas as pd

import settings

ARMS = ["ours", "pass_raw", "pass_shuffle"]


def finished():
    try:
        log = pd.read_csv("outputs/run_log.csv")
    except FileNotFoundError:
        return set()
    log = log[(log["script"] == "train_gnn") & log["holdout"].notna() & (log["holdout"] != "none") & log["real_player_model_m"].notna()]
    log = log[~log["tag"].astype(str).str.contains("test|smoke", case=False)]
    arch = log["arch"].fillna("ours") if "arch" in log.columns else pd.Series("ours", index=log.index)
    return set(zip(arch, log["holdout"], log["seed"].fillna(0).astype(int)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if not a.report:
        folds = a.folds or settings.available(settings.REAL)
        todo = [(arm, m, s) for s in a.seeds for m in folds for arm in a.arms]
        have = finished()
        todo = [t for t in todo if t not in have]
        print(f"{len(todo)} runs to do ({len(a.seeds) * len(folds) * len(a.arms) - len(todo)} already in the run log)\n", flush=True)
        t0 = time.time()
        for i, (arm, m, s) in enumerate(todo, 1):
            print(f"===== run {i}/{len(todo)}: {arm}, hold out {m}, seed {s} =====", flush=True)
            r = subprocess.run([sys.executable, "scripts/train_gnn.py", "--holdout", m, "--seed", str(s), "--arch", arm, "--tag", f"proof_{arm}_{m}_s{s}"])
            if r.returncode != 0:
                raise SystemExit(f"run failed: {arm} {m} seed {s}")
            done = (time.time() - t0) / 60
            print(f"----- {i}/{len(todo)} done, {done:.0f} min so far, about {done / i * (len(todo) - i):.0f} min to go\n", flush=True)
    subprocess.run([sys.executable, "scripts/prove_pass_edges.py"])


if __name__ == "__main__":
    main()
