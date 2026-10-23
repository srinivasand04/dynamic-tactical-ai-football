"""Collect the result tables of several train_gnn.py runs into ONE comparison table.

Reads every  outputs/gnn_<graphs>_<heldout>.txt   (for example gnn_radius_J03WOH.txt, gnn_delaunay_J03WMX.txt)
and prints, per run and averaged per edge definition:
    real  : error of the model on the held-out real match, versus the two simple guesses
    sim   : mean error on the 4 simulated matches
Errors are in metres (lower is better). 'gain' = how much smaller the model's error is than the keep-velocity guess.

Usage :  python scripts\\summarise_gnn.py
"""
import glob
import os
import re
import pandas as pd

ROW = re.compile(r"^(J03W\w+|match_\d+)\s+(held-out real|simulated)\s*\|\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\|\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)")


def read_text(path):
    """PowerShell can save as UTF-16 or UTF-8 with a BOM; accept both."""
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


rows = []
for path in sorted(glob.glob("outputs/gnn_*_*.txt")):
    m = re.match(r"gnn_(radius|delaunay)_(\w+)\.txt", os.path.basename(path))
    if not m:
        continue
    graphs, holdout = m.groups()
    real, sims = None, []
    for line in read_text(path).splitlines():
        r = ROW.match(line.strip())
        if not r:
            continue
        vals = [float(x) for x in r.groups()[2:]]
        if r.group(2) == "held-out real":
            real = vals
        else:
            sims.append(vals)
    if real is None or len(sims) == 0:
        print(f"skipped {path}: no finished result table in it (did the run finish?)")
        continue
    sim = [sum(v[i] for v in sims) / len(sims) for i in range(6)]
    rows.append({"graphs": graphs, "held_out": holdout,
                 "real_model": real[0], "real_still": real[1], "real_keepvel": real[2],
                 "real_gain_%": 100 * (1 - real[0] / real[2]),
                 "sim_model": sim[0], "sim_keepvel": sim[2], "sim_gain_%": 100 * (1 - sim[0] / sim[2]),
                 "ball_real": real[3], "ball_sim": sim[3]})

if not rows:
    raise SystemExit("no result files found in outputs\\ (expected names like gnn_radius_J03WOH.txt)")

df = pd.DataFrame(rows)
print("\nPER RUN   (metres; gain = how much smaller the model's error is than the keep-velocity guess)\n")
print(df.round(2).to_string(index=False))
print("\nAVERAGE PER EDGE DEFINITION\n")
avg = df.drop(columns="held_out").groupby("graphs").mean().round(2)
avg.insert(0, "runs", df.groupby("graphs").size())
print(avg.to_string())