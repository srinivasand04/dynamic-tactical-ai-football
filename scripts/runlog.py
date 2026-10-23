"""Append one row per experiment to outputs/run_log.csv, so every change can be traced to its result.

    import runlog
    runlog.log({"script": "train_gnn", "holdout": "J03WOH", "real_model_m": 1.13, ...})

Columns are added as needed; older rows keep empty cells for columns they did not have.
"""
import datetime
import os

import pandas as pd

PATH = "outputs/run_log.csv"


def log(row, path=PATH):
    row = {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **row}
    new = pd.DataFrame([row])
    if os.path.exists(path):
        try:
            old = pd.read_csv(path)
            new = pd.concat([old, new], ignore_index=True)
        except Exception:
            pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new.to_csv(path, index=False)
