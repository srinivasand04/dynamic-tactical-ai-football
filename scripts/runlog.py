"""Append one row per experiment to outputs/run_log.csv, so every change can be traced to its result.

    import runlog
    runlog.log({"script": "train_gnn", "holdout": "J03WOH", "real_model_m": 1.13, ...})

Columns are added as needed; older rows keep empty cells for columns they did not have.
Runs in parallel processes (tuning_ladder.py) take turns through a lock file, so no row is lost.
"""
import datetime
import os
import time

import pandas as pd

PATH = "outputs/run_log.csv"


def log(row, path=PATH):
    row = {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **row}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock = path + ".lock"
    for _ in range(600):                                   # wait up to a minute for another process to finish writing
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > 60:  # left behind by a crashed process
                    os.remove(lock)
            except OSError:
                pass
            time.sleep(0.1)
    else:
        fd = None
    try:
        new = pd.DataFrame([row])
        if os.path.exists(path):
            try:
                old = pd.read_csv(path)
                new = pd.concat([old, new], ignore_index=True)
            except Exception:
                pass
        new.to_csv(path, index=False)
    finally:
        if fd is not None:
            os.close(fd)
            os.remove(lock)
