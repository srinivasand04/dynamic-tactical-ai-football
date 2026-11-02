"""Run the data pipeline for every match listed in settings.json, skipping what already exists.

Steps, in order (each one only for the matches that still need it):
  download   real DFL matches whose raw files are missing           (download_real.py, save_meta.py)
  convert    raw real data -> common format                          (convert_real.py)
  keystates  Key States of every match                               (extract_keystates.py)
  sequences  boundaries and sequences, all matches together          (find_sequences.py)
  describe   football descriptions of boundaries                     (describe_sequences.py)
  types      sequence types, real vs simulated                       (cluster_sequences.py)
  graphs     one graph per Key State                                 (build_graphs.py)

Because the sequence step uses ONE scale and threshold for all matches, adding a match re-runs
sequences, describe, types and graphs for everybody. The GNN training is not part of this (it is slow): see train_gnn.py.

A dataset with "pool": false in settings.json (for example pff) is not part of those steps. Run it by itself; it needs only
the Key States and graphs of its own matches, so nothing already reported changes:
         python scripts/run_pipeline.py --dataset pff

Usage :  python scripts\\run_pipeline.py
         python scripts\\run_pipeline.py --force            (redo everything)
         python scripts\\run_pipeline.py --only graphs      (just one or more steps)
"""
import argparse
import os
import subprocess
import sys

import settings

PY = sys.executable
STEPS = ["download", "convert", "keystates", "sequences", "describe", "types", "graphs"]


def run(*args):
    print("\n>>", " ".join(str(a) for a in args), flush=True)
    r = subprocess.run([PY, *map(str, args)])
    if r.returncode != 0:
        raise SystemExit(f"step failed: {' '.join(str(a) for a in args)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="redo every step for every match")
    ap.add_argument("--only", nargs="+", choices=STEPS, help="run only these steps")
    ap.add_argument("--dataset", default=None, help="one dataset of settings.json that is kept out of the pool (keystates + graphs only)")
    a = ap.parse_args()
    if a.dataset:
        have = settings.dataset_matches(a.dataset)
        print(f"{a.dataset}: {len(have)} matches with a common-format file")
        todo = [m for m in have if a.force or not os.path.exists(settings.base(m) + "_adaptive_ks.csv")]
        for m in todo:
            run("scripts/extract_keystates.py", "--csv", settings.csv(m))
        need = [m for m in have if a.force or todo or not os.path.exists(settings.graphs(m))]
        if need:
            run("scripts/build_graphs.py", "--csv", *[settings.csv(m) for m in have])
        print(f"\ndone: {len(todo)} matches got Key States, graphs built for {len(have) if need else 0}")
        return
    todo = a.only or STEPS
    ran = set()

    if "download" in todo:
        for m in settings.REAL:
            if settings.dataset(m) == "dfl" and not os.path.exists(f"data/real/raw/{m}_tracking.csv.gz"):
                run("scripts/download_real.py", m)
            if settings.dataset(m) == "dfl" and not os.path.exists(f"data/real/raw/{m}_meta.json") and os.path.exists(f"data/real/raw/{m}_tracking.csv.gz"):
                run("scripts/save_meta.py", m)
        ran.add("download")

    if "convert" in todo:
        for m in settings.REAL:
            if settings.dataset(m) == "dfl" and os.path.exists(f"data/real/raw/{m}_meta.json") and (a.force or not settings.exists(m)):
                run("scripts/convert_real.py", m)
        ran.add("convert")

    have = settings.available(settings.REAL + settings.SIM)
    print(f"\nmatches with a common-format file: {len(settings.available(settings.REAL))} real, {len(settings.available(settings.SIM))} simulated")

    if "keystates" in todo:
        for m in have:
            if a.force or not os.path.exists(settings.base(m) + "_adaptive_ks.csv"):
                run("scripts/extract_keystates.py", "--csv", settings.csv(m))
                ran.add("keystates")

    need_seq = a.force or "keystates" in ran or any(not os.path.exists(settings.base(m) + "_sequences.csv") for m in have)
    if "sequences" in todo and need_seq:
        run("scripts/find_sequences.py")
        ran.add("sequences")

    if "describe" in todo and ("sequences" in ran or a.force or any(not os.path.exists(settings.base(m) + "_sequence_desc.csv") for m in have)):
        for m in have:
            events = f"data/real/raw/{m}_events.csv.gz"
            extra = ["--events", events] if m in settings.REAL and os.path.exists(events) else []
            run("scripts/describe_sequences.py", "--csv", settings.csv(m), *extra)
        ran.add("describe")

    if "types" in todo and ("describe" in ran or a.force):
        run("scripts/cluster_sequences.py")
        ran.add("types")

    if "graphs" in todo and ("sequences" in ran or a.force or any(not os.path.exists(settings.graphs(m)) for m in have)):
        run("scripts/build_graphs.py")
        ran.add("graphs")

    print("\ndone. steps that ran:", ", ".join(s for s in STEPS if s in ran) or "none (everything was up to date)")


if __name__ == "__main__":
    main()
