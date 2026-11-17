"""One command per viewer mode: prepares a file for the Three.js viewer (viewer/) and prints the link to open.

Modes (all on Fortuna Düsseldorf v Jahn Regensburg, J03WOH, except model_vs_model):
  normal          the real match, 57' to 60' (tracking data)
  model_vs_real   real players + our team models playing the same moment (ghosts), from 58:30
  model_vs_model  a full GRF match: Argentina model v France model
  whatif_top5     the real moment + the 5 best defensive changes from the 5,000-run batch (each a ghost layer)
  coach           the coach moves one player (--move NAME FORWARD LEFT, metres) and GRF plays it out next to 'no change'
                  (see the names first: --list-players)

Earlier results are never overwritten: GRF runs that already exist are reused, and new runs write new files only.
Usage : .venv\\Scripts\\python.exe scripts\\viewer_mode.py normal
        .venv\\Scripts\\python.exe scripts\\viewer_mode.py coach --move Klaus 6 8
then  : .venv\\Scripts\\python.exe -m http.server 8765 --directory viewer      and open the printed link
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
WI = "outputs/whatif"
BASE = f"{WI}/grf_whatif_J03WOH_3500_baseline.npz"                     # 'no change' run from 3610 s (Phase 1, kept as it is)
TOP5 = [("double_team", "2 nearest close the carrier"), ("team_press", "whole team 3 m closer"), ("press", "3 nearest 3 m closer"),
        ("man_marking", "each defender marks his attacker"), ("keeper_off_line", "keeper 4 m off his line")]
REAL = ["--match", "J03WOH", "--start-s", "3607", "--minutes", "0.4"]    # the real moment: 3 s before the what-if start (3610 s), 24 s


def run(args):
    print(">", " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd=ROOT)


LABEL = {"mode1_normal.json": "1. Normal match", "mode2_model_vs_real.json": "2. Model vs real", "mode3_model_vs_model.json": "3. Model vs model",
         "mode4_whatif_top5.json": "4. What-if top 5", "mode5_coach.json": "5. Coach move"}


def label_index():
    """Put the mode name in front of each mode file's title in the viewer's drop-down (and list the modes first)."""
    import json
    p = os.path.join(ROOT, "viewer", "data", "index.json")
    idx = json.load(open(p, encoding="utf-8"))
    for e in idx:
        lab = LABEL.get(e["file"])
        if lab and not e["title"].startswith(lab):
            e["title"] = f"{lab}  |  {e['title']}"
    idx.sort(key=lambda e: (e["file"] not in LABEL, e["file"] if e["file"] in LABEL else ""))
    json.dump(idx, open(p, "w", encoding="utf-8"), indent=1)


def export(out, *args):
    run([PY, "scripts/export_viewer.py", *args, "--out", out])
    label_index()
    print(f"\nOpen: http://localhost:8765/?file={out}")


def whatif(extra):
    """Run what-if videos without touching files that already exist (they are restored if GRF rewrote them)."""
    keep = tempfile.mkdtemp()
    before = {f: os.path.getmtime(os.path.join(ROOT, WI, f)) for f in os.listdir(os.path.join(ROOT, WI))}
    for f in before:
        shutil.copy2(os.path.join(ROOT, WI, f), keep)
    try:
        run([PY, "scripts/whatif.py", "--video-only", *extra])
    finally:
        for f, mt in before.items():
            p = os.path.join(ROOT, WI, f)
            if not os.path.exists(p) or os.path.getmtime(p) != mt:
                shutil.copy2(os.path.join(keep, f), p)
        shutil.rmtree(keep, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["normal", "model_vs_real", "model_vs_model", "whatif_top5", "coach"])
    ap.add_argument("--move", nargs=3, metavar=("PLAYER", "FORWARD", "LEFT"), default=["Klaus", "6", "8"])
    ap.add_argument("--list-players", action="store_true", help="coach: show who is on the pitch at the moment, then stop")
    a = ap.parse_args()

    if a.mode == "normal":
        export("mode1_normal.json", "--match", "J03WOH", "--start-s", "3500", "--minutes", "3")
    elif a.mode == "model_vs_real":
        export("mode2_model_vs_real.json", *REAL, "--layer", f"Our team models (no change)={BASE}")
    elif a.mode == "model_vs_model":
        export("mode3_model_vs_model.json", "--grf-match", "outputs/team_league/Argentina_vs_France.npz")
    elif a.mode == "whatif_top5":
        need = [s for s, _ in TOP5 if not os.path.exists(os.path.join(ROOT, WI, f"grf_whatif_J03WOH_3500_{s}.npz"))]
        if need:
            print(f"playing {len(need)} what-if change(s) in GRF (a few minutes)...", flush=True)
            whatif(["--scenarios", *need])
        layers = [f"No change={BASE}"] + [f"#{i + 1} {t}={WI}/grf_whatif_J03WOH_3500_{s}.npz" for i, (s, t) in enumerate(TOP5)]
        export("mode4_whatif_top5.json", *REAL, *sum([["--layer", l] for l in layers], []))
    else:
        if a.list_players:
            run([PY, "scripts/whatif.py", "--list-players"])
            return
        name, fwd, left = a.move
        tag = name.split(":")[-1].split(" ")[-1]
        hits = [f for f in os.listdir(os.path.join(ROOT, WI)) if f.startswith("grf_whatif_J03WOH_3500_3610_move_") and f.endswith(f"_{fwd}_{left}.npz")
                and (tag.lower() in f.lower() or ":" in name)]
        if not hits:
            print(f"coach move: {name} {fwd} m forward, {left} m left - GRF plays it out (a minute or two)...", flush=True)
            whatif(["--move", name, fwd, left])
            hits = [f for f in os.listdir(os.path.join(ROOT, WI)) if f.startswith("grf_whatif_J03WOH_3500_3610_move_") and f.endswith(f"_{fwd}_{left}.npz")]
        mv = sorted(hits, key=lambda f: os.path.getmtime(os.path.join(ROOT, WI, f)))[-1]
        export("mode5_coach.json", *REAL, "--layer", f"No change={BASE}",
               "--layer", f"Coach: {name} {fwd} m forward, {left} m left={WI}/{mv}")


if __name__ == "__main__":
    main()
