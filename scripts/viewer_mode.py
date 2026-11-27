"""One command per viewer mode: prepares a file for the Three.js viewer (viewer/) and prints the link to open.

Modes (all on Fortuna Düsseldorf v Jahn Regensburg, J03WOH, except model_vs_model):
  normal          the real match, 57' to 60' (tracking data)
  model_vs_real   real players + our team models playing the same moment (ghosts), from 58:30
  model_vs_model  a full GRF match: Argentina model v France model
  whatif_top5     the real moment + the 5 best defensive changes from the 5,000-run batch (each a ghost layer)
  coach           the coach moves one player (--move NAME FORWARD LEFT, metres) and GRF plays it out next to 'no change'
                  (see the names first: --list-players)
  coach_goal_1/2  modes 7-8: two coach runs that score (coach_search.py picks, coach_run.py --out coach_goal_N)
  tidy            only rewrite the viewer's match list (the files below, with plain names)
  coach_run       the coach tells one player to RUN to a new spot and the team plays him there (scripts/coach_run.py):
                  real match + 'No change' + 'Coach run' ghosts (J03WOH 27:01, Klaus runs in behind; run coach_run.py first)

Earlier results are never overwritten: GRF runs that already exist are reused, and new runs write new files only.
Usage : .venv\\Scripts\\python.exe scripts\\viewer_mode.py normal
        .venv\\Scripts\\python.exe scripts\\viewer_mode.py coach --move Klaus 6 8
        (mode 6)                 ... viewer_mode.py coach_run
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


# The viewer's match list (viewer/data/index.json): only these files, with plain names (order = list order). The modes come
# first; the rest are loaded by the Demo steps (viewer/demo.js). Other files stay on disk, they are just not listed.
LABEL = {"mode1_normal.json": "1 · Real match: Düsseldorf v Regensburg, 57'-61'",
         "mode2_model_vs_real.json": "2 · Real vs our model: same moment, 58'-60'",
         "mode3_model_vs_model.json": "3 · Model vs model: Argentina v France (simulated)",
         "mode4_whatif_top5.json": "4 · What-if: 5 best defensive changes, 58'",
         "mode5_coach.json": "5 · Coach move: Klaus starts 6 m forward, 8 m left, 58'",
         "mode6_coach_run.json": "6 · Coach run: Klaus runs in behind, 27'",
         "mode7_coach_goal_1.json": "7 · Coach run that scores: Karbownik into the box (v Kaiserslautern)",
         "mode8_coach_goal_2.json": "8 · Coach run that scores: Sobottka in behind (v Rostock)",
         "J03WOH_whatif_3610.json": "Demo · What-if: early pass, 58'",
         "J03WOH_demo_tzdef.json": "Demo · What-if: early pass against TiZero's defence",
         "video_SNGS-116_filled_learned.json": "Demo · TV video to board (SoccerNet clip)",
         "10517_full.json": "Demo · Full match: Argentina v France, World Cup final",
         "J03WOH_full.json": "Demo · Full match: Düsseldorf v Regensburg (player of the match)"}


def label_index():
    """Keep only the listed files in the viewer's drop-down, with their plain names, in this order."""
    import json
    p = os.path.join(ROOT, "viewer", "data", "index.json")
    idx = {e["file"]: e for e in json.load(open(p, encoding="utf-8"))}
    keep = [{"file": f, "title": t} for f, t in LABEL.items()
            if f in idx or os.path.exists(os.path.join(ROOT, "viewer", "data", f))]
    json.dump(keep, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


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


def coach_run(src_name="coach_run", out_name="mode6_coach_run.json"):
    """Modes 6-8: a coach-run what-if (viewer/data/<src_name>.json, written by scripts/coach_run.py) as ghost layers on the real match."""
    import json
    import numpy as np
    src = os.path.join(ROOT, "viewer", "data", f"{src_name}.json")
    if not os.path.exists(src):
        run([PY, "scripts/coach_run.py"])
    d = json.load(open(src, encoding="utf-8"))
    proc = os.path.join(ROOT, "data", "real", "processed")
    starts = sorted(int(f[len(d["match"]) + 9:-4]) for f in os.listdir(proc) if f.startswith(f"{d['match']}_segment_") and f.endswith(".npz"))
    seg = (d["segment"].replace("_segment_", "_") if d.get("segment")                 # the segment the run came from (its shirt numbers)
           else f"{d['match']}_{max(s for s in starts if s <= d['start_s'] - 2)}")
    os.makedirs(os.path.join(ROOT, "outputs", "coach_run"), exist_ok=True)
    who = d["who"].split(" ")[-1]
    layers = []
    for key, lab in (("twin", "No change (model)"), ("run", f"Coach run: {who}, {d.get('label', 'run in behind')}")):
        p = np.array(d[key]["p"], float)
        b = np.array(d[key]["b"], float)
        p[p < -50], b[b < -50] = np.nan, np.nan
        t = d["start_s"] + np.arange(p.shape[1]) / d["fps"]
        f = os.path.join("outputs", "coach_run", f"grf_coachrun_{seg}_{src_name}_{key}.npz")
        np.savez(os.path.join(ROOT, f), grf=p, ball_grf=b[:, :2], time_s=t)
        layers += ["--layer", f"{lab}={f}"]
    export(out_name, "--match", d["match"], "--start-s", str(d["start_s"] - 3), "--minutes", "0.3", *layers)
    out = os.path.join(ROOT, "viewer", "data", out_name)              # follow the runner (the viewer reads "focus")
    v = json.load(open(out, encoding="utf-8"))
    ro, home = v["rosters"][0][1], d["slot"] < 11                                      # the viewer's slots: its own roster order
    v["focus"] = (ro[:11].index(d["shirts"][d["slot"]]) if home else 11 + ro[11:].index(d["shirts"][d["slot"]]))
    json.dump(v, open(out, "w", encoding="utf-8"), separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["normal", "model_vs_real", "model_vs_model", "whatif_top5", "coach", "coach_run", "coach_goal_1", "coach_goal_2", "tidy"])
    ap.add_argument("--move", nargs=3, metavar=("PLAYER", "FORWARD", "LEFT"), default=["Klaus", "6", "8"])
    ap.add_argument("--list-players", action="store_true", help="coach: show who is on the pitch at the moment, then stop")
    a = ap.parse_args()

    if a.mode == "tidy":
        label_index()
        print("viewer match list: " + ", ".join(LABEL.values()))
        return
    if a.mode == "normal":
        export("mode1_normal.json", "--match", "J03WOH", "--start-s", "3500", "--minutes", "3")
    elif a.mode == "model_vs_real":
        export("mode2_model_vs_real.json", *REAL, "--layer", f"Our team models (no change)={BASE}")
    elif a.mode == "model_vs_model":
        export("mode3_model_vs_model.json", "--grf-match", "outputs/team_league/Argentina_vs_France.npz")
    elif a.mode == "coach_run":
        coach_run()
    elif a.mode in ("coach_goal_1", "coach_goal_2"):
        coach_run(a.mode, f"mode{7 if a.mode.endswith('1') else 8}_{a.mode}.json")
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
