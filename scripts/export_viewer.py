"""Export a real match (or a stretch of it) for the Three.js viewer in viewer/ (MVP 1 of the professor's replay / what-if plan).

The viewer only draws what is in the file, so a real match is shown exactly as it was recorded: 22 players and the ball at
every frame, with shirt numbers, names, the match clock and the score. Other runs of the same stretch can ride along as LAYERS
(GRF replays now, what-if simulations later); the viewer draws them as ghost players next to the real ones.

Input : the common-format file of a match in settings.json (any dataset: dfl, pff, skillcorner)
        optional layers: .npz files with grf (22, T, 2), ball_grf (T, 2) and time_s (T,)  (grf_match.py, grf_pass.py, ...)
Output: viewer/data/<name>.json  and the list of available files viewer/data/index.json

File layout (positions in decimetres as integers, so a full match stays small):
    t        seconds of each frame (tracking clock)
    px, py   22 values per frame: slots 0-10 home, 11-21 away (keeper first); -1 = no player in that slot
    bx,by,bz ball per frame
    live     1 = ball in play, per frame;  sh, sa = score
    rosters  [[first frame, [22 shirt numbers]], ...]  (a new entry when a substitution changes a slot)
    names    {"home": {shirt: name}, "away": {...}}
    clock    [[tracking time where a period starts, match clock at that moment], ...]
    layers   [{name, t, px, py, bx, by}]

Usage :  python scripts\\export_viewer.py --match J03WOH --start-s 3500 --minutes 3 --layer "GRF with passes=outputs\\grf_pass_J03WOH_3500.npz"
         python scripts\\export_viewer.py --match J03WOH --every 2            (the whole match at 5 frames/s, about 8 MB)
"""
import argparse
import json
import re
import os

import numpy as np
import pandas as pd

import settings

DM = 10.0                                             # metres -> decimetres


def team_names(m):
    ds = settings.dataset(m)
    reg = settings.registry(ds)
    if reg is None and os.path.exists("data/real/matches.csv"):
        reg = pd.read_csv("data/real/matches.csv", dtype={"match_id": str}, encoding="utf-8")
    if reg is not None and m in set(reg["match_id"]):
        r = reg[reg["match_id"] == m].iloc[0]
        return str(r["home"]), str(r["away"]), str(r.get("score", ""))
    return "Home", "Away", ""


def player_names(m, d):
    """shirt -> name per side, from the DFL player list when there is one (other datasets: shirt numbers only)."""
    out = {"home": {}, "away": {}}
    pfile = f"data/real/raw/{m}_players.csv"
    if os.path.exists(pfile):
        p = pd.read_csv(pfile, encoding="utf-8").set_index("player_id")["name"].to_dict()
        for r in d.drop_duplicates("dfl_id").itertuples():
            if r.dfl_id in p:
                out[r.team][str(int(r.jersey))] = p[r.dfl_id]
    return out


def ints(a):
    """Decimetres as a flat list of integers (frame by frame, 22 values per frame for players); -1 = missing."""
    return np.where(np.isnan(a), -1, np.round(a * DM)).astype(int).ravel().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--start-s", type=float, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--every", type=int, default=1, help="keep every n-th frame (2 = 5 frames/s; the viewer interpolates)")
    ap.add_argument("--layer", action="append", default=[], help='"name=path.npz" of a run of the same stretch (repeatable)')
    ap.add_argument("--out", default=None, help="file name in viewer/data (default: built from the match and stretch)")
    a = ap.parse_args()

    cols = ["frame", "time_s", "team", "player_id", "x_m", "y_m", "ball_x_m", "ball_y_m", "ball_z", "game_mode",
            "score_home", "score_away", "period", "dfl_id", "jersey"]
    d = pd.read_csv(settings.csv(a.match), usecols=lambda c: c in cols)
    if a.start_s is not None:
        end = a.start_s + (a.minutes * 60 if a.minutes else 1e9)
        d = d[(d["time_s"] >= a.start_s) & (d["time_s"] < end)]
    elif a.minutes:
        d = d[d["time_s"] < d["time_s"].min() + a.minutes * 60]
    frames = np.sort(d["frame"].unique())[:: a.every]
    d = d[d["frame"].isin(frames)]
    F = len(frames)
    fi = np.searchsorted(frames, d["frame"].to_numpy())
    slot = np.where(d["team"].to_numpy() == "home", 0, 11) + d["player_id"].to_numpy()
    keep = slot < 22
    px, py, jer = np.full((F, 22), np.nan), np.full((F, 22), np.nan), np.full((F, 22), -1)
    px[fi[keep], slot[keep]] = d["x_m"].to_numpy()[keep]
    py[fi[keep], slot[keep]] = d["y_m"].to_numpy()[keep]
    jer[fi[keep], slot[keep]] = d["jersey"].to_numpy()[keep]
    f = d.drop_duplicates("frame").set_index("frame").loc[frames]

    rosters, last = [], None                                   # shirt numbers per slot, only when they change
    for i in range(F):
        row = jer[i].astype(int).tolist()
        if row != last:
            rosters.append([i, row])
            last = row
    tt = f["time_s"].to_numpy()
    starts = pd.read_csv(settings.csv(a.match), usecols=["time_s", "period"]).groupby("period")["time_s"].min()
    at = {1: 0, 2: 45, 3: 90, 4: 105}                                   # match clock (minutes) at each kick-off, extra time included
    clock = [[float(t0), 60.0 * at[int(p)]] for p, t0 in starts.items() if int(p) in at]
    home, away, score = team_names(a.match)
    out = {"match": a.match, "dataset": settings.dataset(a.match), "home": home, "away": away, "final_score": score,
           "fps": 10.0 / a.every, "t": np.round(tt, 2).tolist(),
           "px": ints(px), "py": ints(py),
           "bx": ints(f["ball_x_m"].to_numpy()), "by": ints(f["ball_y_m"].to_numpy()), "bz": ints(f["ball_z"].fillna(0).to_numpy()),
           "live": (f["game_mode"] == "normal").astype(int).tolist(),
           "sh": f["score_home"].astype(int).tolist(), "sa": f["score_away"].astype(int).tolist(),
           "rosters": rosters, "names": player_names(a.match, d), "clock": clock, "layers": []}
    for spec in a.layer:
        name, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=True)
        g, b, lt = z["grf"], z["ball_grf"], z["time_s"]
        ok = ~np.isnan(g[0, :, 0]) & ~np.isnan(b[:, 0])
        idx = np.flatnonzero(ok)[:: a.every]
        # a run keeps the player order of its segment (keeper, then by role): its shirt numbers pair each ghost with the real player
        mm = re.match(r"(?:grf_[a-z]+_)?([A-Za-z0-9-]+)_(\d+)", os.path.basename(path))       # grf_<kind>_<match>_<start>[_anything]
        seg = f"data/real/processed/{mm.group(1)}_segment_{mm.group(2)}.npz" if mm else ""
        jl = []
        if os.path.exists(seg):
            sz = np.load(seg, allow_pickle=True)
            if "jersey_home" in sz.files:
                jl = [int(x) for x in sz["jersey_home"]] + [int(x) for x in sz["jersey_away"]]
        out["layers"].append({"name": name, "jerseys": jl, "t": np.round(lt[idx], 2).tolist(),
                              "px": ints(g[:, idx, 0].T), "py": ints(g[:, idx, 1].T), "bx": ints(b[idx, 0]), "by": ints(b[idx, 1])})
    name = a.out or (f"{a.match}" + (f"_{int(a.start_s)}" if a.start_s is not None else "_full") + ".json")
    os.makedirs("viewer/data", exist_ok=True)
    with open(f"viewer/data/{name}", "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    idx_path = "viewer/data/index.json"
    index = json.load(open(idx_path, encoding="utf-8")) if os.path.exists(idx_path) else []
    def m_clock(t):
        c = [x for x in clock if x[0] <= t] or clock[:1]
        return t - c[-1][0] + c[-1][1]
    title = (f"{home} v {away} ({a.match})" + (f", {int(m_clock(tt[0]) // 60)}' to {int(m_clock(tt[-1]) // 60) + 1}'" if a.start_s is not None else ", full match")
             + (f" + {', '.join(l['name'] for l in out['layers'])}" if out["layers"] else ""))
    index = [e for e in index if e["file"] != name] + [{"file": name, "title": title}]
    json.dump(index, open(idx_path, "w", encoding="utf-8"), indent=1)
    print(f"wrote viewer/data/{name}: {F} frames ({F / out['fps'] / 60:.1f} min), {len(rosters)} line-ups, {len(out['layers'])} layer(s), "
          f"{os.path.getsize(f'viewer/data/{name}') / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
