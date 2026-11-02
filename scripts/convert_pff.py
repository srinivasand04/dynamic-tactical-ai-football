"""Convert PFF FC World Cup 2022 matches into the SAME layout as the DFL export (convert_real.py).

Input : data/external/pff_worldcup/raw/FIFA World Cup 2022/{Tracking Data,Metadata,Rosters,Event Data}
Output: data/pff/processed/<game_id>.csv.gz   10 frames/s, metres, 105 x 68, origin bottom-left, one row per player per
                                              frame, 'home' always attacks towards x = 105 (the same columns as DFL, plus
                                              'visible' and 'ball_visible', see below)
        data/pff/matches.csv                  one row per match (teams, score, date, round)

What is different from DFL, and how it is handled (this is where the quality of the data is decided):
  * PFF tracking is read from broadcast video. A player outside the camera view is ESTIMATED, not measured. The flag is
    kept: column 'visible' = 1 when the position was seen, 0 when estimated or interpolated. The ball is often missing
    (38% of frames in the first match): gaps up to --ball-gap seconds are interpolated, longer gaps are left out
    (those frames do not exist in the file, so the Key State code never sees a ball it had to invent).
  * 29.97 fps -> 10 fps: the first frame of every 100 ms window that has 11 players per team (a substitution frame has
    12 for a moment and is skipped); short gaps in a player's track (<= --gap seconds) are interpolated.
  * Positions: PFF's SMOOTHED tracks (less jitter than the raw ones), centre origin -> bottom-left origin.
  * Direction: homeTeamStartLeft in the metadata tells which way the home team attacks in each period; periods are flipped
    so the home team always attacks towards x = 105.
  * Ball owner: PFF has no per-frame owner; it is the team of the latest on-the-ball event. game_mode is 'dead' from an
    OUT / SUB / OFF / ON / END event until the next on-the-ball or kickoff event (an approximation, DFL gives a flag).
  * Goals come from the event files (shot outcome 'G'), matched to the tracking through the video clock.

Usage :  python scripts\\convert_pff.py                      (all 64 matches, 4 processes)
         python scripts\\convert_pff.py 3859 3860            (just these)
         python scripts\\convert_pff.py --raw <folder>       (another copy of the dataset)
"""
import argparse, bz2, glob, json, os
from multiprocessing import Pool
import numpy as np
import pandas as pd

L, W, FPS = 105.0, 68.0, 10
RAW = "data/external/pff_worldcup/raw/FIFA World Cup 2022"
OUT = "data/pff/processed"
ROLE = {"GK": "GK", "CB": "CB", "LCB": "CB", "RCB": "CB", "MCB": "CB", "LB": "LB", "RB": "RB", "LWB": "LB", "RWB": "RB",
        "DM": "DM", "CM": "CM", "LCM": "CM", "RCM": "CM", "LM": "LM", "LW": "LM", "RM": "RM", "RW": "RM",
        "AM": "AM", "CF": "CF", "ST": "CF", "LF": "CF", "RF": "CF"}
DEAD = {"OUT", "SUB", "OFF", "ON", "END"}


def read_tracking(path):
    """One (period, tick) -> first frame that has 11 players per team. Returns lists, one entry per kept frame."""
    keep = {}
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            per, el = r.get("period"), r.get("periodElapsedTime")
            h, a = r.get("homePlayersSmoothed"), r.get("awayPlayersSmoothed")
            if per is None or el is None or not h or not a or len(h) != 11 or len(a) != 11:
                continue
            key = (int(per), int(np.floor(el * FPS + 1e-6)))
            if key in keep:
                continue
            ball = r.get("ballsSmoothed")
            ge = r.get("game_event")
            keep[key] = (r["videoTimeMs"] / 1000.0, h, a, ball, ge)
    return keep


def convert(game_id, raw=RAW, gap_s=1.0, ball_gap_s=2.0):
    meta = json.load(open(f"{raw}/Metadata/{game_id}.json", encoding="utf-8"))[0]
    roster = json.load(open(f"{raw}/Rosters/{game_id}.json", encoding="utf-8"))
    ids = {"home": str(meta["homeTeam"]["id"]), "away": str(meta["awayTeam"]["id"])}
    side_of = {v: k for k, v in ids.items()}
    role_of = {(side_of[str(r["team"]["id"])], str(r["shirtNumber"])): ROLE.get(r.get("positionGroupType"), "CM") for r in roster}

    keep = read_tracking(f"{raw}/Tracking Data/{game_id}.jsonl.bz2")
    start_left = {1: meta["homeTeamStartLeft"], 2: None, 3: meta.get("homeTeamStartLeftExtraTime"), 4: None}
    start_left[2] = None if start_left[1] is None else not start_left[1]
    start_left[4] = None if start_left[3] is None else not start_left[3]

    periods = sorted({p for p, _ in keep if start_left.get(p) is not None})
    offset, t0 = {}, 0
    for p in periods:                                   # one clock: a period starts where the previous one ended
        offset[p] = t0
        t0 += max(t for q, t in keep if q == p) + 1

    rows, ball_rows, vid, mode, owner = [], [], {}, {}, {}
    last_mode, last_owner = "normal", "none"
    for (p, t) in sorted(keep):
        if p not in offset:
            continue
        v, h, a, ball, ge = keep[(p, t)]
        g = offset[p] + t
        vid[g] = v
        flip = not start_left[p]                        # home on the right in this period: turn the pitch
        for side, lst in (("home", h), ("away", a)):
            for q in lst:
                if q.get("x") is None or q.get("y") is None:
                    continue
                x, y = q["x"] + L / 2, q["y"] + W / 2
                if flip:
                    x, y = L - x, W - y
                rows.append((g, side, str(q["jerseyNum"]), x, y, 1 if q["visibility"] == "VISIBLE" else 0, p))
        if ball and ball.get("x") is not None and ball.get("y") is not None:
            bx, by = ball["x"] + L / 2, ball["y"] + W / 2
            if flip:
                bx, by = L - bx, W - by
            ball_rows.append((g, bx, by, ball.get("z") or 0.0, 1 if ball.get("visibility") == "VISIBLE" else 0))
        if ge:
            kind = ge.get("game_event_type")
            if kind in DEAD:
                last_mode = "dead"
            elif kind:
                last_mode = "normal"
            if kind == "OTB" and str(ge.get("team_id")) in side_of:
                last_owner = side_of[str(ge["team_id"])]
        mode[g], owner[g] = last_mode, last_owner if last_mode == "normal" else "none"

    d = pd.DataFrame(rows, columns=["frame", "team", "jersey", "x_m", "y_m", "visible", "period"])
    grid = np.arange(d["frame"].min(), d["frame"].max() + 1)
    gap = int(gap_s * FPS)

    # player tracks onto the full 10 Hz grid, short gaps interpolated (interpolated points count as 'not seen')
    parts = []
    for (team, jersey), g in d.groupby(["team", "jersey"]):
        g = g.set_index("frame").reindex(grid)
        seen = g["x_m"].notna()
        g[["x_m", "y_m"]] = g[["x_m", "y_m"]].interpolate(limit=gap, limit_area="inside")
        g["period"] = g["period"].ffill(limit=gap)
        g["visible"] = g["visible"].where(seen, 0)
        g = g[g["x_m"].notna()].copy()
        g["team"], g["jersey"] = team, jersey
        g["vx_ms"] = g["x_m"].diff() * FPS
        g["vy_ms"] = g["y_m"].diff() * FPS
        g.loc[g.index.to_series().diff() != 1, ["vx_ms", "vy_ms"]] = np.nan
        parts.append(g.rename_axis("frame").reset_index())
    d = pd.concat(parts, ignore_index=True)

    b = pd.DataFrame(ball_rows, columns=["frame", "ball_x_m", "ball_y_m", "ball_z", "ball_visible"]).set_index("frame").reindex(grid)
    seen_b = b["ball_x_m"].notna()
    b[["ball_x_m", "ball_y_m", "ball_z"]] = b[["ball_x_m", "ball_y_m", "ball_z"]].interpolate(limit=int(ball_gap_s * FPS), limit_area="inside")
    b["ball_visible"] = b["ball_visible"].where(seen_b, 0).fillna(0)
    b = b[b["ball_x_m"].notna()]

    f = b.rename_axis("frame").reset_index()
    f["game_mode"] = f["frame"].map(mode)
    f["ball_owner_team"] = f["frame"].map(owner)
    f["ball_owner_player"] = -1
    f["time_s"] = f["frame"] / FPS

    # goals: shot outcome 'G' in the event file, placed on the tracking clock through the video time
    ev = json.load(open(f"{raw}/Event Data/{game_id}.json", encoding="utf-8"))
    vt = np.array([vid[k] for k in sorted(vid)])
    vk = np.array(sorted(vid))
    goal_t = {"home": [], "away": []}
    for e in ev:
        pe = e.get("possessionEvents") or {}
        ge = e.get("gameEvents") or {}
        if pe.get("shotOutcomeType") == "G" and ge.get("teamId") is not None:
            s = side_of.get(str(ge["teamId"]))
            if s:
                goal_t[s].append(vk[min(np.searchsorted(vt, e["startTime"]), len(vk) - 1)])
    for s in ("home", "away"):
        f[f"score_{s}"] = np.searchsorted(np.sort(goal_t[s]), f["frame"].to_numpy(), side="right")

    # keep only frames where the ball exists and both teams still have 11; order: keeper first, then jersey number
    d = d.merge(f, on="frame")
    d["role"] = [role_of.get((t, j), "CM") for t, j in zip(d["team"], d["jersey"])]
    d["not_gk"] = d["role"] != "GK"
    d["jersey_n"] = pd.to_numeric(d["jersey"], errors="coerce").fillna(99)
    d = d.sort_values(["frame", "team", "not_gk", "jersey_n"])
    d["player_id"] = d.groupby(["frame", "team"]).cumcount()
    size = d.groupby(["frame", "team"]).size().unstack()
    full = size.index[(size["home"] == 11) & (size["away"] == 11)]
    d = d[d["frame"].isin(full)]
    d["dfl_id"] = "PFF-" + d["team"] + "-" + d["jersey"].astype(str)   # no player id on the tracking rows: side + shirt number is unique inside a match
    cols = ["frame", "time_s", "team", "player_id", "role", "x_m", "y_m", "vx_ms", "vy_ms", "ball_x_m", "ball_y_m", "ball_z",
            "ball_owner_team", "ball_owner_player", "game_mode", "score_home", "score_away", "period", "dfl_id", "jersey",
            "visible", "ball_visible"]
    d = d[cols].sort_values(["frame", "team", "player_id"])
    d["period"] = d["period"].astype(int)
    d[["visible", "ball_visible"]] = d[["visible", "ball_visible"]].astype(int)
    d = d.round(2)

    os.makedirs(OUT, exist_ok=True)
    d.to_csv(f"{OUT}/{game_id}.csv.gz", index=False)
    n = d["frame"].nunique()
    return {"match_id": str(game_id), "home": meta["homeTeam"]["name"], "away": meta["awayTeam"]["name"],
            "score": f"{int(d['score_home'].max())}-{int(d['score_away'].max())}", "date": meta["date"][:10],
            "week": meta.get("week"), "raw_fps": meta.get("fps"), "frames_10fps": n, "minutes": round(n / FPS / 60, 1),
            "span_minutes": round((grid[-1] - grid[0]) / FPS / 60, 1)}


def _one(args):
    try:
        return convert(*args)
    except Exception as e:                                              # one bad file must not stop the other 63
        return {"match_id": str(args[0]), "error": repr(e)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*")
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    games = a.games or sorted(os.path.basename(p).split(".")[0] for p in glob.glob(f"{a.raw}/Tracking Data/*.jsonl.bz2"))
    jobs = [(g, a.raw) for g in games if not os.path.exists(f"{OUT}/{g}.csv.gz") or a.games]
    print(f"{len(jobs)} matches to convert, {len(games) - len(jobs)} already done", flush=True)
    done = []
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(_one, jobs):
            print(r, flush=True)
            done.append(r)
    ok = [r for r in done if "error" not in r]
    if not ok:
        raise SystemExit("nothing converted")
    reg = f"data/pff/matches.csv"
    old = pd.read_csv(reg, dtype={"match_id": str}) if os.path.exists(reg) else pd.DataFrame()
    pd.concat([old, pd.DataFrame(ok)]).drop_duplicates("match_id", keep="last").sort_values("match_id").to_csv(reg, index=False)
    print(f"\nconverted {len(ok)}, failed {len(done) - len(ok)}; registry {reg}")
