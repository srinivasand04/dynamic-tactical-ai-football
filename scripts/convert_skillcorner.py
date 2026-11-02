"""Convert SkillCorner open data (A-League matches) into the SAME layout as the DFL and PFF exports.

Input : data/external/skillcorner_opendata/data/matches/<id>/<id>_match.json and <id>_tracking_extrapolated.jsonl
Output: data/skillcorner/processed/<id>.csv.gz   10 frames/s, metres, 105 x 68, origin bottom-left, one row per player per frame,
                                                 'home' always attacks towards x = 105 (the columns of convert_pff.py)
        data/skillcorner/matches.csv             one row per match (teams, score, date, round)

What is different from DFL, and how it is handled:
  * The tracking is read from broadcast video (like PFF). Players outside the camera view are EXTRAPOLATED. The flag is kept:
    column 'visible' = 1 when the player was detected in that frame, 0 when extrapolated or interpolated; 'ball_visible' the same
    for the ball. The share of detected positions is reported by data_quality.py.
  * Already 10 frames per second. A frame with a team that does not have 11 players (a substitution moment) is skipped; short gaps in
    a player's track (<= --gap seconds) and ball gaps (<= --ball-gap seconds) are interpolated, longer ones are left out.
  * The pitch (104 x 68 here, it differs per stadium) is stretched to 105 x 68, so all datasets share one pitch.
  * Direction: home_team_side in the match file gives the home team's direction in each period; periods are turned so the home
    team always attacks towards x = 105.
  * Ball owner: the 'possession' group of the frame; game_mode is not provided by SkillCorner and is set to 'normal' when somebody has the
    ball or the ball is detected and 'dead' otherwise (an approximation, like PFF's).
  * Score: only the final score is known per match (registry); the per-frame score columns are 0. Movement models do not use them.

Usage :  python scripts\\convert_skillcorner.py                (all matches, 4 processes)
         python scripts\\convert_skillcorner.py 1886347        (just this one)
"""
import argparse, glob, json, os
from multiprocessing import Pool
import numpy as np
import pandas as pd

L, W, FPS = 105.0, 68.0, 10
RAW = "data/external/skillcorner_opendata/data/matches"
OUT = "data/skillcorner/processed"
ROLE = {"GK": "GK", "CB": "CB", "LCB": "CB", "RCB": "CB", "LB": "LB", "RB": "RB", "LWB": "LB", "RWB": "RB", "DM": "DM",
        "CM": "CM", "LM": "LM", "LW": "LM", "RM": "RM", "RW": "RM", "AM": "AM", "CF": "CF", "ST": "CF", "LF": "CF", "RF": "CF",
        "MF": "CM", "DF": "CB", "FW": "CF"}


def convert(mid, raw=RAW, gap_s=1.0, ball_gap_s=2.0):
    base = f"{raw}/{mid}/{mid}"
    meta = json.load(open(base + "_match.json", encoding="utf-8"))
    home_id, away_id = meta["home_team"]["id"], meta["away_team"]["id"]
    pl = {p["id"]: {"side": "home" if p["team_id"] == home_id else "away", "jersey": str(p["number"]),
                    "role": ROLE.get(p["player_role"]["acronym"], "CM")} for p in meta["players"]}
    pl_len = float(meta.get("pitch_length") or L)
    sides = meta["home_team_side"]                                          # ['right_to_left', 'left_to_right']: home direction per period
    home_attacks_right = {i + 1: s == "left_to_right" for i, s in enumerate(sides)}

    rows, ball_rows, owner = [], [], {}
    with open(base + "_tracking_extrapolated.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pd_ = r["player_data"]
            per = r["period"]
            if not pd_ or per is None or per not in home_attacks_right:
                continue
            cnt = {"home": 0, "away": 0}
            tmp = []
            for q in pd_:
                info = pl.get(q["player_id"])
                if info is None or q["x"] is None or q["y"] is None:
                    continue
                cnt[info["side"]] += 1
                tmp.append((info, q))
            if cnt["home"] != 11 or cnt["away"] != 11:
                continue
            g = r["frame"]
            flip = not home_attacks_right[per]
            for info, q in tmp:
                x, y = (q["x"] + pl_len / 2) * L / pl_len, q["y"] + W / 2
                if flip:
                    x, y = L - x, W - y
                rows.append((g, info["side"], info["jersey"], info["role"], x, y, 1 if q["is_detected"] else 0, per))
            b = r["ball_data"]
            if b and b.get("x") is not None and b.get("y") is not None:
                x, y = (b["x"] + pl_len / 2) * L / pl_len, b["y"] + W / 2
                if flip:
                    x, y = L - x, W - y
                ball_rows.append((g, x, y, b.get("z") or 0.0, 1 if b.get("is_detected") else 0))
            grp = (r.get("possession") or {}).get("group")
            owner[g] = {"home team": "home", "away team": "away"}.get(grp, "none")

    d = pd.DataFrame(rows, columns=["frame", "team", "jersey", "role", "x_m", "y_m", "visible", "period"])
    grid = np.arange(d["frame"].min(), d["frame"].max() + 1)
    gap = int(gap_s * FPS)
    parts = []
    for (team, jersey), g in d.groupby(["team", "jersey"]):
        role = g["role"].iloc[0]
        g = g.set_index("frame").reindex(grid)
        seen = g["x_m"].notna()
        g[["x_m", "y_m"]] = g[["x_m", "y_m"]].interpolate(limit=gap, limit_area="inside")
        g["period"] = g["period"].ffill(limit=gap)
        g["visible"] = g["visible"].where(seen, 0)
        g = g[g["x_m"].notna()].copy()
        g["team"], g["jersey"], g["role"] = team, jersey, role
        g["vx_ms"] = g["x_m"].diff() * FPS
        g["vy_ms"] = g["y_m"].diff() * FPS
        g.loc[g.index.to_series().diff() != 1, ["vx_ms", "vy_ms"]] = np.nan
        parts.append(g.rename_axis("frame").reset_index())
    d = pd.concat(parts, ignore_index=True)

    b = pd.DataFrame(ball_rows, columns=["frame", "ball_x_m", "ball_y_m", "ball_z", "ball_visible"]).drop_duplicates("frame").set_index("frame").reindex(grid)
    seen_b = b["ball_x_m"].notna()
    b[["ball_x_m", "ball_y_m", "ball_z"]] = b[["ball_x_m", "ball_y_m", "ball_z"]].interpolate(limit=int(ball_gap_s * FPS), limit_area="inside")
    b["ball_visible"] = b["ball_visible"].where(seen_b, 0).fillna(0)
    b = b[b["ball_x_m"].notna()]
    f = b.rename_axis("frame").reset_index()
    f["ball_owner_team"] = f["frame"].map(owner).fillna("none")
    f["game_mode"] = np.where((f["ball_owner_team"] != "none") | (f["ball_visible"] == 1), "normal", "dead")
    f["ball_owner_player"] = -1
    f["time_s"] = f["frame"] / FPS
    f["score_home"], f["score_away"] = 0, 0

    d = d.merge(f, on="frame")
    d["not_gk"] = d["role"] != "GK"
    d["jersey_n"] = pd.to_numeric(d["jersey"], errors="coerce").fillna(99)
    d = d.sort_values(["frame", "team", "not_gk", "jersey_n"])
    d["player_id"] = d.groupby(["frame", "team"]).cumcount()
    size = d.groupby(["frame", "team"]).size().unstack()
    full = size.index[(size["home"] == 11) & (size["away"] == 11)]
    d = d[d["frame"].isin(full)]
    d["dfl_id"] = "SKC-" + d["team"] + "-" + d["jersey"].astype(str)
    cols = ["frame", "time_s", "team", "player_id", "role", "x_m", "y_m", "vx_ms", "vy_ms", "ball_x_m", "ball_y_m", "ball_z",
            "ball_owner_team", "ball_owner_player", "game_mode", "score_home", "score_away", "period", "dfl_id", "jersey",
            "visible", "ball_visible"]
    d = d[cols].sort_values(["frame", "team", "player_id"])
    d["period"] = d["period"].astype(int)
    d[["visible", "ball_visible"]] = d[["visible", "ball_visible"]].astype(int)
    d = d.round(2)
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(f"{OUT}/{mid}.csv.gz", index=False)
    n = d["frame"].nunique()
    return {"match_id": str(mid), "home": meta["home_team"]["short_name"], "away": meta["away_team"]["short_name"],
            "score": f"{meta['home_team_score']}-{meta['away_team_score']}", "date": meta["date_time"][:10],
            "week": (meta.get("competition_round") or {}).get("name"), "raw_fps": FPS, "frames_10fps": n, "minutes": round(n / FPS / 60, 1),
            "span_minutes": round((grid[-1] - grid[0]) / FPS / 60, 1)}


def _one(mid):
    try:
        return convert(mid)
    except Exception as e:                                              # one bad file must not stop the others
        return {"match_id": str(mid), "error": repr(e)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("matches", nargs="*")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    ids = a.matches or sorted(os.path.basename(p) for p in glob.glob(f"{RAW}/*") if os.path.isdir(p))
    jobs = [m for m in ids if a.matches or not os.path.exists(f"{OUT}/{m}.csv.gz")]
    print(f"{len(jobs)} matches to convert, {len(ids) - len(jobs)} already done", flush=True)
    done = []
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(_one, jobs):
            print(r, flush=True)
            done.append(r)
    ok = [r for r in done if "error" not in r]
    if not ok:
        raise SystemExit("nothing converted")
    reg = "data/skillcorner/matches.csv"
    old = pd.read_csv(reg, dtype={"match_id": str}) if os.path.exists(reg) else pd.DataFrame()
    pd.concat([old, pd.DataFrame(ok)]).drop_duplicates("match_id", keep="last").sort_values("match_id").to_csv(reg, index=False)
    print(f"\nconverted {len(ok)}, failed {len(done) - len(ok)}; registry {reg}")
