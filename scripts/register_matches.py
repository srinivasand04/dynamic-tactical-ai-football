"""Build data/real/matches.csv: one row per real match, identified by its DFL match_id (the key used in every file name).

Reads  data/real/raw/<id>_meta.json, <id>_players.csv, <id>_events.csv.gz and data/real/processed/<id>.csv.gz
Writes data/real/matches.csv

Usage:  python scripts\\register_matches.py
"""
import glob
import json
import os
import pandas as pd

rows = []
for meta_path in sorted(glob.glob("data/real/raw/*_meta.json")):
    mid = os.path.basename(meta_path).replace("_meta.json", "")
    meta = json.load(open(meta_path))
    players = pd.read_csv(f"data/real/raw/{mid}_players.csv")
    teams = players.drop_duplicates("team_id").set_index("ground")
    side = players.drop_duplicates("team_id").set_index("team_id")["ground"]

    row = {"match_id": mid, "home": teams.loc["home", "team_name"], "away": teams.loc["away", "team_name"],
           "score": f"{meta['score_home']}-{meta['score_away']}", "date": meta["date"][:10],
           "pitch_m": f"{meta['pitch_length_m']:.0f}x{meta['pitch_width_m']:.0f}", "raw_fps": meta["frame_rate"]}

    proc = f"data/real/processed/{mid}.csv.gz"
    if os.path.exists(proc):
        d = pd.read_csv(proc, usecols=["frame", "time_s", "team"])
        n = d.groupby(["frame", "team"]).size().unstack()
        row.update({"frames_10fps": d["frame"].nunique(), "minutes": round(d["time_s"].max() / 60, 1),
                    "min_players_home": int(n["home"].min()), "min_players_away": int(n["away"].min())})

    ev = pd.read_csv(f"data/real/raw/{mid}_events.csv.gz")
    reds = ev[(ev["event_type"] == "CARD") & ev["card_type"].isin(["RED", "SECOND_YELLOW"])]
    row["red_cards"] = "; ".join(f"{side.get(r.team_id, '?')} {pd.to_timedelta(r.timestamp).total_seconds() / 60:.0f}' (half {r.period_id})"
                                 for r in reds.itertuples()) or "none"
    rows.append(row)

out = pd.DataFrame(rows)
out.to_csv("data/real/matches.csv", index=False)
print(out.to_string(index=False))
print("\nwrote data/real/matches.csv")
