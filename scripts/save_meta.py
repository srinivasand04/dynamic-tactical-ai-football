"""Save match metadata (teams, players, pitch size, direction, score) to data/real/raw."""
import sys, json
import pandas as pd
from kloppy import sportec

match_id = sys.argv[1] if len(sys.argv) > 1 else "J03WMX"
import os
from kloppy._providers.sportec import get_IDSSE_url
local = {k: f"data/real/raw_xml/{os.path.basename(get_IDSSE_url(match_id, k))}" for k in ("meta", "tracking")}
if all(os.path.exists(p) for p in local.values()):                              # files kept by download_real.py
    meta = sportec.load_tracking(raw_data=local["tracking"], meta_data=local["meta"], limit=10).metadata
else:                                                                           # older matches downloaded before the local copy existed
    meta = sportec.load_open_tracking_data(match_id=match_id, limit=10).metadata   # only 10 frames: fast

players = []
for team in meta.teams:
    for p in team.players:
        players.append({
            "player_id": p.player_id,
            "team_id": team.team_id,
            "team_name": team.name,
            "ground": str(team.ground),            # home / away
            "jersey_no": p.jersey_no,
            "name": p.name,
            "starting": p.starting,
            "position": str(p.starting_position),
        })
pd.DataFrame(players).to_csv(f"data/real/raw/{match_id}_players.csv", index=False)

info = {
    "match_id": match_id,
    "pitch_dimensions": str(meta.pitch_dimensions),
    "coordinate_system": str(meta.coordinate_system),
    "orientation": str(meta.orientation),
    "frame_rate": meta.frame_rate,
    "pitch_length_m": meta.pitch_dimensions.pitch_length,
    "pitch_width_m": meta.pitch_dimensions.pitch_width,
    "date": str(meta.date),
    "score": str(meta.score),
    "score_home": meta.score.home,
    "score_away": meta.score.away,
    "teams": {t.team_id: f"{t.name} ({t.ground})" for t in meta.teams},
}
with open(f"data/real/raw/{match_id}_meta.json", "w") as f:
    json.dump(info, f, indent=2)

print(json.dumps(info, indent=2))
print(f"\n{len(players)} players saved")