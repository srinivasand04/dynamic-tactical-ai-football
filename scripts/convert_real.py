"""Convert a DFL/Sportec match (kloppy CSVs) into the SAME layout as the GRF export.

Input : data/real/raw/<match>_tracking.csv.gz, _events.csv.gz, _players.csv
Output: data/real/processed/<match>.csv.gz   (10 frames/s, metres, 105 x 68, origin bottom-left,
        one row per player per frame, 'home' always attacks towards x = 105)

Usage :  python scripts\\convert_real.py J03WMX
"""
import sys
import numpy as np
import pandas as pd

L, W, FPS = 105.0, 68.0, 10

match_id = sys.argv[1] if len(sys.argv) > 1 else "J03WMX"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/real/processed"
raw = f"data/real/raw/{match_id}"

t = pd.read_csv(f"{raw}_tracking.csv.gz")
ev = pd.read_csv(f"{raw}_events.csv.gz")
pl = pd.read_csv(f"{raw}_players.csv")

# ---- 1. one clock for the whole match (period 2 continues after period 1) ----
t["t_period"] = pd.to_timedelta(t["timestamp"]).dt.total_seconds()
offset = {1: 0.0, 2: t.loc[t.period_id == 1, "t_period"].max() + 0.04}
t["time_s"] = t["t_period"] + t["period_id"].map(offset)

# ---- 2. 25 fps -> 10 fps: keep the first frame in every 100 ms window ----
t["tick"] = np.floor(t["time_s"] * FPS + 1e-6).astype(int)
t = t.drop_duplicates("tick").reset_index(drop=True)
t["frame"] = np.arange(len(t))
t["time_s"] = t["frame"] / FPS

# ---- 3. direction: flip the period in which the home keeper stands on the right (x > 0.5) ----
home_gk = pl[(pl.ground == "home") & (pl.position == "Goalkeeper") & pl.starting].player_id.iloc[0]
flip_periods = [p for p, x in t.groupby("period_id")[f"{home_gk}_x"].mean().items() if x > 0.5]
flip = t["period_id"].isin(flip_periods).to_numpy()
print("flipping periods:", flip_periods)


def to_metres(x, y):
    # kloppy: x, y in [0, 1], origin top-left, y pointing DOWN  ->  metres, origin bottom-left, y UP
    x, y = np.asarray(x, float), np.asarray(y, float)
    x = np.where(flip, 1 - x, x)
    y = np.where(flip, 1 - y, y)
    return x * L, (1 - y) * W


# ---- 4. who is who ----
team_side = {r.team_id: r.ground for r in pl.drop_duplicates("team_id").itertuples()}   # club id -> home/away


def grf_role(pos):
    pos = str(pos)
    if "Goalkeeper" in pos: return "GK"
    if "Center Back" in pos: return "CB"
    if "Left Back" in pos: return "LB"
    if "Right Back" in pos: return "RB"
    if "Defensive" in pos: return "DM"
    if "Attacking" in pos: return "AM"
    if "Left Midfield" in pos or "Left Wing" in pos: return "LM"
    if "Right Midfield" in pos or "Right Wing" in pos: return "RM"
    if "Striker" in pos or "Forward" in pos: return "CF"
    return "CM"


# ---- 5. wide -> long: one row per player per frame ----
bx, by = to_metres(t["ball_x"], t["ball_y"])
parts = []
for r in pl.itertuples():
    if f"{r.player_id}_x" not in t.columns:
        continue                                          # never played
    xm, ym = to_metres(t[f"{r.player_id}_x"], t[f"{r.player_id}_y"])
    p = pd.DataFrame({"frame": t["frame"], "period": t["period_id"], "x_m": xm, "y_m": ym})
    p = p[p["x_m"].notna()].copy()                        # only frames where he is on the pitch
    p["vx_ms"] = p["x_m"].diff() * FPS                    # speed from position change
    p["vy_ms"] = p["y_m"].diff() * FPS
    jump = p["frame"].diff() != 1                         # first frame / after a gap: no speed
    p.loc[jump, ["vx_ms", "vy_ms"]] = np.nan
    p["team"], p["dfl_id"], p["jersey"] = r.ground, r.player_id, r.jersey_no
    p["role"] = grf_role(r.position)
    parts.append(p)
d = pd.concat(parts, ignore_index=True)

# player_id 0..10 inside each team and frame: keeper first (GRF puts him at index 0), then by jersey
d["not_gk"] = d["role"] != "GK"
d = d.sort_values(["frame", "team", "not_gk", "jersey"])
d["player_id"] = d.groupby(["frame", "team"]).cumcount()
size = d.groupby(["frame", "team"]).size()
assert size.max() <= 11, "more than 11 players for a team in some frame"
short = size[size < 11]          # a red card leaves a team with 10 (or fewer): allowed, but reported
if len(short):
    print("frames with fewer than 11 players:", short.index.get_level_values("frame").nunique(),
          "| teams:", sorted(short.index.get_level_values("team").unique()), "| min players:", int(size.min()))

# ---- 6. ball, possession, game mode ----
f = pd.DataFrame({
    "frame": t["frame"], "time_s": t["time_s"],
    "ball_x_m": bx, "ball_y_m": by, "ball_z": t["ball_z"],
    "ball_owner_team": t["ball_owning_team_id"].map(team_side).fillna("none"),
    "ball_owner_player": -1,                               # DFL gives the team only
    "game_mode": np.where(t["ball_state"] == "alive", "normal", "dead"),
})

# ---- 7. score from goals in the event data ----
ev["time_s"] = pd.to_timedelta(ev["timestamp"]).dt.total_seconds() + ev["period_id"].map(offset)
goals = ev[(ev.event_type == "SHOT") & (ev.result == "GOAL")]
for side in ("home", "away"):
    g = np.sort(goals[goals.team_id.map(team_side) == side]["time_s"].to_numpy())
    f[f"score_{side}"] = np.searchsorted(g, f["time_s"].to_numpy(), side="right")

# ---- 8. join and write in the GRF column order ----
d = d.merge(f, on="frame")
cols = ["frame", "time_s", "team", "player_id", "role", "x_m", "y_m", "vx_ms", "vy_ms",
        "ball_x_m", "ball_y_m", "ball_z", "ball_owner_team", "ball_owner_player",
        "game_mode", "score_home", "score_away", "period", "dfl_id", "jersey"]
d = d[cols].sort_values(["frame", "team", "player_id"]).round(2)
out = f"{out_dir}/{match_id}.csv.gz"
d.to_csv(out, index=False)

print(f"wrote {d['frame'].nunique()} frames ({d['frame'].nunique() / FPS / 60:.1f} min), {len(d)} rows to {out}")
print("final score:", int(f["score_home"].iloc[-1]), "-", int(f["score_away"].iloc[-1]))
print("ball in play:", f"{(f['game_mode'] == 'normal').mean() * 100:.0f}%")
