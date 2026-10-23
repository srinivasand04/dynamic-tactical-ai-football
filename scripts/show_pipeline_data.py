"""Print a few real rows from every stage of the pipeline, in the order the data flows.

Usage:  python scripts\\show_pipeline_data.py                      (real J03WMX and simulated match_001)
        python scripts\\show_pipeline_data.py --real J03WN1 --sim match_002 --time 600
"""
import argparse
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 42)


def title(text):
    print("\n" + "=" * 100 + f"\n{text}\n" + "=" * 100)


ap = argparse.ArgumentParser()
ap.add_argument("--real", default="J03WMX")
ap.add_argument("--sim", default="match_001")
ap.add_argument("--time", type=float, default=600.0, help="show the Key State closest to this second (real match)")
a = ap.parse_args()
R, S = f"data/real/processed/{a.real}", f"data/simulated/{a.sim}"

title("STEP 0  RAW REAL DATA   data/real/raw/<id>_tracking.csv.gz   (one row per frame, 25 per second, x and y between 0 and 1)")
t = pd.read_csv(f"data/real/raw/{a.real}_tracking.csv.gz", nrows=3)
print(t.iloc[:, :14].round(3).to_string(index=False))

title("STEP 0  RAW REAL EVENTS   data/real/raw/<id>_events.csv.gz")
e = pd.read_csv(f"data/real/raw/{a.real}_events.csv.gz", nrows=3)
print(e[["event_type", "period_id", "timestamp", "team_id", "player_id", "set_piece_type", "result", "coordinates_x", "coordinates_y"]].to_string(index=False))

title("STEP 0  RAW PLAYERS   data/real/raw/<id>_players.csv   (which team and position each player id belongs to)")
print(pd.read_csv(f"data/real/raw/{a.real}_players.csv").head(3).to_string(index=False))

title("STEP 1  COMMON FORMAT, REAL   data/real/processed/<id>.csv.gz   (metres, 10 frames per second, one row per player per frame)")
c = pd.read_csv(R + ".csv.gz", nrows=2000)
print(c[(c.frame == 1) & (c.player_id < 3)].drop(columns=["ball_z"]).to_string(index=False))

title("STEP 1  COMMON FORMAT, SIMULATED   data/simulated/match_00N.csv   (the same columns, without period / dfl_id / jersey)")
c = pd.read_csv(S + ".csv", nrows=200)
print(c[(c.frame == 1) & (c.player_id < 3)].drop(columns=["ball_z"]).to_string(index=False))

title("STEP 2  KEY STATES   <name>_adaptive_ks.csv   (the selected frames)")
for label, p in (("real", R), ("simulated", S)):
    print(f"-- {label}")
    print(pd.read_csv(p + "_adaptive_ks.csv").iloc[1:5].round(2).to_string(index=False))

title("STEP 3  GEOMETRY FEATURES PER KEY STATE   <name>_ks_features.csv   (used to find sequence boundaries)")
f = pd.read_csv(R + "_ks_features.csv")
k = int((f["time_s"] - a.time).abs().idxmin())
print(f.iloc[k:k + 2].round(2).T.to_string(header=False))

title("STEP 4  SEQUENCES   <name>_sequences.csv, <name>_boundary_events.csv, <name>_sequence_types.csv")
print(pd.read_csv(R + "_sequences.csv").iloc[30:35].round(1).to_string(index=False))
print()
print(pd.read_csv(R + "_boundary_events.csv").head(4).to_string(index=False))
print()
st = pd.read_csv(R + "_sequence_types.csv")
print(st[["seq_id", "start_s", "duration_s", "n_ks", "mean_speed", "width", "depth", "ball_dead_%", "possession", "type", "starts_after", "ends_with"]].iloc[[86, 117]].round(1).to_string(index=False))

title("STEP 5  GRAPH ARRAYS   <name>_graphs.npz   (23 node slots: 0-10 home, 11-21 away, 22 = ball)")
g = np.load(R + "_graphs.npz")
for key in g.files:
    print(f"  {key:10s} shape {str(g[key].shape):16s}")
k = int(np.argmin(np.abs(g["time_s"] - a.time)))
print(f"\nKey State {int(g['ks_id'][k])} at {g['time_s'][k]:.1f} s (gap {g['gap_s'][k]:.1f} s, sequence {int(g['seq_id'][k])}); nodes shown: a few of 23")
roles = ["GK", "CB", "LB", "RB", "DM", "CM", "LM", "RM", "AM", "CF"]
rows = []
for i in (0, 1, 5, 10, 11, 12, 16, 21, 22):
    rows.append({"slot": i, "team": ["home", "away", "ball"][g["team"][i]], "role": roles[g["role"][k, i]] if g["role"][k, i] >= 0 else "-",
                 "x_m": round(g["xy"][k, i, 0], 1), "y_m": round(g["xy"][k, i, 1], 1),
                 "vx": round(g["vel"][k, i, 0], 1), "vy": round(g["vel"][k, i, 1], 1),
                 "dist_ball": round(g["dist_ball"][k, i], 1),
                 "dx_prev": round(g["dxy_prev"][k, i, 0], 1), "dy_prev": round(g["dxy_prev"][k, i, 1], 1)})
print(pd.DataFrame(rows).to_string(index=False))
print()
for name in ("adj_team", "adj_opp", "adj_ball"):
    print(f"  {name}: {int(np.triu(g[name][k]).sum())} links in this graph")
ii, jj = np.nonzero(np.triu(g["adj_opp"][k]))
print("  opponent links (slot, slot, metres apart):",
      [(int(i), int(j), round(float(np.linalg.norm(g['xy'][k, i] - g['xy'][k, j])), 1)) for i, j in zip(ii, jj)][:5])
ii, jj = np.nonzero(np.triu(g["adj_ball"][k]))
print("  ball links (slot, slot):", [(int(i), int(j)) for i, j in zip(ii, jj)])
