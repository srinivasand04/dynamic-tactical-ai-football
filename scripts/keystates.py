"""Key states from a GRF tracking CSV  (the same rules as the 'Tiny Test' slide).

Input : data/simulated/match_001.csv   (made by export_tracking.py: 10 frames/s, metres, 105 x 68 pitch)
Output: data/simulated/match_001_keystates.csv   (one row per second)

Usage :  python scripts\\keystates.py
         python scripts\\keystates.py --csv data\\simulated\\match_001.csv --out data\\simulated\\match_001_keystates.csv
"""
import argparse
import numpy as np
import pandas as pd

L, W, FPS = 105.0, 68.0, 10           # pitch length, width (m), frames per second

# ---- placeholder thresholds (to confirm with the professor) ----
PRESSURE_M   = 3.0    # nearest opponent closer than this  -> "under pressure"
COUNTER_M    = 20.0   # ball moves this far forward ...
COUNTER_S    = 6.0    # ... within this many seconds of winning it -> counter-attack


def load(path):
    d = pd.read_csv(path)
    d["x_m"] = d["x_m"].clip(0, L)            # GRF keepers stand slightly behind the goal line
    d["y_m"] = d["y_m"].clip(0, W)
    return d.sort_values(["frame", "team", "player_id"]).reset_index(drop=True)


def frame_table(d):
    """One row per frame: ball, possession, score, mode + player arrays shaped (frames, 11, 2)."""
    F = d["frame"].nunique()
    home = d[d.team == "home"][["x_m", "y_m"]].to_numpy().reshape(F, 11, 2)
    away = d[d.team == "away"][["x_m", "y_m"]].to_numpy().reshape(F, 11, 2)
    f = d.drop_duplicates("frame").set_index("frame")
    f = f[["time_s", "ball_x_m", "ball_y_m", "ball_owner_team", "game_mode", "score_home", "score_away"]].copy()
    # possession = the LAST team that touched the ball (the ball is 'owned' by nobody while it is in flight)
    f["poss"] = f["ball_owner_team"].replace("none", np.nan).ffill()
    return f, home, away


def compute(f, home, away):
    n = len(f)
    poss = f["poss"].to_numpy()
    bx, by = f["ball_x_m"].to_numpy(), f["ball_y_m"].to_numpy()
    ball = np.c_[bx, by]

    # ---- 1. attacking direction: 'home' attacks towards x = 105, 'away' towards x = 0 ----
    att_x = np.where(poss == "home", bx, np.where(poss == "away", L - bx, np.nan))   # 0 = own goal, 105 = opponent goal

    # ---- 2. pitch third of the ball, from the possessing team's point of view ----
    third = np.where(att_x < L / 3, "own", np.where(att_x < 2 * L / 3, "middle", "final"))
    third = np.where(np.isnan(att_x), "-", third)

    # ---- 3. pressure = distance from the ball to the nearest opponent of the possessing team ----
    press = np.full(n, np.nan)
    width = np.full(n, np.nan)
    depth = np.full(n, np.nan)
    for i in range(n):
        if poss[i] not in ("home", "away"):
            continue
        mine, theirs = (home[i], away[i]) if poss[i] == "home" else (away[i], home[i])
        press[i] = np.linalg.norm(theirs - ball[i], axis=1).min()
        outfield = mine[1:]                                  # skip the goalkeeper (index 0)
        width[i] = outfield[:, 1].max() - outfield[:, 1].min()   # spread across the pitch
        depth[i] = outfield[:, 0].max() - outfield[:, 0].min()   # spread along the pitch

    # ---- 4. turnovers: possession changes from one team to the other ----
    turnover = np.zeros(n, bool)
    turnover[1:] = (poss[1:] != poss[:-1]) & pd.notna(poss[1:]) & pd.notna(poss[:-1])

    # ---- 5. counter-attack: after a turnover the new team gets the ball >= 20 m forward within 6 s ----
    counter = np.zeros(n, bool)
    for t0 in np.flatnonzero(turnover):
        team = poss[t0]
        end = min(n, t0 + int(COUNTER_S * FPS) + 1)
        for j in range(t0, end):
            if poss[j] != team:
                break                                        # they lost it again
            if att_x[j] - att_x[t0] >= COUNTER_M:
                counter[t0:j + 1] = True
                break

    # ---- 6. goals: the score changes ----
    goal = np.zeros(n, bool)
    s = (f["score_home"] + f["score_away"]).to_numpy()
    goal[1:] = s[1:] > s[:-1]

    out = pd.DataFrame({
        "frame": f.index, "t": f["time_s"].to_numpy(), "poss": poss, "ball_x": bx, "ball_y": by,
        "att_x": att_x, "third": third, "press_m": press, "width_m": width, "depth_m": depth,
        "turnover": turnover, "counter": counter, "goal": goal, "mode": f["game_mode"].to_numpy(),
    })
    return out


def label(row):
    """Turn the numbers into ONE key state (priority order: first rule that matches wins)."""
    if row["goal"]:                         st = "GOAL"
    elif row["mode"] == "dead":             st = "STOPPAGE"
    elif row["mode"] != "normal":           st = "SET-PIECE (" + row["mode"] + ")"
    elif row["counter"]:                    st = "COUNTER-ATTACK"
    elif row["turnover"]:                   st = "TURNOVER"
    elif pd.isna(row["poss"]):              st = "LOOSE BALL"
    elif row["third"] == "final":           st = "FINAL THIRD"
    elif row["third"] == "middle":          st = "PROGRESSION"
    else:                                   st = "BUILD-UP"
    if pd.notna(row["press_m"]) and row["press_m"] < PRESSURE_M and st in ("BUILD-UP", "PROGRESSION", "FINAL THIRD"):
        st += " + pressure"
    return st


def to_seconds(fr):
    """Frames -> one row per second. Take the frame at the start of each second; events use the whole second."""
    fr = fr.copy()
    fr["sec"] = (fr["frame"] // FPS).astype(int)
    g = fr.groupby("sec")
    start = fr[fr["frame"] % FPS == 0].set_index("sec")      # exactly one frame per second (no mixing of frames)
    sec = start[["poss", "ball_x", "ball_y", "att_x", "third", "press_m", "width_m", "depth_m", "mode"]].copy()
    sec["turnover"] = g["turnover"].any()
    sec["counter"] = g["counter"].any()
    sec["goal"] = g["goal"].any()
    sec["state"] = sec.apply(label, axis=1)
    return sec.reset_index().rename(columns={"sec": "t"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/simulated/match_001.csv")
    ap.add_argument("--out", default="data/simulated/match_001_keystates.csv")
    a = ap.parse_args()

    d = load(a.csv)
    f, home, away = frame_table(d)
    fr = compute(f, home, away)
    sec = to_seconds(fr)
    sec.round(2).to_csv(a.out, index=False)

    # ---- quick summary: the numbers to compare with the real match ----
    print(f"wrote {len(sec)} seconds to {a.out}\n")
    print("Possession share (seconds):")
    print(sec["poss"].fillna("loose").value_counts(normalize=True).mul(100).round(1).to_string(), "\n")
    print("Key-state share (%):")
    base = sec["state"].str.replace(r"\(.*\)", "", regex=True).str.replace(r" \+ pressure", "", regex=True).str.strip()
    print(base.value_counts(normalize=True).mul(100).round(1).to_string(), "\n")
    print(f"turnovers: {int(fr['turnover'].sum())}   counter-attack seconds: {int(sec['counter'].sum())}   goals: {int(fr['goal'].sum())}")
    print(f"under pressure (of seconds with possession): {(sec['press_m'] < PRESSURE_M).sum() / sec['press_m'].notna().sum() * 100:.0f}%")
    print(f"mean nearest-defender distance: {sec['press_m'].mean():.1f} m   mean attack width: {sec['width_m'].mean():.1f} m   mean depth: {sec['depth_m'].mean():.1f} m")


if __name__ == "__main__":
    main()