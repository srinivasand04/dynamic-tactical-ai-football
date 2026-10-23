"""Is the passing gap between real and simulated football just the simulator's speed?

The simulated players run much faster than real ones (see the realism scorecard). The pass models were learned on real
players, so a fast simulated runner looks 'off the scale' to them, and some differences (who is predicted, the lane time margin,
where the receiver will be) may be caused by speed alone. This script repeats the comparison after slowing the simulated
players down to real speed:

    scale = real in-play median speed of outfield players / simulated in-play median speed (per simulator setting)
    every simulated velocity is multiplied by the scale before the pass features are built (positions are not touched).

Some measures do not use velocity at all (pass length, forward gain, completion rate). They cannot change with this
check: if they differ, the difference is real football behaviour, not speed. The others are shown before and after.

Input : outputs/decision_quality.csv (the run with the speeds as they are), the models of fit_pass_model.py
Output: the table, outputs/speed_check.csv
Usage :  python scripts\\speed_check.py
"""
import os

import numpy as np
import pandas as pd

import decision_quality as D
import fit_pass_model as F
import pass_features as P
import settings

SPEED_FREE = ["pass_length_m", "forward_gain_m", "completed_%"]
SHOW = ["pass_length_m", "forward_gain_m", "completed_%", "predicted_%", "predicted_top3_%", "arrives_best", "danger_best", "chose_best_%", "regret_%"]


def inplay_speed(m):
    """Median speed (m/s) of the outfield players while the ball is in play."""
    c = pd.read_csv(settings.csv(m), usecols=["role", "vx_ms", "vy_ms", "game_mode"])
    c = c[(c["role"] != "GK") & (c["game_mode"] == "normal")]
    sp = np.hypot(c["vx_ms"], c["vy_ms"])
    return float(sp[sp < 12].median())


def main():
    if not os.path.exists("outputs/decision_quality.csv"):
        raise SystemExit("run scripts/decision_quality.py first")
    base = pd.read_csv("outputs/decision_quality.csv", index_col="group")
    real = [m for m in settings.available(settings.REAL) if os.path.exists(settings.base(m) + "_passfeat.npz")]
    sim = [m for m in settings.available(settings.SIM) if os.path.exists(settings.base(m) + "_passfeat.npz")]
    v_real = float(np.mean([inplay_speed(m) for m in real]))
    speed = {m: inplay_speed(m) for m in sim}
    groups = sorted({settings.variant(m) for m in sim})
    scale = {g: v_real / float(np.mean([speed[m] for m in sim if settings.variant(m) == g])) for g in groups}
    print(f"in-play outfield speed: real {v_real:.2f} m/s")
    for g in groups:
        print(f"  {g:12s} simulated {v_real / scale[g]:.2f} m/s  ->  velocities multiplied by {scale[g]:.2f}")

    models = F.from_json()
    rows = []
    for m in sim:
        z = P.sim_match(m, scale[settings.variant(m)])
        d = {m: F.prepare(z)}
        n_ok = len(z["target"])
        rows.append({"group": settings.variant(m), **D.measure(m, d, models),
                     "completed_%": 100 * n_ok / max(n_ok + int(z["n_intercept"]), 1)})
    after = pd.DataFrame(rows).groupby("group")[D.MEASURES].mean()

    table = []
    for g in groups:
        for k in SHOW:
            r, b, a = float(base.loc["real", k]), float(base.loc[g, k]), float(after.loc[g, k])
            table.append({"group": g, "measure": k, "real": r, "simulated": b, "simulated_slowed": a,
                          "gap_%": 100 * abs(b - r) / abs(r), "gap_slowed_%": 100 * abs(a - r) / abs(r),
                          "uses_velocity": k not in SPEED_FREE})
    t = pd.DataFrame(table)
    os.makedirs("outputs", exist_ok=True)
    t.round(3).to_csv("outputs/speed_check.csv", index=False)
    print("\nREAL vs SIMULATED, before and after slowing the simulated players to real speed (group averages)\n")
    for g in groups:
        print(f"-- {g}")
        print(t[t["group"] == g].set_index("measure")[["real", "simulated", "simulated_slowed", "gap_%", "gap_slowed_%"]].round(2).to_string())
        print()
    s = t[t["uses_velocity"]].groupby("measure")[["gap_%", "gap_slowed_%"]].mean().round(1)
    print("average gap to real over the four settings, measures that use velocity:\n")
    print(s.to_string())
    free = t[~t["uses_velocity"]].groupby("measure")["gap_%"].mean().round(1)
    print("\nmeasures without velocity (cannot change):", free.to_dict())
    print("\nsaved outputs/speed_check.csv")


if __name__ == "__main__":
    main()
