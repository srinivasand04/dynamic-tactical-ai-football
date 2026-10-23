"""Describe the data-driven boundaries and sequences in football language, and test how well they agree.

The boundaries come from geometry only (find_sequences.py). Here, AFTERWARDS, we look up what the match
recorded around each boundary:
  both sources : ball dead / play restarts, set-piece mode, goal, possession change, start of a half
  real match   : plus events file: shot, foul, ball out, substitution, card, set piece taken, recovery, tackle
  simulator    : no event log, so only what the simulator reports (game mode, ball owner, score)

Agreement is compared with RANDOM boundaries (same number, random times), so a high number that is
only chance (e.g. passes happen every few seconds) is not mistaken for a result.

Input : common-format CSV, its _sequences.csv, optionally the real events file
Output: <name>_boundary_events.csv, <name>_sequence_desc.csv, a printed summary

Usage :  python scripts\\describe_sequences.py --csv data\\real\\processed\\J03WMX.csv.gz --events data\\real\\raw\\J03WMX_events.csv.gz
         python scripts\\describe_sequences.py --csv data\\simulated\\match_001.csv
"""
import argparse
import numpy as np
import pandas as pd
import settings

STRONG, WEAK = "strong", "weak"


def base_name(csv):
    return csv.replace(".csv.gz", "").replace(".csv", "")


def common_events(d):
    """(time, label, strength) from what both the real file and the simulator report."""
    ev = []
    t = d["time_s"].to_numpy()
    mode = d["game_mode"].to_numpy()
    for i in range(1, len(d)):
        if mode[i] != mode[i - 1]:
            if mode[i - 1] == "normal":
                label = "ball dead" if mode[i] == "dead" else f"{str(mode[i]).replace('_', ' ')} (set piece)"
            elif mode[i] == "normal":
                label = "play restarts"
            else:
                label = f"{mode[i - 1]} -> {mode[i]}"
            ev.append((t[i], label, STRONG))
    own = d["ball_owner_team"].to_numpy()
    last = None
    for i in range(len(d)):
        if own[i] in ("home", "away"):
            if last is not None and own[i] != last:
                ev.append((t[i], f"possession -> {own[i]}", WEAK))
            last = own[i]
    score = (d["score_home"] + d["score_away"]).to_numpy()
    for i in np.flatnonzero(np.diff(score) > 0) + 1:
        ev.append((t[i], "goal", STRONG))
    if "period" in d.columns:
        per = d["period"].to_numpy()
        for i in np.flatnonzero(np.diff(per) != 0) + 1:
            ev.append((t[i], "second half starts", STRONG))
    return ev


def real_events(events_csv, d):
    offset2 = d.loc[d["period"] == 2, "time_s"].min()          # second-half clock starts here
    e = pd.read_csv(events_csv)
    e["time_s"] = pd.to_timedelta(e["timestamp"]).dt.total_seconds() + np.where(e["period_id"] == 2, offset2, 0.0)
    ev = []
    for r in e.itertuples():
        et, sp = r.event_type, r.set_piece_type
        if isinstance(sp, str):
            ev.append((r.time_s, f"{sp.lower().replace('_', ' ')} taken", STRONG))
        elif et == "SHOT":
            ev.append((r.time_s, f"shot ({str(r.result).lower().replace('_', ' ')})", STRONG))
        elif et == "FOUL_COMMITTED":
            ev.append((r.time_s, "foul", STRONG))
        elif et == "BALL_OUT":
            ev.append((r.time_s, "ball out", STRONG))
        elif et == "SUBSTITUTION":
            ev.append((r.time_s, "substitution", STRONG))
        elif et == "CARD":
            ev.append((r.time_s, "card", STRONG))
        elif et == "RECOVERY":
            ev.append((r.time_s, "recovery", WEAK))
        elif et == "GENERIC:TacklingGame":
            ev.append((r.time_s, "tackle", WEAK))
    return ev


def near(times, ref, tol):
    """True for each time that has a reference time within tol seconds."""
    if len(ref) == 0:
        return np.zeros(len(times), bool)
    ref = np.sort(ref)
    i = np.searchsorted(ref, times)
    left = np.abs(times - ref[np.clip(i - 1, 0, len(ref) - 1)])
    right = np.abs(times - ref[np.clip(i, 0, len(ref) - 1)])
    return np.minimum(left, right) <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--events", default=None)
    ap.add_argument("--tol", type=float, default=settings.get("sequences", "event_tolerance_s", default=2.0))       # seconds around a boundary
    a = ap.parse_args()

    name = base_name(a.csv)
    cols = ["frame", "time_s", "game_mode", "ball_owner_team", "score_home", "score_away", "ball_x_m"]
    d = pd.read_csv(a.csv, usecols=cols + (["period"] if a.events else [])).drop_duplicates("frame").reset_index(drop=True)
    seq = pd.read_csv(name + "_sequences.csv")
    T = d["time_s"].iloc[-1]

    ev = common_events(d) + (real_events(a.events, d) if a.events else [])
    ev = pd.DataFrame(ev, columns=["time_s", "label", "strength"]).sort_values("time_s").reset_index(drop=True)
    tb = seq["start_time_s"].iloc[1:].to_numpy()               # boundaries (first sequence starts at 0)

    # ---- describe each boundary ----
    rows = []
    for t0 in tb:
        w = ev[(ev["time_s"] >= t0 - a.tol) & (ev["time_s"] <= t0 + a.tol)]
        labels = list(dict.fromkeys(w.sort_values("strength", ascending=False)["label"]))
        rows.append({"time_s": round(t0, 1), "explained_strong": bool((w["strength"] == STRONG).any()),
                     "explained_any": len(w) > 0, "football_description": "; ".join(labels[:4])})
    be = pd.DataFrame(rows)
    be.to_csv(name + "_boundary_events.csv", index=False)

    # ---- agreement against random boundaries ----
    strong_t = ev.loc[ev["strength"] == STRONG, "time_s"].to_numpy()
    any_t = ev["time_s"].to_numpy()
    rng = np.random.default_rng(0)
    rand = [rng.uniform(0, T, len(tb)) for _ in range(300)]
    dead_t = ev.loc[ev["label"].isin(["ball dead", "play restarts"]) | ev["label"].str.contains("set piece"), "time_s"].to_numpy()
    res = {
        "boundaries": len(tb),
        "explained_strong_%": round(100 * near(tb, strong_t, a.tol).mean(), 0),
        "random_strong_%": round(100 * np.mean([near(r, strong_t, a.tol).mean() for r in rand]), 0),
        "explained_any_%": round(100 * near(tb, any_t, a.tol).mean(), 0),
        "random_any_%": round(100 * np.mean([near(r, any_t, a.tol).mean() for r in rand]), 0),
        "strong_events": len(strong_t),
        "recalled_strong_%": round(100 * near(strong_t, tb, a.tol).mean(), 0),
        "random_recalled_%": round(100 * np.mean([near(strong_t, r, a.tol).mean() for r in rand]), 0),
        "dead_ball_changes": len(dead_t),
        "recalled_dead_%": round(100 * near(dead_t, tb, a.tol).mean(), 0) if len(dead_t) else np.nan,
        "random_dead_%": round(100 * np.mean([near(dead_t, r, a.tol).mean() for r in rand]), 0) if len(dead_t) else np.nan,
    }

    # ---- describe each sequence ----
    out = []
    t_arr = d["time_s"].to_numpy()
    for r in seq.itertuples():
        m = (t_arr >= r.start_time_s) & (t_arr < r.end_time_s)
        seg = d[m]
        if len(seg) == 0:
            continue
        owners = seg.loc[seg["ball_owner_team"].isin(["home", "away"]), "ball_owner_team"]
        team = owners.mode().iloc[0] if len(owners) else "nobody"
        share = round(100 * (owners == team).mean()) if len(owners) else 0
        sign = 1 if team == "home" else -1 if team == "away" else 0           # home attacks toward x = 105
        adv = sign * (seg["ball_x_m"].iloc[-1] - seg["ball_x_m"].iloc[0])
        still = 100 * (seg["game_mode"] != "normal").mean()
        start_d = "start of match" if r.seq_id == 0 else be.loc[r.seq_id - 1, "football_description"] or "no event nearby"
        end_d = "end of match" if r.seq_id == len(seq) - 1 else be.loc[r.seq_id, "football_description"] or "no event nearby"
        kind = "stoppage-like" if still >= 80 else f"{team} ball, {adv:+.0f} m toward goal" if team != "nobody" else "loose"
        out.append({"seq_id": r.seq_id, "start_s": round(r.start_time_s, 1), "end_s": round(r.end_time_s, 1),
                    "duration_s": round(r.duration_s, 1), "ball_dead_%": round(still), "possession": team,
                    "possession_%": share, "ball_advance_m": round(adv, 1), "summary": kind,
                    "starts_after": start_d, "ends_with": end_d})
    pd.DataFrame(out).to_csv(name + "_sequence_desc.csv", index=False)

    print(pd.Series(res).to_string())
    print(f"\nwrote {name}_boundary_events.csv and {name}_sequence_desc.csv")
    top = be["football_description"].replace("", "no event nearby").value_counts().head(6)
    print("\nMost common descriptions at boundaries:")
    print(top.to_string())


if __name__ == "__main__":
    main()
