"""Trial of the open-source unravelsports package on our real DFL (Sportec) matches.

1. Pressing Intensity (Bekkers 2024): per frame, the chance that each defender can reach each attacker within 1.5 s.
   Team pressing score per frame = mean over attackers of the highest chance any defender reaches them.
2. EFPI formation detection per team and half.
3. Real-data check of the GRF tournament result ("team press lowers danger"): does the attacking team shoot less in the
   next 10 s when the pressing score is high? Compared by pressing quartile.

Runs in its own environment (.venv-unravel: unravelsports 1.2.1, kloppy 3.19.1). Writes outputs/unravel/ only.
Usage : .venv-unravel\\Scripts\\python.exe scripts\\unravel_trial.py [J03WOY ...] [--fps 5]
"""
import argparse
import glob
import os
import time

import numpy as np
import polars as pl
from kloppy import sportec
from unravel.soccer import KloppyPolarsDataset
from unravel.soccer.models import PressingIntensity
from unravel.soccer.models.formations.efpi import EFPI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "real", "raw_xml")
OUT = os.path.join(ROOT, "outputs", "unravel")
HORIZON = 10.0   # seconds ahead to look for a shot (same window as our threat model)


def files(mid):
    f = lambda p: glob.glob(os.path.join(RAW, f"{p}*{mid}.xml"))[0]
    return f("DFL_02_01_matchinformation"), f("DFL_04_03_positions_raw_observed"), f("DFL_03_02_events_raw")


def team_pressing(pi_out):
    """Per frame: mean over attackers (columns) of the max probability any defender (row) reaches them."""
    pti = pi_out["probability_to_intercept"].to_list()
    score = [float(np.mean(np.max(np.asarray(m, float), axis=0))) if m else np.nan for m in pti]
    return pi_out.select("period_id", "frame_id", "timestamp").with_columns(pl.Series("press", score))


def shots_table(meta, ev):
    events = sportec.load_event(event_data=ev, meta_data=meta)
    rows = []
    for e in events.events:
        if e.event_type.name == "SHOT":
            rows.append((e.period.id, e.timestamp.total_seconds(), e.team.team_id))
    return rows, {t.team_id: t.name for t in events.metadata.teams}


def run(mid, fps):
    t0 = time.time()
    meta, pos, ev = files(mid)
    kd = sportec.load_tracking(meta_data=meta, raw_data=pos, sample_rate=fps / 25, only_alive=True)
    ds = KloppyPolarsDataset(kloppy_dataset=kd)
    keep = ds.data.group_by("frame_id").agg(pl.col("is_ball_carrier").any().alias("has")).filter(pl.col("has"))["frame_id"]
    ds.data = ds.data.filter(pl.col("frame_id").is_in(keep.implode()))   # pressing needs a known ball carrier
    df = ds.data
    print(f"{mid}: {df['frame_id'].n_unique():,} frames at {fps}/s loaded in {time.time() - t0:.0f}s", flush=True)

    # who is attacking in each frame
    own = df.filter(pl.col("team_id") != "ball").group_by("frame_id").agg(pl.col("ball_owning_team_id").first().alias("att"))

    pi = PressingIntensity(dataset=ds)
    pi.fit(method="teams", ball_method="max", orient="pressing")
    ball = df.filter(pl.col("team_id") == "ball").select("frame_id", pl.col("x").alias("ball_x"), pl.col("y").alias("ball_y"))
    press = team_pressing(pi.output).join(own, on="frame_id", how="left").join(ball, on="frame_id", how="left")
    press = press.with_columns((pl.col("timestamp").dt.total_milliseconds() / 1000).alias("t"))
    print(f"  pressing intensity: {len(press):,} frames in {time.time() - t0:.0f}s", flush=True)

    shots, names = shots_table(meta, ev)
    S = pl.DataFrame(shots, schema=["period_id", "ts", "team"], orient="row")
    # shot by the attacking team within HORIZON s after the frame
    nxt = []
    for p, t, a in press.select("period_id", "t", "att").iter_rows():
        s = S.filter((pl.col("period_id") == p) & (pl.col("team") == a) & (pl.col("ts") > t) & (pl.col("ts") <= t + HORIZON))
        nxt.append(len(s) > 0)
    press = press.with_columns(pl.Series("shot_next_10s", nxt), pl.lit(mid).alias("match"))
    print(f"  shots in match: {len(S)}  ({time.time() - t0:.0f}s)", flush=True)

    ef = EFPI(dataset=ds)
    ef.fit(every="60m")   # one window per half; every="period" hits a join bug in unravelsports 1.2.1
    form = ef.output
    print(f"  formations done ({time.time() - t0:.0f}s)", flush=True)
    return press, form, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matches", nargs="*", default=["J03WOY"])
    ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--reuse", action="store_true", help="read the saved pressing_<match>.csv files instead of recomputing")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    allp = []
    if a.reuse:
        allp = [pl.read_csv(os.path.join(OUT, f"pressing_{m}.csv")) for m in a.matches]
        a.matches = []
    for mid in a.matches:
        press, form, names = run(mid, a.fps)
        allp.append(press)
        press.drop("timestamp").write_csv(os.path.join(OUT, f"pressing_{mid}.csv"))
        cols = [c for c in form.columns if c in ("period_id", "team_id", "formation", "is_attacking", "object_id", "position")]
        f = form.select(cols).unique(subset=[c for c in ("period_id", "team_id", "is_attacking") if c in cols]) if cols else form
        f = f.with_columns(pl.col("team_id").replace(names).alias("team")) if "team_id" in f.columns else f
        print(f"\n{mid} formations per half:\n{f.sort([c for c in ('period_id', 'team_id') if c in f.columns])}")
        form.write_parquet(os.path.join(OUT, f"formations_{mid}.parquet"))

    P = pl.concat(allp).drop_nulls("press").filter(pl.col("press").is_not_nan())
    q = np.quantile(P["press"].to_numpy(), [0.25, 0.5, 0.75])
    P = P.with_columns(pl.col("press").cut(list(q), labels=["Q1 low", "Q2", "Q3", "Q4 high"]).alias("quartile"))
    tab = P.group_by("quartile").agg(pl.len().alias("frames"), pl.col("press").mean().round(3).alias("press_mean"),
                                     (pl.col("shot_next_10s").mean() * 100).round(2).alias("shot_next_10s_%")).sort("quartile")
    print("\nAttacking team shoots in the next 10 s, by pressing quartile (real matches):")
    print(tab)
    lo, hi = P.filter(pl.col("quartile") == "Q1 low")["shot_next_10s"], P.filter(pl.col("quartile") == "Q4 high")["shot_next_10s"]
    d = (hi.mean() - lo.mean()) * 100
    print(f"\nhigh vs low pressing: {d:+.2f} percentage points ({hi.mean() * 100:.2f}% vs {lo.mean() * 100:.2f}%)")
    tab.write_csv(os.path.join(OUT, "pressing_vs_shots.csv"))

    # fair comparison: pressing is naturally higher near goal, where shots happen anyway -> compare within the same ball zone
    # (data is oriented so the ball-owning team attacks towards +x; pitch centred on 0)
    print(f"\nball x range: {P['ball_x'].min():.1f} .. {P['ball_x'].max():.1f} m")
    P = P.with_columns(pl.when(pl.col("ball_x") < -17.5).then(pl.lit("1 own third"))
                       .when(pl.col("ball_x") < 17.5).then(pl.lit("2 middle third")).otherwise(pl.lit("3 final third")).alias("zone"))
    rows = []
    for z in sorted(P["zone"].unique().to_list()):
        Z = P.filter(pl.col("zone") == z)
        qz = np.quantile(Z["press"].to_numpy(), [0.25, 0.75])
        lo = Z.filter(pl.col("press") <= qz[0])["shot_next_10s"].mean() * 100
        hi = Z.filter(pl.col("press") >= qz[1])["shot_next_10s"].mean() * 100
        n_lo, n_hi = (Z["press"] <= qz[0]).sum(), (Z["press"] >= qz[1]).sum()
        se = np.sqrt(lo / 100 * (1 - lo / 100) / n_lo + hi / 100 * (1 - hi / 100) / n_hi) * 100
        rows.append((z, len(Z), round(Z["shot_next_10s"].mean() * 100, 2), round(lo, 2), round(hi, 2), round(hi - lo, 2), round(1.96 * se, 2)))
    Zt = pl.DataFrame(rows, schema=["zone", "frames", "shot_%_all", "shot_%_low_press", "shot_%_high_press", "high_minus_low", "ci95_±"],
                      orient="row")
    print("\nWithin the same ball zone: shot in next 10 s, lowest vs highest quarter of pressing (frames are not independent:"
          " the interval is optimistic):")
    print(Zt)
    Zt.write_csv(os.path.join(OUT, "pressing_vs_shots_by_zone.csv"))
    print(f"saved {os.path.relpath(OUT, ROOT)}/")


if __name__ == "__main__":
    main()
