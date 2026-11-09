"""Play a stretch of a real match in GRF (stage 2, GRF environment).

The stretch is cut into short windows. Each window starts from the real positions of the 22 players and the ball at
that moment, then the controller walks every GRF player along his real path for the window's length. Re-starting from
the real state at every window keeps the ball (which no controller steers yet) from drifting far.

Input : data/real/processed/<match>_segment_<start>.npz   (export_segment.py)
Output: outputs/grf_match_<match>_<start>.npz  (GRF and real tracks, for make_match_video.py) and a printed summary

Usage :  python scripts\\grf_match.py --segment data\\real\\processed\\J03WOH_segment_600.npz
         python scripts\\grf_match.py --segment ... --watch      (ONE GRF window, real speed; ~12 s to open, then plays)
         python scripts\\grf_match.py --segment ... --render     (GRF window per 3 s window: ~12 s to open each, slow)
"""
import argparse, os
import grf_names
grf_names.setup_from_args()                   # real player names in GRF's picture; before gfootball is imported
import cv2
import numpy as np
import grf_goal
from grf_follow import window
from grf_replay import FPS, L, W, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", required=True)
    ap.add_argument("--window", type=int, default=30, help="steps per window (10 steps = 1 s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--kick-s", type=float, default=None,
                    help="match time (s) of a real goal event: GRF's real shooter takes that shot himself (see grf_goal.py)")
    ap.add_argument("--record", default=None, help="save GRF's own rendered picture as this .mp4 (a GRF window flickers meanwhile)")
    ap.add_argument("--watch", action="store_true",
                    help="one GRF window for the whole stretch at real speed (no re-sync, so the ball drifts)")
    a = ap.parse_args()

    z = np.load(a.segment, allow_pickle=True)
    if a.watch:
        a.render, a.window = True, len(z["time_s"])        # a GRF window takes ~12 s to open, so open it once
    real = np.concatenate([z["home"], z["away"]], axis=1).transpose(1, 0, 2)       # (22, T, 2)
    ball_real = z["ball"]
    T = real.shape[1]
    grf = np.full_like(real, np.nan)
    ball = np.full_like(ball_real, np.nan)
    video = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720)) if a.record else None
    show = a.render or video is not None
    kick = None
    if a.kick_s is not None:                       # a goal inside the stretch: settings found by grf_goal.py (lead 3, centre, seed 1)
        grf_goal.SHOT_HOLD = 1
        m = grf_goal.find_kick(ball_real, z["time_s"], a.kick_s)
        kick = m - 3
        print(f"goal replay at {z['time_s'][m]:.1f} s (index {m})")
    s = 0
    while s < T - 1:
        if kick is not None and s == kick:
            pl, gb, scored, _, _, n = grf_goal.play(z, real, ball_real, m, 3, 40, "centre", 1, render=show, video=video)
            print(f"  goal window: {'GOAL scored in GRF' if scored else 'no goal'}", flush=True)
        else:
            steps = min(a.window, T - 1 - s)
            if kick is not None and s < kick < s + steps:       # end this window exactly where the goal window begins
                steps = kick - s
            snap = {"ball": np.clip(ball_real[s], [0.5, 0.5], [L - 0.5, W - 0.5]), "home": real[:11, s], "away": real[11:, s],
                    "role_home": z["role_home"], "role_away": z["role_away"]}
            pl, gb = window(snap, lambda t, s=s: real[:, min(s + t + 2, T - 1)], steps, a.seed, show, a.render, video)
        n = min(pl.shape[1], T - s)
        grf[:, s:s + n], ball[s:s + n] = pl[:, :n], gb[:n]
        if (s // a.window) % 20 == 0:
            print(f"  {z['time_s'][s]:7.1f} s  ({s}/{T} frames)", flush=True)
        s += max(n - 1, 1) if n > 1 else 1

    if video is not None:
        video.release()
        print("wrote", a.record)
    ok = ~np.isnan(grf[0, :, 0])
    err_p = np.linalg.norm(grf - real, axis=-1)[:, ok]
    err_b = np.linalg.norm(ball - ball_real, axis=-1)[ok]
    print(f"\nGRF vs the real stretch ({ok.sum() / FPS:.0f} s): players {err_p.mean():.2f} m mean error "
          f"(95% of players within {np.percentile(err_p, 95):.1f} m), ball {err_b.mean():.2f} m mean error")
    key = os.path.basename(a.segment).replace("_segment_", "_").replace(".npz", "")
    out = f"outputs/grf_match_{key}{'_watch' if a.watch else ''}.npz"      # never overwrite the re-synced run
    np.savez_compressed(out, grf=grf, ball_grf=ball, real=real, ball_real=ball_real, time_s=z["time_s"],
                        live=z["live"], window=a.window)
    print("wrote", out)


if __name__ == "__main__":
    main()
