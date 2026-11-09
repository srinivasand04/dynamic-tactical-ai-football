"""Play a real stretch in GRF WITH the real passes: the ball is no longer placed every half second, GRF's physics plays it.

The real kicks are found from the ball track (a jump in ball speed with a player at the ball). The stretch is then cut into
windows that each begin 2 steps before a real kick. A window starts from the real positions and ball, walks all 22 players
along their real paths (as grf_match.py does), and at the kick the real kicker turns towards where the real ball went and
presses pass (short under 20 m, long beyond) or, for the kick that ends in the goal, shoot. GRF's pass is assisted: the ball goes
to the teammate in the pressed direction, with a speed that fits the distance (checked: 24 of 24 test passes reached the target
teammate to within 0.4 m). After the kick the ball is GRF's until the next window.

Input : data/real/processed/<match>_segment_<start>.npz   (export_segment.py)
Output: outputs/grf_pass_<match>_<start>.npz  (same layout as grf_match.py, plus the kick table, for make_match_video.py)
        and, with --record, GRF's own rendered picture as an .mp4

Usage :  python scripts\\grf_pass.py --segment data\\real\\processed\\J03WOH_segment_3500.npz --kicks-only      (list the real kicks, no GRF)
         .venv\\Scripts\\python.exe scripts\\grf_pass.py --segment data\\real\\processed\\J03WOH_segment_3500.npz --goal-s 3612.6
         .venv\\Scripts\\python.exe scripts\\grf_pass.py --segment ... --goal-s 3612.6 --record outputs\\pass_3min_GRF.mp4
"""
import argparse, os
import numpy as np

FPS, L, W = 10, 105.0, 68.0                           # (not imported from grf_replay.py, which needs the GRF environment)

DIR_ACTION = [5, 4, 3, 2, 1, 8, 7, 6]                 # east, NE, N, NW, W, SW, S, SE (same as grf_follow.py, kept here so --kicks-only needs no GRF)
SHORT, LONG, SHOT = 11, 9, 12
FACE = 2                                              # steps in which the kicker turns to face the target before he presses the button
DELAY = 5                                             # steps from the button press until GRF's ball leaves the foot (measured)
SYNC_BALL, SYNC_PLAYERS = 2.0, 2.0                         # metres: a window may end when GRF's ball and players are this close to the real ones
WARM = 6                                              # hidden warm-up steps before every window (players are running when the picture starts)
ARRIVE = 4                                            # steps after the real ball arrives at which a kick window ends
LEAD = FACE + DELAY                                   # a window starts this many steps before the real kick, so the ball leaves on time


def detect_kicks(ball, real, min_gap=6):
    """Real kicks: frames where the ball starts to fly (speed up by > 5 m/s to > 7 m/s) with a player within 3.5 m of it."""
    T = len(ball)
    v = np.zeros(T)
    v[1:-1] = np.linalg.norm(ball[2:] - ball[:-2], axis=1) * FPS / 2          # speed centred on each frame, m/s
    kicks, t = [], 3
    while t < T - 6:
        jump = v[t + 1] - v[t - 2] > 5 and v[t + 1] > 7                          # the ball speeds up sharply (also inside a flight: a one-touch pass)
        start = v[t + 1] > 6 and v[max(t - 5, 0):t].max() < 4                    # or it leaves a ball that was (nearly) still
        if jump or start:
            t0 = t
            while t0 > 1 and v[t0 - 1] > v[t0 - 2] + 0.5 and v[t0 - 1] > 2.5:   # back to where the speed starts to rise
                t0 -= 1
            d = np.linalg.norm(real[:, t0] - ball[t0], axis=1)
            k = int(d.argmin())
            if d[k] < 4.5 and (not kicks or t0 >= kicks[-1]["t"] + min_gap):
                kicks.append({"t": t0, "kicker": k})
                t = max(t + 1, t0 + min_gap)                                       # the next kick may begin while this ball is still flying
                continue                                                           # (never step back: that found the same kick forever)
        t += 1
    for i, kk in enumerate(kicks):                                                # where each flight ends: the ball slows down, or the next kick
        t0, k = kk["t"], kk["kicker"]
        limit = kicks[i + 1]["t"] if i + 1 < len(kicks) else T - 1
        e = t0 + 2
        while e < limit and e < T - 1 and e - t0 < 80 and v[e] > 4:
            e += 1
        end, seg = ball[e], ball[t0:e + 1]
        dr = np.linalg.norm(real[:, e] - end, axis=1)
        r = int(dr.argmin())
        kk.update({"end": e, "end_xy": end, "receiver": r, "recv_m": float(dr[r]), "len_m": float(np.linalg.norm(end - ball[t0])),
                   "goal": bool(np.any(((seg[:, 0] < 0) | (seg[:, 0] > L)) & (abs(seg[:, 1] - W / 2) < 3.66))),
                   "same_team": (r < 11) == (k < 11)})
    return [k for k in kicks if k["len_m"] > 3.0]                                  # a nudge of the ball is not a pass


def banner(frame, t_match, a):
    """Cover GRF's own scoreboard (its clock restarts with every window) with the real match clock."""
    import cv2
    cv2.rectangle(frame, (150, 4), (700, 56), (25, 25, 25), -1)
    clock = t_match - a.p2_start_s + 45 * 60 if (a.p2_start_s and t_match >= a.p2_start_s) else t_match
    cv2.putText(frame, f"{int(clock // 60):02d}:{int(clock % 60):02d}", (165, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{a.home} v {a.away}", (275, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "GRF replay of real tracking data", (275, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
    return frame


def face_action(frm, to):
    ang = np.degrees(np.arctan2(to[1] - frm[1], to[0] - frm[0]))
    return DIR_ACTION[int(round(ang / 45.0)) % 8]


def overrides_for(kicks, s0, e0, goal_act):
    """Button presses for every kick that falls in [s0, e0): turn for FACE steps, press DELAY steps before the real kick."""
    ov = {}
    for k in kicks:
        press = k["t"] - DELAY - s0
        if press - FACE < 0 or k["t"] >= e0:
            continue
        who = k["kicker"]
        if k["goal"]:
            face, act = goal_act(who), SHOT
        else:
            tgt = k["recv_xy"] if k["same_team"] and k["recv_m"] < 3 else k["end_xy"]
            face, act = face_action(k["ball_xy"], tgt), (LONG if k["len_m"] > 20 else SHORT)
        for t in range(press - FACE, press):
            ov.setdefault(t, {})[who] = face
        ov.setdefault(press, {})[who] = act
    return ov


def continuous(a, z, real, ball_real, kicks, t_s, grf, ball, video, show):
    """One GRF game from the start of the stretch to the goal, a second one from the kick-off after it."""
    import grf_follow, grf_goal
    from grf_follow import window
    T = real.shape[1]
    for k in kicks:
        k["recv_xy"], k["ball_xy"] = real[k["receiver"], k["t"]], ball_real[k["t"]]
    goal_act = lambda who: grf_goal.aim_action(who < 11, "centre")
    grf_goal.SHOT_HOLD = 1
    goals = [k for k in kicks if k["goal"]]
    part_ends = [(goals[0]["t"] + 60) if goals else T - 1]
    parts = [(0, part_ends[0])]
    if goals:
        after = np.arange(goals[0]["t"] + 100, T - 20)           # the real kick-off: ball back near the centre spot, play live
        centre = [i for i in after if z["live"][i] and np.linalg.norm(ball_real[i] - [L / 2, W / 2]) < 4]
        s2 = int(centre[0]) if centre else goals[0]["t"] + 150
        if T - 1 - s2 > 30:
            parts.append((s2, T - 1))
    grf_goal.SHOT_HOLD = 1
    for pi, (s0, e0) in enumerate(parts):
        steps = e0 - s0
        snap = {"ball": np.clip(ball_real[s0], [0.5, 0.5], [L - 0.5, W - 0.5]), "home": real[:11, s0], "away": real[11:, s0],
                "role_home": z["role_home"], "role_away": z["role_away"]}
        ov = overrides_for(kicks, s0, e0, goal_act)
        ks = [k for k in kicks if s0 <= k["t"] - DELAY - FACE and k["t"] < e0]

        def targets(t, s0=s0, ks=ks):
            tg = real[:, min(s0 + t + 2, T - 1)].copy()
            for k in ks:                                          # the kicker goes to the ball and stays on it until he has kicked it
                if k["t"] - s0 - DELAY - FACE - 6 <= t <= k["t"] - s0:
                    tg[k["kicker"]] = ball_real[min(s0 + t + 2, T - 1)]
            return tg

        seed = a.seed
        if pi == 0 and goals:                                     # GRF decides whether the shot scores: find a seed that does (no picture yet)
            for sd in range(a.seed, a.seed + 12):
                window(snap, targets, steps, sd, False, False, None, overrides=ov)
                if grf_follow.LAST["scored"]:
                    seed = sd
                    break
            print(f"  game 1: seed {seed}, " + ("GRF scores" if grf_follow.LAST["scored"] else "no goal in 12 seeds"), flush=True)
        pl, gb = window(snap, targets, steps, seed, show, a.render, video, overrides=ov)
        n = min(pl.shape[1], T - s0)
        grf[:, s0:s0 + n], ball[s0:s0 + n] = pl[:, :n], gb[:n]
        print(f"  game {pi + 1}: {t_s[s0]:.1f} s to {t_s[min(s0 + n - 1, T - 1)]:.1f} s ({n} steps)", flush=True)
    if video is not None:
        video.release()
        print("wrote", a.record)
    ok = ~np.isnan(grf[0, :, 0])
    err_b = np.linalg.norm(ball - ball_real, axis=-1)
    err_p = np.linalg.norm(grf - real, axis=-1).mean(0)
    print(f"\nGRF vs the real stretch ({ok.sum() / FPS:.0f} s played): players {np.nanmean(err_p):.2f} m, ball {np.nanmean(err_b):.2f} m mean error")
    for q in range(0, T, 300):
        sl = slice(q, min(q + 300, T))
        if ok[sl].any():
            print(f"  {t_s[q] - t_s[0]:5.0f}-{t_s[min(q + 299, T - 1)] - t_s[0]:5.0f} s into the stretch: players {np.nanmean(err_p[sl]):5.2f} m, ball {np.nanmean(err_b[sl]):5.2f} m")
    key = os.path.basename(a.segment).replace("_segment_", "_").replace(".npz", "")
    out = f"outputs/grf_pass_{key}_continuous.npz"
    np.savez_compressed(out, grf=grf, ball_grf=ball, real=real, ball_real=ball_real, time_s=t_s, live=z["live"], window=0,
                        kick_t=np.array([k["t"] for k in kicks]), kick_who=np.array([k["kicker"] for k in kicks]))
    print("wrote", out)


def main():
    import grf_names
    grf_names.setup_from_args()                   # real player names in GRF's picture; before gfootball is imported
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", required=True)
    ap.add_argument("--goal-s", type=float, default=None, help="match time (s) of a real goal event inside the stretch")
    ap.add_argument("--tail", type=int, default=6, help="steps a kick window may run on after the real ball arrived, waiting for GRF to catch up")
    ap.add_argument("--min-window", type=int, default=6, help="shortest carry window in steps (10 = 1 s)")
    ap.add_argument("--max-window", type=int, default=10, help="longest carry window in steps (10 = 1 s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--home", default="Home", help="team names written on the banner")
    ap.add_argument("--away", default="Away")
    ap.add_argument("--p2-start-s", type=float, default=0.0, help="tracking time (s) at which the second half starts (match clock 45:00)")
    ap.add_argument("--kicks-only", action="store_true")
    ap.add_argument("--continuous", action="store_true",
                    help="ONE GRF game for the whole stretch (no re-sync: one camera, one clock); it ends at the goal and a second game "
                         "continues from the real kick-off. Ball and players may drift from the real ones, the picture never cuts")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--record", default=None, help="save GRF's own rendered picture as this .mp4")
    a = ap.parse_args()

    z = np.load(a.segment, allow_pickle=True)
    real = np.concatenate([z["home"], z["away"]], axis=1).transpose(1, 0, 2)
    ball_real, t_s = z["ball"], z["time_s"]
    T = real.shape[1]
    kicks = detect_kicks(ball_real, real)
    if a.goal_s is not None:
        import grf_goal
        m = grf_goal.find_kick(ball_real, t_s, a.goal_s)
        for k in kicks:
            if abs(k["t"] - m) <= 5:
                k["goal"] = True
    print(f"{len(kicks)} real kicks found in {T / FPS:.0f} s; {sum(k['same_team'] and k['recv_m'] < 3 for k in kicks)} look like completed passes, "
          f"{sum(k['goal'] for k in kicks)} end in a goal")
    for k in kicks[:12]:
        print(f"  {t_s[k['t']]:7.1f} s  kicker {k['kicker']:2d}  flight {k['len_m']:5.1f} m  receiver {k['receiver']:2d}"
              f"{' (same team)' if k['same_team'] else ' (opponent)'}  {'GOAL' if k['goal'] else ''}")
    if a.kicks_only:
        return

    import cv2
    import grf_goal
    import grf_follow
    from grf_follow import window
    grf = np.full_like(real, np.nan)
    ball = np.full_like(ball_real, np.nan)
    video = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720)) if a.record else None
    show = a.render or video is not None
    if a.continuous:
        continuous(a, z, real, ball_real, kicks, t_s, grf, ball, video, show)
        return
    # Windows end where the picture will not jump. A kick window starts LEAD steps before the real kick and may end only after the
    # real ball has arrived; a carry window runs at least --min-window steps. In both cases the window then goes on until GRF's ball
    # and players are close to the real ones (SYNC_BALL / SYNC_PLAYERS), or until its latest allowed end, and only then is the game
    # re-created from the real state. A re-sync at a moment of agreement cannot be seen; one in the middle of a pass was the
    # 'repeating frames' (the ball flew on, then was pulled back).
    grf_goal.SHOT_HOLD = 1
    ks = [k for k in kicks if k["t"] - LEAD >= 0]
    s, n, ki, wlog = 0, 0, 0, []
    while s < T - 1:
        while ki < len(ks) and ks[ki]["t"] - LEAD < s:
            ki += 1                                        # a kick that falls inside the last window is left to the ball track
        k = ks[ki] if ki < len(ks) and ks[ki]["t"] - LEAD <= s + 1 else None
        if k is not None:
            s, lead = max(s, k["t"] - LEAD), LEAD
            nxt = ks[ki + 1]["t"] - LEAD if ki + 1 < len(ks) else T - 1
            lo = max(k["end"] + ARRIVE, k["t"] + 5) - s
            hi = min(nxt - s, lo + a.tail, T - 1 - s)
            lo = min(lo, hi)
            ki += 1
        else:
            lead = 0
            nxt = ks[ki]["t"] - LEAD if ki < len(ks) else T - 1
            hi = min(a.max_window, nxt - s, T - 1 - s)
            lo = min(a.min_window, hi)
        steps = hi
        if steps < 1:
            s += 1
            continue

        wm = min(WARM, s)                                     # hidden warm-up: the game starts WARM steps early so nobody starts standing still
        sw = s - wm

        def stop(t, pl_t, ball_t, sw=sw, lo=wm + lo, hi=wm + hi):
            if t >= hi:
                return True
            if t < lo:
                return False
            return (np.linalg.norm(ball_t - ball_real[sw + t]) < SYNC_BALL
                    and np.linalg.norm(pl_t - real[:, sw + t], axis=1).mean() < SYNC_PLAYERS)

        snap = {"ball": np.clip(ball_real[sw], [0.5, 0.5], [L - 0.5, W - 0.5]), "home": real[:11, sw], "away": real[11:, sw],
                "role_home": z["role_home"], "role_away": z["role_away"]}
        ov, who, face, act = {}, None, None, None
        if k is not None:
            who = k["kicker"]
            if k["goal"]:
                face, act = grf_goal.aim_action(who < 11, "centre"), SHOT
            else:
                tgt = real[k["receiver"], k["t"]] if k["same_team"] and k["recv_m"] < 3 else k["end_xy"]
                face, act = face_action(ball_real[k["t"]], tgt), (LONG if k["len_m"] > 20 else SHORT)

        def make(press_at, who=who, face=None, act=None):
            o = {}
            if who is not None:
                for t in range(max(0, press_at - FACE), press_at):
                    o[t] = {who: face}
                o[press_at] = {who: act}
            return o

        def targets(t, sw=sw, who=who, lead=wm + lead):
            tg = real[:, min(sw + t + 2, T - 1)].copy()
            if who is not None and t <= lead:                    # the kicker stays on the ball until he has kicked it
                tg[who] = ball_real[min(sw + t + 2, T - 1)]
            return tg

        hook = (lambda t, fr, sw=sw: banner(fr, t_s[min(sw + t, T - 1)], a)) if video is not None else None
        steps = wm + hi
        if k is not None and k["goal"]:                          # GRF decides whether the shot scores: find a run that does (no picture yet)
            best = (wm + FACE, a.seed)
            for press_at, seed in [(wm + p, sd) for p in (FACE, FACE - 1, FACE + 1) for sd in (a.seed, a.seed + 1, a.seed + 2)]:
                window(snap, targets, steps, seed, False, False, None, overrides=make(press_at, face=face, act=act))
                if grf_follow.LAST["scored"]:
                    best = (press_at, seed)
                    break
            ov, seed_use = make(best[0], face=face, act=act), best[1]
            print(f"  goal window at {t_s[s]:.1f} s: " + ("GRF scores" if grf_follow.LAST["scored"] else "no goal in any try"), flush=True)
        else:
            ov, seed_use = make(wm + FACE, face=face, act=act), a.seed
        pl, gb = window(snap, targets, steps, seed_use, show, a.render, video, overrides=ov, stop=stop, skip=wm, frame_hook=hook)
        pl, gb = pl[:, wm:], gb[wm:]                          # the warm-up steps are not part of the clip
        nn = min(pl.shape[1], T - s)
        grf[:, s:s + nn], ball[s:s + nn] = pl[:, :nn], gb[:nn]
        if n % 20 == 0:
            print(f"  window {n}  {t_s[s]:7.1f} s  ({nn - 1} steps)", flush=True)
        n += 1
        wlog.append((s, nn - 1, float(np.linalg.norm(gb[nn - 1] - ball_real[min(s + nn - 1, T - 1)])),
                     float(np.linalg.norm(pl[:, nn - 1] - real[:, min(s + nn - 1, T - 1)], axis=1).mean()), k is not None))
        s = s + max(nn - 1, 1)
        if k is not None and k["goal"] and grf_follow.LAST["scored"]:
            s = max(s, k["end"] + 10)                      # GRF has scored: carry on after the real ball has gone in, do not replay the shot
    if video is not None:
        video.release()
        print("wrote", a.record)
    w = np.array(wlog, float)
    if len(w):
        synced = (w[:, 2] < SYNC_BALL) & (w[:, 3] < SYNC_PLAYERS)
        print(f"\n{len(w)} windows, median length {np.median(w[:, 1]) / FPS:.1f} s; the re-sync was invisible (ball < {SYNC_BALL:.0f} m and players < {SYNC_PLAYERS:.0f} m) "
              f"in {synced.mean() * 100:.0f}% of them; ball jump at the other re-syncs: median {np.median(w[~synced, 2]) if (~synced).any() else 0:.1f} m, "
              f"max {w[:, 2].max():.0f} m")

    ok = ~np.isnan(grf[0, :, 0])
    err_p = np.linalg.norm(grf - real, axis=-1)[:, ok]
    err_b = np.linalg.norm(ball - ball_real, axis=-1)[ok]
    print(f"\nGRF vs the real stretch ({ok.sum() / FPS:.0f} s): players {err_p.mean():.2f} m, ball {err_b.mean():.2f} m mean error")
    rows = []                                              # the passes: where is GRF's ball when the real ball arrives, and who is nearest to it?
    for k in kicks:
        e = k["end"]
        if e < T and not np.isnan(ball[e, 0]) and not np.isnan(grf[0, e, 0]):
            near = int(np.linalg.norm(grf[:, e] - ball[e], axis=1).argmin())
            rows.append((float(np.linalg.norm(ball[e] - ball_real[e])), near == k["receiver"], (near < 11) == (k["receiver"] < 11)))
    if rows:
        r = np.array(rows, float)
        print(f"{len(rows)} kicks scored: GRF ball {np.median(r[:, 0]):.1f} m median ({r[:, 0].mean():.1f} m mean) from the real ball when it arrived; "
              f"same receiver {r[:, 1].mean() * 100:.0f}%, same team {r[:, 2].mean() * 100:.0f}%")
    key = os.path.basename(a.segment).replace("_segment_", "_").replace(".npz", "")
    out = f"outputs/grf_pass_{key}.npz"
    np.savez_compressed(out, grf=grf, ball_grf=ball, real=real, ball_real=ball_real, time_s=t_s, live=z["live"], window=a.max_window,
                        kick_t=np.array([k["t"] for k in kicks]), kick_who=np.array([k["kicker"] for k in kicks]))
    print("wrote", out)


if __name__ == "__main__":
    main()
