"""Side-by-side video: the real match (left) and the same stretch played in GRF (right).

Input : outputs/grf_match_<match>_<start>.npz   (grf_match.py)
Output: outputs/grf_match_<match>_<start>.mp4

Usage :  python scripts\\make_match_video.py --run outputs\\grf_match_J03WOH_600.npz
"""
import argparse
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

L, W = 105.0, 68.0


def pitch(ax, title):
    ax.set_xlim(-2, L + 2), ax.set_ylim(-2, W + 2), ax.set_aspect("equal"), ax.axis("off"), ax.set_title(title)
    ax.add_patch(plt.Rectangle((0, 0), L, W, fill=False, color="0.4"))
    ax.plot([L / 2, L / 2], [0, W], color="0.6", lw=1)
    ax.add_patch(plt.Circle((L / 2, W / 2), 9.15, fill=False, color="0.6"))
    for x0 in (0, L - 16.5):
        ax.add_patch(plt.Rectangle((x0, W / 2 - 20.15), 16.5, 40.3, fill=False, color="0.6"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--step", type=int, default=1, help="draw every n-th frame")
    ap.add_argument("--mark-s", type=float, default=None, help="match time (s) of an event, e.g. a shot; also saves a still of it")
    a = ap.parse_args()

    z = np.load(a.run, allow_pickle=True)
    seg_path = re.sub(r"^grf_[a-z]+_", "", os.path.basename(a.run)).replace("_watch", "").replace(".npz", "")
    seg_path = "data/real/processed/" + seg_path.replace("_", "_segment_", 1) + ".npz"
    seg = np.load(seg_path, allow_pickle=True) if os.path.exists(seg_path) else None
    nums = np.concatenate([seg["jersey_home"], seg["jersey_away"]]) if seg is not None and "jersey_home" in seg.files else None
    names = np.concatenate([seg["name_home"], seg["name_away"]]) if nums is not None else None
    real, grf, br, bg, t = z["real"], z["grf"], z["ball_real"], z["ball_grf"], z["time_s"]
    T = real.shape[1]
    if a.mark_s is not None:                                           # real vs GRF at the start, at the event, and at the end
        m = int(np.argmin(abs(t - a.mark_s)))
        picks = [(0, "start"), (m, "shot"), (T - 1, "end")]
        sf, sa = plt.subplots(2, 3, figsize=(15, 7.5), facecolor="white")
        for c, (i, label) in enumerate(picks):
            for r, (arr, b, name) in enumerate(((real, br, "Real"), (grf, bg, "GRF"))):
                pitch(sa[r, c], f"{name}  t = {t[i]:.1f} s  ({label})")
                sa[r, c].scatter(arr[:11, i, 0], arr[:11, i, 1], c="#d62728", s=55, zorder=3)
                sa[r, c].scatter(arr[11:, i, 0], arr[11:, i, 1], c="#1f77b4", s=55, zorder=3)
                sa[r, c].scatter([b[i, 0]], [b[i, 1]], c="k", s=45, zorder=4)
                if nums is not None:
                    for k in range(22):
                        sa[r, c].text(arr[k, i, 0] + 1.0, arr[k, i, 1] + 1.0, str(nums[k]), fontsize=7)
        sf.tight_layout()
        still = a.run.replace(".npz", "_shot.png")
        sf.savefig(still, dpi=110)
        plt.close(sf)
        print("wrote", still)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), facecolor="white")
    who = ""
    if seg is not None and "name_home" in seg.files:
        who = f"  (red = home, blue = away; numbers = shirt numbers)"
    pitch(axs[0], "REAL DATA (DFL tracking)" + who), pitch(axs[1], "GRF SIMULATION (same stretch)")
    sc = []
    for ax, arr, b in ((axs[0], real, br), (axs[1], grf, bg)):
        h = ax.scatter(arr[:11, 0, 0], arr[:11, 0, 1], c="#d62728", s=60, zorder=3)
        v = ax.scatter(arr[11:, 0, 0], arr[11:, 0, 1], c="#1f77b4", s=60, zorder=3)
        bl = ax.scatter([b[0, 0]], [b[0, 1]], c="k", s=35, zorder=4)
        sc.append((h, v, bl))
    txt = fig.text(0.5, 0.02, "", ha="center")
    labels = [[ax.text(0, 0, str(nums[k]) if nums is not None else "", fontsize=7, zorder=5) for k in range(22)] for ax in axs]
    shooter, ring = None, [ax.scatter([], [], s=260, facecolors="none", edgecolors="gold", linewidths=2, zorder=2) for ax in axs]
    if a.mark_s is not None and "owner" in z.files or (a.mark_s is not None and seg is not None):
        m = int(np.argmin(abs(t - a.mark_s)))
        side = 0 if (seg is not None and str(seg["owner"][m]) != "away") else 11     # attacking team = ball owner at the shot
        shooter = side + int(np.argmin(np.linalg.norm(real[side:side + 11, m] - br[m], axis=-1)))

    def draw(i):
        for (h, v, bl), arr, b in zip(sc, (real, grf), (br, bg)):
            h.set_offsets(arr[:11, i]), v.set_offsets(arr[11:, i]), bl.set_offsets(b[i][None])
        for ax_i, arr in enumerate((real, grf)):
            for k in range(22):
                labels[ax_i][k].set_position((arr[k, i, 0] + 1.0, arr[k, i, 1] + 1.0))
            if shooter is not None and a.mark_s is not None and -3 <= t[i] - a.mark_s <= 3:
                ring[ax_i].set_offsets(arr[shooter, i][None])
            else:
                ring[ax_i].set_offsets(np.empty((0, 2)))
        e = np.linalg.norm(grf[:, i] - real[:, i], axis=-1).mean()
        live = "open play" if z["live"][i] else "ball out of play"
        shot = "   <<< SHOT" if a.mark_s is not None and 0 <= t[i] - a.mark_s <= 0.6 else ""
        who_s = f"  |  gold ring = shooter #{nums[shooter]} {names[shooter]}" if shooter is not None and nums is not None and -3 <= t[i] - a.mark_s <= 3 else ""
        txt.set_text(f"{t[i] - t[0]:6.1f} s into the stretch  |  {live}  |  GRF players {e:.1f} m from the real ones{shot}{who_s}")
        return []

    frames = [i for i in range(0, T, a.step) if not np.isnan(grf[0, i, 0])]
    out = a.run.replace(".npz", ".mp4")
    animation.FuncAnimation(fig, draw, frames=frames).save(out, writer=animation.FFMpegWriter(fps=a.fps // a.step * 1 or 1))
    print("wrote", out)


if __name__ == "__main__":
    main()
