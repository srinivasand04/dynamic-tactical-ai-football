"""Phase A3: detector + tracker on full clips, scored against SoccerNet's true track IDs; writes a positions file per clip.

Per clip (frames every --step-th, from data/soccernet_gsr_full/test/<clip>/img1):
  RF-DETR persons -> ByteTrack (supervision) -> our pitch mapping (pitch_calib.solve) -> shirt-colour team (2 groups)
Scores:
  coverage     share of true person boxes matched by a tracked box (IoU >= 0.5)
  id_switches  times a true player's matched track ID changes (lower is better), also per true player per minute
  idf1_like    share of each true player's frames carrying its most common track ID (1.0 = one ID throughout)
  pos_m        mean position error (m) of tracked players with our mapping vs the true pitch position
Output: outputs/vision/tracks/<clip>_positions.csv (frame, time_s, track_id, team, x_m, y_m), outputs/vision/track_eval.csv
--detector rfdetr_sn (opt-in; default unchanged) uses the football-trained RF-DETR SoccerNet: referees are kept out of the
  team-colour split (team "ref"), each track gets its majority role, and the best ball box per frame is mapped to the pitch.
  Writes outputs/vision/tracks_sn/<clip>_positions.csv, <clip>_ball.csv and outputs/vision/track_eval_sn.csv.
Usage : .venv-vision\\Scripts\\python.exe scripts\\track_eval.py --clips SNGS-116 SNGS-117 SNGS-118 [--detector rfdetr_sn]
"""
import argparse
import glob
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from trackers import ByteTrackTracker   # ByteTrack moved out of supervision (0.27+) into Roboflow "trackers"
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import vision_poc as V
import pitch_calib as PC

ROOT = "data/soccernet_gsr_full/test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--fps", type=float, default=12.5, help="frame rate of the fetched frames (25 / step)")
    ap.add_argument("--detector", choices=["rfdetr", "rfdetr_sn"], default="rfdetr")
    a = ap.parse_args()
    sn = a.detector == "rfdetr_sn"
    tdir, evf = ("outputs/vision/tracks_sn", "outputs/vision/track_eval_sn.csv") if sn else ("outputs/vision/tracks", "outputs/vision/track_eval.csv")
    os.makedirs(tdir, exist_ok=True)
    V.ROOT, PC.V.ROOT = ROOT, ROOT
    det_model = V.RFDETRSoccerNet() if sn else V.RFDETR()
    res = []
    for clip in a.clips:
        files, by = V.labels(clip)
        lines = PC.frame_lines(clip)
        tracker = ByteTrackTracker(frame_rate=a.fps, track_activation_threshold=0.5, high_conf_det_threshold=0.5)
        frames = sorted(glob.glob(f"{ROOT}/{clip}/img1/*.jpg"))
        seq, pos_rows, perr, n_gt, n_hit = defaultdict(list), [], [], 0, 0
        colours, roles, ball_rows = defaultdict(list), defaultdict(list), []
        for f in frames:
            fn = os.path.basename(f)
            img = Image.open(f).convert("RGB")
            alld = det_model(img)
            dets = [d for d in alld if d["label"] == "person"]
            d = sv.Detections(xyxy=np.array([x["box"] for x in dets]).reshape(-1, 4), confidence=np.array([x["score"] for x in dets]),
                              class_id=np.zeros(len(dets), int))
            t = tracker.update(d)
            t = t[np.asarray(t.tracker_id) >= 0] if len(t) else t   # drop not-yet-confirmed tracks (id -1)
            iid = files[fn]
            gt = [g for g in by.get(iid, []) if g["role"] in ("player", "goalkeeper", "referee")]
            Hm = PC.solve(lines.get(iid, {}))
            bgr = cv2.imread(f)
            balls = [d for d in alld if d["label"] == "ball"]
            if sn and balls and Hm is not None:
                b = max(balls, key=lambda d: d["score"])
                bxy = PC.apply(Hm, [V.foot(b["box"])])[0]
                ball_rows.append(dict(frame=int(fn[:-4]), time_s=int(fn[:-4]) / 25, x_m=bxy[0], y_m=bxy[1], score=b["score"]))
            if sn and len(t):
                for box, tid in zip(t.xyxy, t.tracker_id):
                    ious = [V.iou(box, x["box"]) for x in dets]
                    if ious and max(ious) > 0.5:
                        roles[int(tid)].append(dets[int(np.argmax(ious))].get("role", "player"))
            for box, tid in zip(t.xyxy, t.tracker_id):
                x0, y0, x1, y1 = [int(v) for v in box]
                c = bgr[y0 + (y1 - y0) // 5: y0 + (y1 - y0) // 2, x0 + (x1 - x0) // 4: x1 - (x1 - x0) // 4]
                if c.size:
                    colours[int(tid)].append(c.reshape(-1, 3).mean(0))
                xy = PC.apply(Hm, [V.foot(box)])[0] if Hm is not None else [np.nan, np.nan]
                pos_rows.append(dict(frame=int(fn[:-4]), time_s=int(fn[:-4]) / 25, track_id=int(tid), x_m=xy[0], y_m=xy[1]))
            for g in gt:
                n_gt += 1
                ious = [V.iou(g["box"], b) for b in t.xyxy]
                if ious and max(ious) >= 0.5:
                    n_hit += 1
                    k = int(np.argmax(ious))
                    seq[g["track"]].append(int(t.tracker_id[k]))
                    if Hm is not None and g["pitch"][0] is not None:
                        perr.append(np.linalg.norm(PC.apply(Hm, [V.foot(t.xyxy[k])])[0] - g["pitch"]))
        sw = sum(sum(1 for i in range(1, len(s)) if s[i] != s[i - 1]) for s in seq.values())
        idf1 = sum(Counter(s).most_common(1)[0][1] for s in seq.values()) / max(sum(len(s) for s in seq.values()), 1)
        role = {k: Counter(v).most_common(1)[0][0] for k, v in roles.items()}
        tids = sorted(k for k in colours if role.get(k) != "referee")
        if len(tids) >= 2:
            X = np.float32([np.mean(colours[k], 0) for k in tids])
            _, lab, _ = cv2.kmeans(X, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1), 5, cv2.KMEANS_PP_CENTERS)
            team = dict(zip(tids, lab.ravel()))
        else:
            team = {}
        pdf = pd.DataFrame(pos_rows)
        pdf["team"] = pdf["track_id"].map(team)
        if sn:
            pdf["role"] = pdf["track_id"].map(role).fillna("player")
            pdf.loc[pdf["role"] == "referee", "team"] = "ref"
            pd.DataFrame(ball_rows, columns=["frame", "time_s", "x_m", "y_m", "score"]).to_csv(f"{tdir}/{clip}_ball.csv", index=False)
        pdf.to_csv(f"{tdir}/{clip}_positions.csv", index=False)
        minutes = len(frames) / a.fps / 60
        r = dict(clip=clip, frames=len(frames), true_players=len(seq), track_ids=pdf["track_id"].nunique(), coverage=n_hit / max(n_gt, 1),
                 id_switches=sw, switches_per_player_min=sw / max(len(seq), 1) / minutes, idf1_like=idf1, pos_m=float(np.mean(perr)) if perr else np.nan)
        if sn:
            r["ball_frames"] = len(ball_rows) / max(len(frames), 1)
        res.append(r)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    pd.DataFrame(res).to_csv(evf, index=False)


if __name__ == "__main__":
    main()
