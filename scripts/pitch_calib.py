"""Phase A2: our own pitch mapping (homography) from pitch lines, instead of the dataset's camera calibration.

Each SoccerNet GSR frame has its visible pitch lines marked as points ('Side line top', 'Big rect. left main', ...).
We know where every straight line lies on a 105 x 68 m pitch, so each marked point gives one equation
'this image point maps onto that pitch line'. With 4+ lines (2+ each way) the homography is solved by least squares.
Circles and goal posts are not used (curved / not on the ground).

This is the upper bound for a line-based pipeline: the lines come from labels here; a line detector would replace them.

Scores per frame:
  calib_m  error (m) of true player foot points mapped with OUR homography vs their true pitch position
  det_m    error (m) of RF-DETR's matched boxes mapped with OUR homography (needs outputs/vision_poc/rfdetr/*.json)
Output: outputs/vision/pitch_calib.csv + printed summary
Usage : .venv-vision\\Scripts\\python.exe scripts\\pitch_calib.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import vision_poc as V

L2, W2 = 52.5, 34.0
X = lambda k: (1.0, 0.0, -k)
Y = lambda k: (0.0, 1.0, -k)
LINES = {"Side line left": X(-L2), "Side line right": X(L2), "Side line top": Y(-W2), "Side line bottom": Y(W2), "Middle line": X(0),
         "Big rect. left main": X(-36.0), "Big rect. right main": X(36.0), "Small rect. left main": X(-47.0), "Small rect. right main": X(47.0),
         "Big rect. left top": Y(-20.16), "Big rect. right top": Y(-20.16), "Big rect. left bottom": Y(20.16), "Big rect. right bottom": Y(20.16),
         "Small rect. left top": Y(-9.16), "Small rect. right top": Y(-9.16), "Small rect. left bottom": Y(9.16), "Small rect. right bottom": Y(9.16)}


def solve(lines, W=1920, H=1080, sx=1, sy=1):
    pts, ls = [], []
    for name, p in lines.items():
        if name not in LINES:
            continue
        a, b, c = LINES[name]
        l = np.array([a * sx, b * sy, c])
        for q in p:
            pts.append([q["x"] * W, q["y"] * H, 1.0])
            ls.append(l)
    names = [n for n in lines if n in LINES]
    xs = {LINES[n][2] for n in names if LINES[n][0]}
    ys = {LINES[n][2] for n in names if LINES[n][1]}
    if len(xs) < 2 or len(ys) < 2 or len(pts) < 8:
        return None
    pts, ls = np.array(pts), np.array(ls)
    m, s = pts[:, :2].mean(0), np.sqrt(2) / np.linalg.norm(pts[:, :2] - pts[:, :2].mean(0), axis=1).mean()
    T = np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
    S = np.diag([1 / 50, 1 / 50, 1.0])
    pn, ln = pts @ T.T, ls @ np.linalg.inv(S)                        # l' = S^-T l  (row form: l @ S^-1)
    A = np.einsum("ni,nj->nij", ln, pn).reshape(len(pn), 9)
    Hn = np.linalg.svd(A)[2][-1].reshape(3, 3)
    return np.linalg.inv(S) @ Hn @ T


def apply(Hm, xy):
    p = np.c_[np.asarray(xy, float), np.ones(len(xy))] @ Hm.T
    return p[:, :2] / p[:, 2:]


def frame_lines(clip):
    d = json.load(open(f"{V.ROOT}/{clip}/Labels-GameState.json"))
    return {a["image_id"]: a["lines"] for a in d["annotations"] if a.get("supercategory") == "pitch"}


def run(sx, sy, frames):
    rows = []
    for f in frames:
        clip, fn = f.replace("\\", "/").split("/")[-3], os.path.basename(f)
        files, by = V.labels(clip)
        iid = files[fn]
        gt = [g for g in by.get(iid, []) if g["pitch"][0] is not None and g["role"] != "ball"]
        Hm = solve(frame_lines(clip).get(iid, {}), sx=sx, sy=sy)
        r = dict(clip=clip, frame=fn, calibrated=Hm is not None, calib_m=np.nan, det_m=np.nan)
        if Hm is not None and gt:
            r["calib_m"] = float(np.median(np.linalg.norm(apply(Hm, [V.foot(g["box"]) for g in gt]) - [g["pitch"] for g in gt], axis=1)))
            jp = f"outputs/vision_poc/rfdetr/{clip}_{fn[:-4]}.json"
            if os.path.exists(jp):
                det = [d for d in json.load(open(jp))["detections"] if d["label"] == "person"]
                errs = []
                for g in gt:
                    best = max(det, key=lambda d: V.iou(g["box"], d["box"]), default=None)
                    if best and V.iou(g["box"], best["box"]) >= 0.5:
                        errs.append(np.linalg.norm(apply(Hm, [V.foot(best["box"])])[0] - g["pitch"]))
                r["det_m"] = float(np.mean(errs)) if errs else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    frames = sorted(glob.glob(f"{V.ROOT}/*/img1/*.jpg"))
    first = [f for f in frames if os.path.basename(os.path.dirname(os.path.dirname(f))) == sorted({f.replace("\\", "/").split("/")[-3] for f in frames})[0]]
    conv = min(((sx, sy) for sx in (1, -1) for sy in (1, -1)), key=lambda c: np.nanmedian(run(*c, first)["calib_m"]))
    print(f"pitch sign convention chosen on the first clip: x*{conv[0]}, y*{conv[1]}")
    df = run(*conv, frames)
    os.makedirs("outputs/vision", exist_ok=True)
    df.to_csv("outputs/vision/pitch_calib.csv", index=False)
    c = df[df["calibrated"]]
    print(f"frames {len(df)}, calibrated {len(c)} ({len(c) / len(df):.0%})")
    print(f"true players, our mapping : median {c['calib_m'].median():.2f} m, mean {c['calib_m'].mean():.2f} m, under 0.5 m in {(c['calib_m'] < 0.5).mean():.0%} of frames")
    print(f"RF-DETR boxes, our mapping: median {c['det_m'].median():.2f} m, mean {c['det_m'].mean():.2f} m, under 0.5 m in {(c['det_m'] < 0.5).mean():.0%} of frames")


if __name__ == "__main__":
    main()
