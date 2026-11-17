"""Phase 2 proof of concept: can an open model find players and the ball well enough for our graphs?

On SoccerNet GSR frames with known answers (fetch_gsr_sample.py) it runs each detector, saves its output as structured JSON,
draws the boxes, and scores against the labels:
  persons   precision / recall at IoU >= 0.5 (players, goalkeepers, referees)
  ball      found or not (centre within 20 px of the true ball)
  pitch_m   position error in metres: the detection's foot point mapped to the pitch with a homography fitted on the
            frame's own labels (so only the detector's error is measured, not calibration)
  team      colour clustering (2 groups, shirt colour of matched players) vs the true left/right team
  s/frame   seconds per frame on this CPU

Models: rfdetr (RF-DETR Medium, COCO person + sports ball, Apache 2.0), florence (Florence-2 base, MIT, '<OD>' + grounding 'ball').
Output: outputs/vision_poc/<model>/<clip>_<frame>.json + .jpg, outputs/vision_poc/frames.csv, summary.csv
Usage : .venv-vision\\Scripts\\python.exe scripts\\vision_poc.py --models rfdetr florence
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image

OUT = "outputs/vision_poc"
ROOT = "data/soccernet_gsr/test"


def labels(clip):
    d = json.load(open(f"{ROOT}/{clip}/Labels-GameState.json"))
    roles = {c["id"]: c["name"] for c in d["categories"]}
    by = {}
    for a in d["annotations"]:
        if a.get("supercategory") != "object":
            continue
        b = a["bbox_image"]
        p = a.get("bbox_pitch") or {}
        by.setdefault(a["image_id"], []).append(dict(track=a.get("track_id"), role=roles[a["category_id"]], team=(a.get("attributes") or {}).get("team"),
                                                    box=[b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]],
                                                    pitch=[p.get("x_bottom_middle"), p.get("y_bottom_middle")]))
    files = {im["file_name"]: im["image_id"] for im in d["images"]}
    return files, by


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def foot(b):
    return [(b[0] + b[2]) / 2, b[3]]


class RFDETR:
    def __init__(self):
        from rfdetr import RFDETRMedium
        from rfdetr.assets.coco_classes import COCO_CLASSES
        self.m, self.names = RFDETRMedium(), COCO_CLASSES

    def __call__(self, img):
        d = self.m.predict(img, threshold=0.35)
        out = []
        for box, c, s in zip(d.xyxy, d.class_id, d.confidence):
            n = self.names.get(int(c), str(c))
            if n in ("person", "sports ball"):
                out.append(dict(label="ball" if n == "sports ball" else "person", box=[float(v) for v in box], score=float(s)))
        return out


class Florence:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        mid = "florence-community/Florence-2-base"
        self.p = AutoProcessor.from_pretrained(mid)
        self.m = Florence2ForConditionalGeneration.from_pretrained(mid, dtype=torch.float32).eval()
        self.torch = torch

    def ask(self, img, task, text=""):
        inp = self.p(text=task + text, images=img, return_tensors="pt")
        with self.torch.no_grad():
            ids = self.m.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"], max_new_tokens=1024, num_beams=3, do_sample=False)
        txt = self.p.batch_decode(ids, skip_special_tokens=False)[0]
        return self.p.post_process_generation(txt, task=task, image_size=img.size)[task]

    def __call__(self, img):
        out = []
        r = self.ask(img, "<OD>")
        for box, lab in zip(r["bboxes"], r["labels"]):
            if lab.lower() in ("person", "man", "football player", "player", "woman", "boy"):
                out.append(dict(label="person", box=[float(v) for v in box], score=1.0))
        r = self.ask(img, "<CAPTION_TO_PHRASE_GROUNDING>", "a football")
        for box in r["bboxes"][:1]:
            out.append(dict(label="ball", box=[float(v) for v in box], score=1.0))
        return out


class Qwen:
    """Qwen3-VL-2B-Instruct (Apache 2.0): one prompt, JSON out. Boxes come on a 0-1000 grid (Qwen3-VL convention)."""
    PROMPT = ("Detect every person on the pitch (players, goalkeepers, referees) and the football in this broadcast frame. "
              "Return only a JSON list, one item per object: "
              '{"label": "player"|"goalkeeper"|"referee"|"ball", "bbox_2d": [x1, y1, x2, y2], "shirt": "<main shirt colour>", "number": "<jersey number or null>"}')

    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        mid = "Qwen/Qwen3-VL-2B-Instruct"
        self.p = AutoProcessor.from_pretrained(mid)
        self.m = Qwen3VLForConditionalGeneration.from_pretrained(mid, dtype=torch.bfloat16).eval()
        self.torch = torch

    def __call__(self, img):
        small = img.resize((1280, 720))
        msg = [{"role": "user", "content": [{"type": "image", "image": small}, {"type": "text", "text": self.PROMPT}]}]
        inp = self.p.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        with self.torch.no_grad():
            ids = self.m.generate(**inp, max_new_tokens=1800, do_sample=False)
        txt = self.p.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        self.raw = txt
        a, b = txt.find("["), txt.rfind("]")
        try:
            items = json.loads(txt[a:b + 1])
        except Exception:
            items = []
        W, H = img.size
        out = []
        for it in items if isinstance(items, list) else []:
            try:
                x0, y0, x1, y1 = [float(v) for v in it["bbox_2d"]]
            except Exception:
                continue
            lab = str(it.get("label", "")).lower()
            out.append(dict(label="ball" if lab == "ball" else "person", role=lab, box=[x0 / 1000 * W, y0 / 1000 * H, x1 / 1000 * W, y1 / 1000 * H],
                            score=1.0, shirt=it.get("shirt"), number=it.get("number")))
        return out


MODELS = {"rfdetr": RFDETR, "florence": Florence, "qwen": Qwen}


def team_accuracy(crops_rgb, teams):
    """2-group k-means on mean shirt colour; accuracy against true left/right (best of the two namings)."""
    if len(crops_rgb) < 4:
        return np.nan
    X = np.float32(crops_rgb)
    _, lab, _ = cv2.kmeans(X, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1), 5, cv2.KMEANS_PP_CENTERS)
    t = np.array([1 if x == "left" else 0 for x in teams])
    acc = (lab.ravel() == t).mean()
    return float(max(acc, 1 - acc))


def evaluate(gt, det, img_bgr):
    persons = [g for g in gt if g["role"] in ("player", "goalkeeper", "referee")]
    ball = [g for g in gt if g["role"] == "ball"]
    dp = [d for d in det if d["label"] == "person"]
    used, matches = set(), []
    for g in sorted(persons, key=lambda g: -(g["box"][2] - g["box"][0]) * (g["box"][3] - g["box"][1])):
        best, bi = 0.5, None
        for i, d in enumerate(dp):
            if i not in used and iou(g["box"], d["box"]) >= best:
                best, bi = iou(g["box"], d["box"]), i
        if bi is not None:
            used.add(bi)
            matches.append((g, dp[bi]))
    tp = len(matches)
    r = dict(gt_persons=len(persons), det_persons=len(dp), tp=tp, precision=tp / max(len(dp), 1), recall=tp / max(len(persons), 1))
    db = [d for d in det if d["label"] == "ball"]
    if ball:
        gc = np.array(foot(ball[0]["box"])) - [0, (ball[0]["box"][3] - ball[0]["box"][1]) / 2]
        r["ball_found"] = float(any(np.linalg.norm(np.array([(d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2]) - gc) < 20 for d in db))
    else:
        r["ball_found"] = np.nan
    src = np.float32([foot(g["box"]) for g in gt if g["pitch"][0] is not None])
    dst = np.float32([g["pitch"] for g in gt if g["pitch"][0] is not None])
    if len(src) >= 6:
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 1.0)
        mm = [(g, d) for g, d in matches if g["pitch"][0] is not None]
        if H is not None and mm:
            p = cv2.perspectiveTransform(np.float32([foot(d["box"]) for _, d in mm])[None], H)[0]
            r["pitch_m"] = float(np.mean(np.linalg.norm(p - np.float32([g["pitch"] for g, _ in mm]), axis=1)))
            q = cv2.perspectiveTransform(src[None], H)[0]
            r["homography_floor_m"] = float(np.median(np.linalg.norm(q - dst, axis=1)))
    pl = [(g, d) for g, d in matches if g["role"] == "player" and g["team"] in ("left", "right")]
    crops = []
    for g, d in pl:
        x0, y0, x1, y1 = [int(v) for v in d["box"]]
        c = img_bgr[y0 + (y1 - y0) // 5: y0 + (y1 - y0) // 2, x0 + (x1 - x0) // 4: x1 - (x1 - x0) // 4]
        crops.append(c.reshape(-1, 3).mean(0)[::-1] if c.size else [0, 0, 0])
    r["team_acc"] = team_accuracy(crops, [g["team"] for g, _ in pl])
    return r


def draw(img_bgr, gt, det, path):
    im = img_bgr.copy()
    for g in gt:
        x0, y0, x1, y1 = map(int, g["box"])
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 200, 0), 1)
    for d in det:
        x0, y0, x1, y1 = map(int, d["box"])
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255) if d["label"] == "ball" else (255, 120, 0), 2)
    cv2.imwrite(path, im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    rows_p = f"{OUT}/frames.csv"
    if not a.report:
        rows = []
        frames = sorted(glob.glob(f"{ROOT}/*/img1/*.jpg"))
        for name in a.models:
            os.makedirs(f"{OUT}/{name}", exist_ok=True)
            t0 = time.time()
            try:
                model = MODELS[name]()
            except Exception as e:
                print(f"{name}: could not load ({e})", flush=True)
                continue
            print(f"{name}: loaded in {time.time() - t0:.0f} s", flush=True)
            for f in frames:
                clip, fn = f.replace("\\", "/").split("/")[-3], os.path.basename(f)
                files, by = labels(clip)
                gt = by.get(files[fn], [])
                img = Image.open(f).convert("RGB")
                t = time.time()
                det = model(img)
                sec = time.time() - t
                bgr = cv2.imread(f)
                tag = f"{clip}_{fn[:-4]}"
                json.dump({"model": name, "clip": clip, "frame": fn, "seconds": sec, "detections": det, "raw": getattr(model, "raw", None)}, open(f"{OUT}/{name}/{tag}.json", "w"), indent=1)
                draw(bgr, gt, det, f"{OUT}/{name}/{tag}.jpg")
                r = dict(model=name, clip=clip, frame=fn, seconds=sec, **evaluate(gt, det, bgr))
                rows.append(r)
                print(f"{name} {tag}: recall {r['recall']:.2f} precision {r['precision']:.2f} ball {r['ball_found']} pitch {r.get('pitch_m', np.nan):.2f} m team {r['team_acc']:.2f} {sec:.1f} s", flush=True)
            prev = pd.read_csv(rows_p) if os.path.exists(rows_p) else pd.DataFrame()
            if len(prev):
                prev = prev[prev["model"] != name]
            pd.concat([prev, pd.DataFrame([x for x in rows if x["model"] == name])]).to_csv(rows_p, index=False)
    if not os.path.exists(rows_p):
        return
    df = pd.read_csv(rows_p)
    sm = df.groupby("model").agg(frames=("frame", "count"), recall=("recall", "mean"), precision=("precision", "mean"),
                                 ball_found=("ball_found", "mean"), pitch_m=("pitch_m", "mean"), homography_floor_m=("homography_floor_m", "mean"),
                                 team_acc=("team_acc", "mean"), s_per_frame=("seconds", "mean")).round(3)
    sm.to_csv(f"{OUT}/summary.csv")
    print(sm.to_string())


if __name__ == "__main__":
    main()
