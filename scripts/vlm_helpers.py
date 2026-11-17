"""Phase A4: the VLM jobs a detector cannot do, scored against SoccerNet labels.

--task jersey : crops of players whose number is labelled (box at least 90 px tall); Qwen3-VL-2B and Florence-2 OCR read
                the number. Score: exact-match accuracy, seconds per crop.
--task moment : one frame from the middle of each clip (640 x 360); Qwen3-VL-2B names the game situation, compared with the
                clip's action label (e.g. Corner). Score: accuracy, seconds per frame.
Output: outputs/vision/vlm_<task>.csv
Usage : .venv-vision\\Scripts\\python.exe scripts\\vlm_helpers.py --task jersey --n 120
        .venv-vision\\Scripts\\python.exe scripts\\vlm_helpers.py --task moment
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import vision_poc as V

SITUATIONS = ["corner", "free kick", "throw-in", "goal kick", "penalty", "kick-off", "open play"]


def qwen():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    mid = "Qwen/Qwen3-VL-2B-Instruct"
    return AutoProcessor.from_pretrained(mid), Qwen3VLForConditionalGeneration.from_pretrained(mid, dtype=torch.bfloat16).eval()


def ask_qwen(qp, img, text, n=12):
    p, m = qp
    msg = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}]
    inp = p.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    with torch.no_grad():
        ids = m.generate(**inp, max_new_tokens=n, do_sample=False)
    return p.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def number(s):
    m = re.search(r"\d{1,2}", s or "")
    return m.group(0) if m else "none"


def jersey(n):
    crops = []
    rows = []
    for clip in sorted({f.replace("\\", "/").split("/")[-3] for f in glob.glob(f"{V.ROOT}/*/img1/*.jpg")}):
        d = json.load(open(f"{V.ROOT}/{clip}/Labels-GameState.json"))
        names = {im["image_id"]: im["file_name"] for im in d["images"]}
        have = {os.path.basename(f) for f in glob.glob(f"{V.ROOT}/{clip}/img1/*.jpg")}
        for a in d["annotations"]:
            at, b = a.get("attributes") or {}, a.get("bbox_image") or {}
            if a.get("supercategory") == "object" and at.get("jersey") and names.get(a["image_id"]) in have and b.get("h", 0) >= 90:
                crops.append((clip, names[a["image_id"]], at["jersey"], [b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]]))
    step = max(1, len(crops) // n)
    crops = crops[::step][:n]
    print(f"{len(crops)} player crops with a labelled number", flush=True)
    qp = qwen()
    from transformers import AutoProcessor, Florence2ForConditionalGeneration
    fp = AutoProcessor.from_pretrained("florence-community/Florence-2-base")
    fm = Florence2ForConditionalGeneration.from_pretrained("florence-community/Florence-2-base", dtype=torch.float32).eval()
    for clip, fn, true, box in crops:
        img = Image.open(f"{V.ROOT}/{clip}/img1/{fn}").convert("RGB").crop(box)
        img = img.resize((max(64, img.width * 2), max(128, img.height * 2)))
        t = time.time()
        q = number(ask_qwen(qp, img, "What is the jersey number on this player's shirt? Answer with the number only, or none if it is not visible."))
        tq = time.time() - t
        t = time.time()
        inp = fp(text="<OCR>", images=img, return_tensors="pt")
        with torch.no_grad():
            ids = fm.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"], max_new_tokens=20, num_beams=3)
        fl = number(fp.batch_decode(ids, skip_special_tokens=True)[0])
        tf = time.time() - t
        rows.append(dict(clip=clip, frame=fn, true=str(true), qwen=q, florence=fl, qwen_s=tq, florence_s=tf))
        print(f"{clip} {fn} true {true:>3}  qwen {q:>4}  florence {fl:>4}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/vision/vlm_jersey.csv", index=False)
    print(f"accuracy: qwen {(df.qwen == df.true).mean():.0%} ({df.qwen_s.mean():.1f} s/crop), florence OCR {(df.florence == df.true).mean():.0%} ({df.florence_s.mean():.1f} s/crop)")


def moment():
    qp = qwen()
    rows = []
    for clip in sorted({f.replace("\\", "/").split("/")[-3] for f in glob.glob(f"{V.ROOT}/*/img1/*.jpg")}):
        info = json.load(open(f"{V.ROOT}/{clip}/Labels-GameState.json"))["info"]
        fs = sorted(glob.glob(f"{V.ROOT}/{clip}/img1/*.jpg"))
        img = Image.open(fs[len(fs) // 2]).convert("RGB").resize((640, 360))
        t = time.time()
        ans = ask_qwen(qp, img, "This is a football broadcast frame. Which game situation is it? Answer with one of: " + ", ".join(SITUATIONS) + ".", n=8).lower()
        pred = next((s for s in SITUATIONS if s in ans), ans)
        rows.append(dict(clip=clip, true=str(info.get("action_class", "")).lower(), qwen=pred, seconds=time.time() - t))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/vision/vlm_moment.csv", index=False)
    print(df["true"].value_counts().to_string())
    print(f"accuracy: {(df.true.str.replace('-', ' ') == df.qwen.str.replace('-', ' ')).mean():.0%}, {df.seconds.mean():.1f} s/frame")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["jersey", "moment"], required=True)
    ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args()
    os.makedirs("outputs/vision", exist_ok=True)
    jersey(a.n) if a.task == "jersey" else moment()
