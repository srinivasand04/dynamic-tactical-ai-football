"""Re-score saved Qwen answers with a tolerant parser: keep every complete {...} object, drop exact repeats (looping).
Adds rows 'qwen_repaired' to outputs/vision_poc/frames.csv and prints how many answers were broken or looping."""
import glob, json, re, cv2, numpy as np, pandas as pd, sys, os
sys.path.insert(0, "scripts")
import vision_poc as V
rows, broken, looped = [], 0, 0
for f in sorted(glob.glob("outputs/vision_poc/qwen/*.json")):
    j = json.load(open(f)); raw = j["raw"] or ""
    a, b = raw.find("["), raw.rfind("]")
    try: json.loads(raw[a:b + 1])
    except Exception: broken += 1
    objs, seen = [], set()
    for m in re.finditer(r"\{[^{}]*\}", raw):
        try: it = json.loads(m.group(0))
        except Exception: continue
        k = json.dumps(it, sort_keys=True)
        if k in seen: continue
        seen.add(k); objs.append(it)
    if len(seen) < len(re.findall(r"\{[^{}]*\}", raw)) - 2: looped += 1
    img = f"{V.ROOT}/{j['clip']}/img1/{j['frame']}"; bgr = cv2.imread(img); H, W = bgr.shape[:2]
    det = []
    for it in objs:
        try: x0, y0, x1, y1 = [float(v) for v in it["bbox_2d"]]
        except Exception: continue
        det.append(dict(label="ball" if str(it.get("label")).lower() == "ball" else "person", box=[x0/1000*W, y0/1000*H, x1/1000*W, y1/1000*H]))
    files, by = V.labels(j["clip"])
    rows.append(dict(model="qwen_repaired", clip=j["clip"], frame=j["frame"], seconds=j["seconds"], **V.evaluate(by.get(files[j["frame"]], []), det, bgr)))
p = "outputs/vision_poc/frames.csv"; df = pd.read_csv(p); df = df[df["model"] != "qwen_repaired"]
pd.concat([df, pd.DataFrame(rows)]).to_csv(p, index=False)
print(f"{len(rows)} answers: {broken} not valid JSON, {looped} looping")
