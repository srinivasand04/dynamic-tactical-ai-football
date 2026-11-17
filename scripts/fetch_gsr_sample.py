"""Phase 2: fetch a small sample of SoccerNet Game State Reconstruction (GSR, GPL-3.0) without downloading the 8.9 GB zip.

Reads the zip's table of contents over HTTP range requests and extracts only: each chosen clip's Labels-GameState.json
and every --step-th frame. Default: 2 clips x 10 frames.

Output: data/soccernet_gsr/<split>/<clip>/Labels-GameState.json, img1/<frame>.jpg
Usage : .venv-vision\\Scripts\\python.exe scripts\\fetch_gsr_sample.py --clips 2 --step 75
"""
import argparse
import os
from collections import defaultdict

from remotezip import RemoteZip

URL = "https://huggingface.co/datasets/SoccerNet/SN-GSR-2025/resolve/main/{}.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--clips", type=int, default=2)
    ap.add_argument("--step", type=int, default=75)
    ap.add_argument("--list", action="store_true", help="only list clips")
    ap.add_argument("--out", default="data/soccernet_gsr", help="folder (full clips for tracking go to data/soccernet_gsr_full)")
    ap.add_argument("--only", nargs="*", default=None, help="these clip names only")
    a = ap.parse_args()
    out = os.path.join(a.out, a.split)
    with RemoteZip(URL.format(a.split)) as z:
        names = z.namelist()
        clips = defaultdict(list)
        for n in names:
            if not n.endswith("/"):
                clips[n.split("/")[0]].append(n)
        keys = sorted(k for k in clips if any(n.endswith("Labels-GameState.json") for n in clips[k]))
        print(f"{len(names)} files, {len(keys)} labelled clips; first: {keys[:5]}")
        if a.list:
            return
        for c in (a.only or keys[:a.clips]):
            files = clips[c]
            lab = [n for n in files if n.endswith("Labels-GameState.json")]
            imgs = sorted(n for n in files if n.endswith(".jpg"))[::a.step]
            for n in lab + imgs:
                z.extract(n, out)
            print(f"{c}: labels + {len(imgs)} frames")


if __name__ == "__main__":
    main()
