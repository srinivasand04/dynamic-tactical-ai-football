"""Record viewer/showcase.html (three.js) to an mp4, frame by frame: the page runs on one timeline (window.__seek(t)),
so every frame is exact (no dropped frames, no screen recorder).

Needs: playwright (chromium) and ffmpeg on PATH; internet for the three.js module (cdn.jsdelivr.net).
Output: outputs/showcase/player_models_showcase.mp4 (not committed: *.mp4 is in .gitignore)
Usage : python scripts/record_showcase.py [--fps 25] [--stills]       (--stills: one PNG per scene for a quick check)
        python scripts/record_showcase.py --page coach_run.html          (the coach run what-if -> outputs/showcase/coach_run.mp4)
        python scripts/record_showcase.py --page coach_run.html --data coach_goal_1   (another scenario -> coach_goal_1.mp4)
"""
import argparse
import functools
import http.server
import os
import subprocess
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "viewer")
OUT = os.path.join(ROOT, "outputs", "showcase")


def serve():
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=VIEWER)
    h.log_message = lambda *a: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--page", default="showcase.html", help="page in viewer/ with window.__seek / __duration / __ready")
    ap.add_argument("--data", default=None, help="coach_run.html: the data file (viewer/data/<data>.json); also names the video and stills")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    os.makedirs(OUT, exist_ok=True)
    srv, port = serve()
    with sync_playwright() as pw:
        exe = None
        if not os.path.exists(pw.chromium.executable_path):                  # fall back to an installed Chromium build
            import glob
            found = sorted(glob.glob(os.path.join(os.path.expanduser("~"), "AppData", "Local", "ms-playwright", "chromium-*", "chrome-win*", "chrome.exe")))
            exe = found[-1] if found else None
        br = pw.chromium.launch(executable_path=exe, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        pg = br.new_page(viewport={"width": 1920, "height": 1080})
        pg.goto(f"http://127.0.0.1:{port}/{a.page}?record" + (f"&data={a.data}" if a.data else ""), wait_until="networkidle")
        pg.wait_for_function("window.__ready === true", timeout=120000)
        dur = pg.evaluate("window.__duration")
        if a.stills:
            starts = pg.evaluate("window.__scenes || (() => { let a = 0, r = []; for (const s of [7,12,11,11,10,11,9,9,10,10,7]) { r.push(a + s * 0.75); a += s; } return r; })()")
            for i, t in enumerate(starts):
                pg.evaluate(f"window.__seek({t})")
                pg.screenshot(path=os.path.join(OUT, f"{'still' if a.page == 'showcase.html' else (a.data or a.page.replace('.html', ''))}_{i:02d}.png"))
            print(f"{len(starts)} stills in {OUT}")
        else:
            n = int(dur * a.fps)
            out = os.path.join(OUT, "player_models_showcase.mp4" if a.page == "showcase.html" else (a.data or a.page.replace(".html", "")) + ".mp4")
            ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(a.fps), "-c:v", "mjpeg", "-i", "-",
                                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", "-movflags", "+faststart", out],
                                  stdin=subprocess.PIPE)
            for f in range(n):
                pg.evaluate(f"window.__seek({f / a.fps})")
                ff.stdin.write(pg.screenshot(type="jpeg", quality=92))
                if (f + 1) % (10 * a.fps) == 0:
                    print(f"  {f + 1}/{n} frames", flush=True)
            ff.stdin.close()
            ff.wait()
            print(f"{out}: {n} frames, {dur:.0f} s, {os.path.getsize(out) // 1024 // 1024} MB")
        br.close()
    srv.shutdown()


if __name__ == "__main__":
    main()
