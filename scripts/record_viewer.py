"""Record the usual three.js viewer (viewer/index.html) in mode 6, the coach run, to an mp4, frame by frame.

The viewer is driven through its own debug hooks (window.__viewer, window.__viewerStep): each frame sets the clock, steps the
viewer once and takes a screenshot, so nothing is dropped. Three parts, each with a caption on top of the viewer:
    1 split view   real match (left) | coach run what-if (right), broadcast camera, Klaus followed (white = real, gold = what-if)
    2 overlay      the what-if ghosts on the real pitch, top camera
    3 split view   real | model with NO change (the control: same model, no instruction)
Needs  : playwright (chromium) and ffmpeg on PATH; viewer/data/mode6_coach_run.json (scripts/viewer_mode.py coach_run)
Output : outputs/showcase/coach_run_viewer.mp4
Usage  : python scripts/record_viewer.py [--fps 25] [--stills]
"""
import argparse
import json
import os
import subprocess

import record_showcase as RS

ROOT, VIEWER, OUT = RS.ROOT, RS.VIEWER, RS.OUT


def parts(cr):
    who = cr["who"].split(" ")[-1]
    t0, r = cr["start_s"], cr["run"]
    ps, ts = r.get("pass_s"), r.get("touch_s")
    sm = {s["metric"]: s for s in cr["summary"]}
    th = sm["max_threat"]

    def cap_run(t):
        d = t - t0
        if d < 0:
            return f"27:01. Left: the real match. Right: the same moment played by our team models in GRF, with the coach's instruction: “{who}, run in behind; we play you.”"
        if ps is not None and d < ps:
            return f"Right: {who} sprints in behind the left-back (gold dashed = his what-if path). Left: the real {who} drops wide (white path)."
        if ts is not None and d < ts:
            return f"Right: he is played through as he passes their line ({ps:.1f} s). Left: the real ball goes wide to {who} on the touchline."
        if ts is not None and d < ts + 3:
            return f"Right: {who} controls it in the box ({ts:.1f} s). Left: the real attack is lost after {cr['real']['lost_ball_s']:.1f} s."
        return f"This run: {r['shots']} shots, no goal. Over {cr['seeds']} paired runs the threat goes {th['no_change']:.2f} → {th['coach_run']:.2f} (±{th['ci95']:.2f})."

    return [
        {"mode": "split", "layer": 1, "cam": "follow", "a": t0 - 3, "b": t0 + 12, "cap": cap_run, "tag": "1 · Real | coach run"},
        {"mode": "overlay", "layer": 1, "cam": "top", "a": t0, "b": t0 + 9,
         "cap": lambda t: f"Overlay: real players solid, the coach-run run of {who} as a ghost; white = where he really ran, gold dashed = the instructed run.",
         "tag": "2 · Overlay, top view"},
        {"mode": "split", "layer": 0, "cam": "follow", "a": t0, "b": t0 + 9,
         "cap": lambda t: f"Control: the same model with NO instruction. {who} stays wide, as in the real match; the threat stays at {th['no_change']:.2f}. The difference is the coach's run.",
         "tag": "3 · Real | model, no change"},
    ]


SETUP = """async ([mode, layer, cam, tag]) => {
  const v = document.getElementById('viewMode'); v.value = mode; v.dispatchEvent(new Event('change'));
  const l = document.getElementById('viewLayer'); l.value = String(layer); l.dispatchEvent(new Event('change'));
  document.querySelector(`[data-cam="${cam}"]`).click();
  const S = window.__viewer; S.playing = false;
  let c = document.getElementById('recCap');
  if (!c) {
    c = document.createElement('div'); c.id = 'recCap';
    c.style.cssText = 'position:fixed;left:40px;right:40px;bottom:150px;z-index:99;font:600 26px Segoe UI,Arial;color:#eef2f8;background:rgba(10,18,34,.82);padding:14px 20px;border-radius:10px;border-left:6px solid #ffc83d';
    document.body.appendChild(c);
    const t = document.createElement('div'); t.id = 'recTag';
    t.style.cssText = 'position:fixed;left:40px;top:120px;z-index:99;font:700 22px Segoe UI,Arial;letter-spacing:.08em;color:#ffc83d;background:rgba(10,18,34,.8);padding:8px 14px;border-radius:8px';
    document.body.appendChild(t);
  }
  document.getElementById('recTag').textContent = tag;
}"""
STEP = """([t, cap, dt]) => {
  const S = window.__viewer; S.t = t; S.animDt = dt / 2;
  window.__viewerStep(); window.__viewerStep();
  const c = document.getElementById('recCap'); if (c.textContent !== cap) c.textContent = cap;
}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--stills", action="store_true")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    cr = json.load(open(os.path.join(VIEWER, "data", "coach_run.json"), encoding="utf-8"))
    os.makedirs(OUT, exist_ok=True)
    srv, port = RS.serve()
    with sync_playwright() as pw:
        exe = None
        if not os.path.exists(pw.chromium.executable_path):
            import glob
            found = sorted(glob.glob(os.path.join(os.path.expanduser("~"), "AppData", "Local", "ms-playwright", "chromium-*", "chrome-win*", "chrome.exe")))
            exe = found[-1] if found else None
        br = pw.chromium.launch(executable_path=exe, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        pg = br.new_page(viewport={"width": 1920, "height": 1080})
        pg.goto(f"http://127.0.0.1:{port}/index.html?file=mode6_coach_run.json", wait_until="networkidle")
        pg.wait_for_function("window.__viewer && window.__viewer.data && window.__viewer.data.focus != null", timeout=120000)
        pg.evaluate("() => { const p = document.getElementById('panel'); if (!p.classList.contains('hidden')) document.getElementById('togglePanel').click(); }")
        ff, out = None, os.path.join(OUT, "coach_run_viewer.mp4")
        if not a.stills:
            ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(a.fps), "-c:v", "mjpeg", "-i", "-",
                                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", "-movflags", "+faststart", out],
                                  stdin=subprocess.PIPE)
        n = 0
        for i, p in enumerate(parts(cr)):
            pg.evaluate(SETUP, [p["mode"], p["layer"], p["cam"], p["tag"]])
            steps = int((p["b"] - p["a"]) * a.fps)
            if a.stills:
                for j, u in enumerate((0.15, 0.45, 0.8)):
                    t = p["a"] + u * (p["b"] - p["a"])
                    for _ in range(20):                                    # let the camera settle on that moment
                        pg.evaluate(STEP, [t, p["cap"](t), 0.0])
                    pg.screenshot(path=os.path.join(OUT, f"coach_run_viewer_{i}{j}.png"))
                continue
            for _ in range(15):
                pg.evaluate(STEP, [p["a"], p["cap"](p["a"]), 0.0])
            for k in range(steps):
                t = p["a"] + k / a.fps
                pg.evaluate(STEP, [t, p["cap"](t), 1.0 / a.fps])
                ff.stdin.write(pg.screenshot(type="jpeg", quality=92))
                n += 1
                if n % (10 * a.fps) == 0:
                    print(f"  {n} frames", flush=True)
        if ff:
            ff.stdin.close()
            ff.wait()
            print(f"{out}: {n} frames, {n / a.fps:.0f} s, {os.path.getsize(out) // 1024 // 1024} MB")
        else:
            print(f"stills in {OUT} (coach_run_viewer_*.png)")
        br.close()
    srv.shutdown()


if __name__ == "__main__":
    main()
