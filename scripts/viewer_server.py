"""The viewer's web server, with the coach board: serves viewer/ like `python -m http.server`, and when the coach presses
Play on the board it saves the board and runs scripts/coach_board.py (GRF) in the background.

    POST /api/board          body = board JSON from the viewer  -> {"id": ...}
    GET  /api/board/<id>     -> {"state": "running" | "done" | "failed", "line": last log line, "file": viewer file}

Local only (127.0.0.1). Usage: .venv\\Scripts\\python.exe scripts\\viewer_server.py      then open http://localhost:8765/
"""
import functools
import json
import os
import re
import subprocess
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "coach_board")
JOBS = {}


class Handler(SimpleHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/board":
            return self._json({"error": "not found"}, 404)
        try:
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            assert b.get("board") in ("forward", "midfield", "defence") and b.get("moves") and re.fullmatch(r"[A-Za-z0-9]+", str(b.get("match", "")))
        except (ValueError, AssertionError):
            return self._json({"error": "bad board"}, 400)
        os.makedirs(OUT, exist_ok=True)
        bid = time.strftime("%Y%m%d_%H%M%S")
        b["id"] = bid
        path = os.path.join(OUT, f"{bid}.json")
        json.dump(b, open(path, "w", encoding="utf-8"), indent=1)
        log = open(os.path.join(OUT, f"{bid}.log"), "w", encoding="utf-8")
        JOBS[bid] = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "scripts", "coach_board.py"), "--board", path],
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"coach board {bid}: {b['board']} board, {len(b['moves'])} player(s) moved -> GRF running", flush=True)
        self._json({"id": bid})

    def do_GET(self):
        m = re.fullmatch(r"/api/board/([0-9_]+)", self.path)
        if not m:
            return super().do_GET()
        bid = m.group(1)
        p = JOBS.get(bid)
        if p is None:
            return self._json({"error": "unknown board"}, 404)
        lines = [l.strip() for l in open(os.path.join(OUT, f"{bid}.log"), encoding="utf-8", errors="replace") if l.strip()]
        rc = p.poll()
        state = "running" if rc is None else ("done" if rc == 0 else "failed")
        self._json({"state": state, "line": lines[-1] if lines else "starting GRF…", "file": f"board_{bid}.json"})

    def log_message(self, fmt, *args):            # keep the console quiet (only the board messages)
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    srv = ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=os.path.join(ROOT, "viewer")))
    print(f"viewer + coach board on http://localhost:{port}/  (Ctrl+C to stop)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
