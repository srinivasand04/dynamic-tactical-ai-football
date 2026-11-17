"""Project video v2 (about 10 minutes): presentation-style slides, an opening animation, an animated architecture and GRF clips.

No shaking: there is no zoom or pan anywhere. Every frame is drawn here at exactly 1920 x 1080; static slides repeat the
same pixels; animations move only shapes, with sub-pixel anti-aliasing; text never moves (it only fades).
Slides build up step by step (soft cross-fades), and scenes cross-fade into each other.

Voice : Microsoft neural voice through edge-tts (needs internet). Cached in build/ (delete voice_NN.mp3 to redo one).
Output: presentations/Dynamic_Tactical_AI_Project_Video_v2.mp4
        presentations/Dynamic_Tactical_AI_Video_v2_slides.pptx   (the same slides, with the GRF clips embedded)
        presentations/architecture_animation.gif / .mp4          (LinkedIn-style architecture animation, silent)
Usage : python deck_work\\video2\\build_video_v2.py --stills     (quick check: one picture per scene -> build/sheet.png)
        python deck_work\\video2\\build_video_v2.py              (full render, about 15-25 minutes)
"""
import argparse
import asyncio
import glob
import json
import math
import os
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORK = os.path.join(HERE, "build")
PRES = os.path.join(ROOT, "presentations")
OUT = os.path.join(PRES, "Dynamic_Tactical_AI_Project_Video_v2.mp4")
VOICE, RATE = "en-US-AndrewNeural", "-4%"
NAME = "Srinivasan Dhakshanamoorthy"
W, H, FPS = 1920, 1080, 25
XF = 0.5                                                              # cross-fade seconds (between builds and scenes)

NAVY, NAVY2, CARD = (10, 18, 34), (18, 30, 54), (26, 40, 68)
WHITE, MUTED, LINE = (240, 243, 248), (150, 162, 184), (52, 68, 98)
RED, TEAL, GOLD, GREEN, BLUE = (230, 50, 90), (40, 200, 190), (245, 190, 70), (70, 200, 120), (80, 150, 255)
FD = "C:/Windows/Fonts/"
O = lambda p: os.path.join(ROOT, "outputs", p)
DEMO = lambda p: os.path.join(ROOT, "deck_work", "demo", p)


def font(size, bold=False):
    for n in (("segoeuib.ttf" if bold else "segoeui.ttf"), ("arialbd.ttf" if bold else "arial.ttf")):
        if os.path.exists(FD + n):
            return ImageFont.truetype(FD + n, size)
    return ImageFont.load_default()


# ----------------------------------------------------------------------------------------------- slide drawing
class Slide:
    """A slide is a background plus build steps; state k shows steps 0..k."""

    def __init__(self, kicker, title, page):
        self.kicker, self.title, self.page, self.steps = kicker, title, page, []

    def step(self, fn):
        self.steps.append(fn)
        return self

    def base(self):
        im = Image.new("RGB", (W, H), NAVY)
        d = ImageDraw.Draw(im)
        for y in range(H):                                            # gentle vertical gradient
            k = y / H
            d.line([0, y, W, y], fill=tuple(int(NAVY[i] * (1 - k) + NAVY2[i] * k) for i in range(3)))
        d.rectangle([0, 0, 10, H], fill=RED)
        if self.title:
            d.text((120, 70), self.kicker.upper(), font=font(24, True), fill=TEAL)
            d.text((120, 110), self.title, font=font(58, True), fill=WHITE)
            d.line([120, 200, 1800, 200], fill=LINE, width=2)
        d.line([120, 1010, 1800, 1010], fill=LINE, width=1)
        d.text((120, 1025), "Dynamic Tactical AI in Football  ·  ESILV · De Vinci Research Center  ·  Nov 2026", font=font(20), fill=MUTED)
        d.text((1800, 1022), NAME, font=font(24, True), fill=WHITE, anchor="ra")
        return im

    def states(self):
        im = self.base()
        out = []
        for fn in self.steps or [lambda d, im: None]:
            fn(ImageDraw.Draw(im), im)
            out.append(np.asarray(im).copy())
        return out


def wrap(d, text, xy, width, f, fill, gap=10):
    x, y = xy
    line = ""
    for w in text.split():
        t = (line + " " + w).strip()
        if d.textlength(t, font=f) > width and line:
            d.text((x, y), line, font=f, fill=fill)
            y += f.size + gap
            line = w
        else:
            line = t
    if line:
        d.text((x, y), line, font=f, fill=fill)
        y += f.size + gap
    return y


def card(x, y, w, h, big=None, label="", sub="", accent=TEAL, bigsize=84):
    def fn(d, im):
        d.rounded_rectangle([x, y, x + w, y + h], 18, fill=CARD, outline=LINE, width=2)
        d.rectangle([x, y + 18, x + 6, y + h - 18], fill=accent)
        yy = y + 26
        if big:
            d.text((x + 34, yy), big, font=font(bigsize, True), fill=accent)
            yy += bigsize + 22
        d.text((x + 34, yy), label, font=font(32, True), fill=WHITE)
        wrap(d, sub, (x + 34, yy + 50), w - 64, font(25), MUTED, 8)
    return fn


def bullets(x, y, items, width=760, size=30, accent=TEAL):
    def fn(d, im):
        yy = y
        for it in items:
            d.ellipse([x, yy + size // 2 - 6, x + 12, yy + size // 2 + 6], fill=accent)
            yy = wrap(d, it, (x + 32, yy), width, font(size), WHITE, 8) + 18
    return fn


def picture(path, x, y, w, h, caption=None, pad=14):
    def fn(d, im):
        d.rounded_rectangle([x, y, x + w, y + h], 18, fill=(250, 251, 253))
        if os.path.exists(path):
            p = Image.open(path).convert("RGB")
            p.thumbnail((w - 2 * pad, h - 2 * pad - (40 if caption else 0)), Image.LANCZOS)
            im.paste(p, (x + (w - p.width) // 2, y + pad + (h - 2 * pad - (40 if caption else 0) - p.height) // 2))
        else:
            d.text((x + 30, y + 30), "missing: " + os.path.basename(path), font=font(24), fill=RED)
        if caption:
            d.text((x + w // 2, y + h - 40), caption, font=font(24, True), fill=(40, 48, 66), anchor="ma")
    return fn


def banner(x, y, w, text, accent=GOLD, size=30):
    def fn(d, im):
        f = font(size, True)
        lines = 1 + int(d.textlength(text, font=f) // (w - 80))
        h = 34 + lines * (size + 12)
        d.rounded_rectangle([x, y, x + w, y + h], 16, fill=(40, 36, 30), outline=accent, width=2)
        wrap(d, text, (x + 40, y + 18), w - 80, f, accent, 12)
    return fn


def table(x, y, cols, rows, widths, size=27, hi=None):
    def fn(d, im):
        rh = size + 26
        d.rounded_rectangle([x, y, x + sum(widths), y + rh * (len(rows) + 1)], 14, fill=CARD, outline=LINE, width=2)
        cx = x
        for c, wd in zip(cols, widths):
            d.text((cx + 22, y + 13), c, font=font(size - 3, True), fill=TEAL)
            cx += wd
        for r, row in enumerate(rows, 1):
            yy = y + r * rh
            d.line([x + 10, yy, x + sum(widths) - 10, yy], fill=LINE, width=1)
            cx = x
            for j, (v, wd) in enumerate(zip(row, widths)):
                d.text((cx + 22, yy + 13), str(v), font=font(size, j == 0 or r == hi), fill=GOLD if r == hi else WHITE)
                cx += wd
    return fn


def flow(x, y, labels, w=196, h=110, gap=44, accent=TEAL, size=24):
    def fn(d, im):
        for i, t in enumerate(labels):
            xx = x + i * (w + gap)
            last = i == len(labels) - 1
            d.rounded_rectangle([xx, y, xx + w, y + h], 16, fill=(60, 40, 20) if last else CARD, outline=GOLD if last else accent, width=3)
            wrap(d, t, (xx + 16, y + 18), w - 32, font(size, True), GOLD if last else WHITE, 6)
            if not last:
                ax = xx + w + 6
                d.line([ax, y + h // 2, ax + gap - 14, y + h // 2], fill=accent, width=4)
                d.polygon([(ax + gap - 12, y + h // 2 - 9), (ax + gap - 2, y + h // 2), (ax + gap - 12, y + h // 2 + 9)], fill=accent)
    return fn


# ----------------------------------------------------------------------------------------------- numbers from disk
def vision_numbers():
    n = {"track": "running", "jersey": "running", "moment": "running"}
    try:
        import pandas as pd
        t = pd.read_csv(O("vision/track_eval.csv"))
        n["track"] = (f"{t.coverage.min():.0%}-{t.coverage.max():.0%} of players followed; "
                      f"{t.switches_per_player_min.min():.0f}-{t.switches_per_player_min.max():.0f} ID switches per player per minute")
        j = pd.read_csv(O("vision/vlm_jersey.csv"), dtype=str)
        n["jersey"] = f"Qwen {(j.qwen == j.true).mean():.0%} vs OCR {(j.florence == j.true).mean():.0%} ({len(j)} crops)"
        m = pd.read_csv(O("vision/vlm_moment.csv"), dtype=str).fillna("")
        n["moment"] = f"{(m['true'].str.replace('-', ' ') == m.qwen.str.replace('-', ' ')).mean():.0%} of {len(m)} clips named correctly"
    except Exception:
        pass
    return n


def grab(path, t, out):
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(t), "-i", path, "-frames:v", "1", out], check=True)
    return out


# ----------------------------------------------------------------------------------------------- animations
def overlay(frame, layer, alpha):
    """Blend an RGBA text layer (x, y, array) onto the frame; position fixed, only alpha changes."""
    if alpha <= 0:
        return
    x, y, a = layer
    h, w = a.shape[:2]
    m = a[..., 3:4].astype(np.float32) / 255 * min(alpha, 1)
    roi = frame[y:y + h, x:x + w].astype(np.float32)
    frame[y:y + h, x:x + w] = (roi * (1 - m) + a[..., :3] * m).astype(np.uint8)


def text_layer(text, size, fill, x, y, bold=True, anchor="la", width=None):
    f = font(size, bold)
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines = [text] if not width else _lines(tmp, text, f, width)
    tw = int(max(tmp.textlength(l, font=f) for l in lines)) + 8
    th = len(lines) * (size + 14) + 10
    im = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, l in enumerate(lines):
        d.text((4 if anchor == "la" else tw // 2, i * (size + 14)), l, font=f, fill=fill + (255,), anchor="la" if anchor == "la" else "ma")
    x0 = x if anchor == "la" else x - tw // 2
    return (int(x0), int(y), np.asarray(im))


def _lines(d, text, f, width):
    out, line = [], ""
    for w in text.split():
        t = (line + " " + w).strip()
        if d.textlength(t, font=f) > width and line:
            out.append(line)
            line = w
        else:
            line = t
    return out + [line]


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def fade(t, a, b, d=0.6):
    """1 between a and b, fading in/out over d seconds."""
    return min(ease((t - a) / d), ease((b - t) / d))


S16 = 16                                                               # cv2 sub-pixel factor (shift=4)


def pt(x, y):
    return (int(round(x * S16)), int(round(y * S16)))


def disc(img, x, y, r, col, thick=-1):
    cv2.circle(img, pt(x, y), int(r * S16), col, thick, cv2.LINE_AA, 4)


def seg(img, a, b, col, w=2):
    cv2.line(img, pt(*a), pt(*b), col, w, cv2.LINE_AA, 4)


class Intro:
    """Opening animation: 'Can AI suggest better football tactics, and show them working in a game?'"""
    PX, X0, Y0 = 12.0, 330, 96                                         # 12 px per metre; pitch 1260 x 816

    def __init__(self):
        self.A0 = np.array([[2, -16], [10, 12], [18, -4], [27, 20], [14, -27]], float)
        self.A1 = self.A0 + [[7, 2], [8, -1], [12, 0], [9, -2], [8, 3]]
        self.D0 = np.array([[30, -13], [33, -1], [32, 12], [26, -24], [22, 5]], float)
        self.D1 = self.D0 + [[4, 1], [3, 0], [3, -1], [4, 2], [5, -2]]
        self.GK = np.array([49.0, 0.0])
        self.pitch = self._pitch()
        L = text_layer
        self.t_title1 = L("Can AI suggest better football tactics", 76, WHITE, W // 2, 380, anchor="ma")
        self.t_title2 = L("... and show them working in a game?", 60, TEAL, W // 2, 500, anchor="ma")
        self.caps = [(5.0, 12.4, L("1  A real attack, from tracking data", 40, WHITE, W // 2, 950, anchor="ma")),
                     (12.4, 17.4, L("2  The AI reads the play as a graph of players", 40, WHITE, W // 2, 950, anchor="ma")),
                     (17.4, 25.0, L("3  The AI suggests a change: close down the ball carrier", 40, GOLD, W // 2, 950, anchor="ma")),
                     (25.0, 40.0, L("4  ... and we test it in a football game simulator", 40, TEAL, W // 2, 950, anchor="ma"))]
        self.t_thr = L("THREAT", 26, MUTED, 1615, 300)
        self.t_thr2 = L("-0.038", 44, GOLD, 1615, 690)
        self.t_thr3 = L("best of 24 tested", 22, MUTED, 1615, 745, bold=False)
        self.t_ai = L("AI", 30, GOLD, 0, 0)
        self.t_name = text_layer(NAME, 24, WHITE, 0, 1022)
        self.t_name = (1800 - self.t_name[2].shape[1], 1022, self.t_name[2])
        g = Image.open(DEMO("whatif_grf.png")).convert("RGB") if os.path.exists(DEMO("whatif_grf.png")) else Image.new("RGB", (16, 9))
        g.thumbnail((1100, 640), Image.LANCZOS)
        self.grf = np.asarray(g)

    def m2p(self, p):
        return self.X0 + (p[0] + 52.5) * self.PX, self.Y0 + (p[1] + 34) * self.PX

    def _pitch(self):
        im = np.zeros((H, W, 3), np.uint8)
        im[:] = NAVY
        x0, y0, x1, y1 = self.X0, self.Y0, self.X0 + 105 * self.PX, self.Y0 + 68 * self.PX
        for k in range(10):                                            # mowing stripes
            c = (24, 92, 52) if k % 2 else (28, 104, 58)
            cv2.rectangle(im, (int(x0 + k * 10.5 * self.PX), int(y0)), (int(x0 + (k + 1) * 10.5 * self.PX), int(y1)), c, -1)
        lc = (215, 235, 220)
        P = lambda x, y: self.m2p((x, y))
        for a, b in [((-52.5, -34), (52.5, -34)), ((52.5, -34), (52.5, 34)), ((52.5, 34), (-52.5, 34)), ((-52.5, 34), (-52.5, -34)),
                     ((0, -34), (0, 34)), ((36, -20.16), (52.5, -20.16)), ((36, 20.16), (52.5, 20.16)), ((36, -20.16), (36, 20.16)),
                     ((-36, -20.16), (-52.5, -20.16)), ((-36, 20.16), (-52.5, 20.16)), ((-36, -20.16), (-36, 20.16))]:
            seg(im, P(*a), P(*b), lc, 2)
        cx, cy = P(0, 0)
        cv2.circle(im, pt(cx, cy), int(9.15 * self.PX * S16), lc, 2, cv2.LINE_AA, 4)
        return im

    def state(self, t):
        a = self.A0 + (self.A1 - self.A0) * ease((t - 5) / 7)
        d = self.D0 + (self.D1 - self.D0) * ease((t - 5.5) / 7)
        closer = ease((t - 19.5) / 3.0)
        target = a[2] + [3.0, 1.5]
        d[4] = d[4] + (target - d[4]) * closer
        holders = [(0, 0, 6.5), (0, 1, 8.5), (1, 1, 9.5), (1, 2, 11.5)]   # ball path: from, to, finish time
        ball = a[0]
        for f, to, end in holders:
            if t >= end - 2:
                ball = a[f] + (a[to] - a[f]) * ease((t - (end - 2)) / 2)
        return a, d, ball + [1.2, 0.9], closer

    def frame(self, t):
        f = np.zeros((H, W, 3), np.uint8)
        f[:] = NAVY
        title = fade(t, 0.3, 4.4, 0.8)
        pitch_a = fade(t, 4.4, 99, 0.8)
        if pitch_a > 0:
            p = self.pitch.copy()
            a, d, ball, closer = self.state(t)
            g = ease((t - 12.6) / 1.2) * (1 - 0.6 * ease((t - 19) / 1.0))
            if g > 0:                                                  # graph edges between near players
                layer = p.copy()
                pts = [self.m2p(q) for q in np.vstack([a, d])]
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        dist = np.linalg.norm(np.vstack([a, d])[i] - np.vstack([a, d])[j])
                        if dist < 17:
                            seg(layer, pts[i], pts[j], (120, 230, 255) if (i < 5) == (j < 5) else (255, 200, 120), 2)
                p = cv2.addWeighted(layer, g, p, 1 - g, 0)
            hl = fade(t, 17.6, 99, 0.6)
            if hl > 0:                                                 # suggestion: pulsing ring + dashed arrow
                dx, dy = self.m2p(self.D1[4] if closer == 0 else d[4])
                r = 26 + 6 * math.sin((t - 17.6) * 5)
                ring = p.copy()
                disc(ring, dx, dy, r, GOLD, 3)
                if closer < 1:
                    s, e = self.m2p(self.D1[4]), self.m2p(self.state(30)[0][2] + [3.0, 1.5])
                    for k in range(0, 10, 2):
                        q0 = (s[0] + (e[0] - s[0]) * k / 10, s[1] + (e[1] - s[1]) * k / 10)
                        q1 = (s[0] + (e[0] - s[0]) * (k + 1) / 10, s[1] + (e[1] - s[1]) * (k + 1) / 10)
                        seg(ring, q0, q1, GOLD, 4)
                p = cv2.addWeighted(ring, hl, p, 1 - hl, 0)
            for q in a:
                x, y = self.m2p(q)
                disc(p, x, y, 15, (255, 255, 255))
                disc(p, x, y, 12.5, RED)
            for q in list(d) + [self.GK]:
                x, y = self.m2p(q)
                disc(p, x, y, 15, (255, 255, 255))
                disc(p, x, y, 12.5, BLUE if q is not self.GK else (120, 120, 140))
            bx, by = self.m2p(ball)
            disc(p, bx, by, 8, (255, 255, 255))
            disc(p, bx, by, 8, (30, 30, 30), 2)
            f = cv2.addWeighted(p, pitch_a, f, 1 - pitch_a, 0)
            thr = fade(t, 17.6, 25.2, 0.6)
            if thr > 0:                                                # threat meter
                m = f.copy()
                cv2.rectangle(m, (1615, 345), (1665, 675), CARD, -1)
                v = 0.78 - 0.30 * closer
                cv2.rectangle(m, (1615, int(675 - 330 * v)), (1665, 675), RED if closer < 0.5 else GOLD, -1)
                f = cv2.addWeighted(m, thr, f, 1 - thr, 0)
                overlay(f, self.t_thr, thr)
                overlay(f, self.t_thr2, thr * closer)
                overlay(f, self.t_thr3, thr * closer)
            for a0, b0, lay in self.caps:
                ca = fade(t, a0, b0, 0.5)
                if ca > 0:
                    cv2.rectangle(f, (300, 930), (1620, 1010), NAVY, -1)
                    overlay(f, lay, ca)
            ga = fade(t, 26.0, 99, 0.8)
            if ga > 0:                                                 # GRF picture card fades in over the pitch
                f = (f.astype(np.float32) * (1 - 0.65 * ga)).astype(np.uint8)
                h, w = self.grf.shape[:2]
                x, y = (W - w) // 2, 140
                roi = f[y - 12:y + h + 12, x - 12:x + w + 12].astype(np.float32)
                box = roi.copy()
                box[:] = (250, 251, 253)
                box[12:12 + h, 12:12 + w] = self.grf
                f[y - 12:y + h + 12, x - 12:x + w + 12] = (roi * (1 - ga) + box * ga).astype(np.uint8)
                cv2.rectangle(f, (300, 930), (1620, 1010), NAVY, -1)
                overlay(f, self.caps[-1][2], fade(t, self.caps[-1][0], self.caps[-1][1], 0.5))
        if title > 0:
            overlay(f, self.t_title1, title)
            overlay(f, self.t_title2, ease((t - 1.8) / 0.8) * title)
        overlay(f, self.t_name, 1.0)
        return f


ARCH = [("Broadcast video", "NEW: a vision model (RF-DETR + tracker) turns TV video into player positions", (860, 250)),
        ("Data", "91 real matches of tracking data: DFL, World Cup 2022 (PFF), A-League (SkillCorner)", None),
        ("Key States", "keep only the moments that matter: passes, shots, ball wins, pressure", None),
        ("Graphs", "each moment becomes a network: 22 players + ball, linked to their neighbours", None),
        ("Models", "graph neural networks learn movement and passing; xG and threat value each moment", None),
        ("What-if", "change one thing (a player, a line, a pass) and predict what happens next", None),
        ("GRF game", "play it out in Google Research Football with team models in real team styles", None),
        ("Best tactic", "rank the changes by threat: the coach sees the best option, with its uncertainty", None)]


class Arch:
    """Architecture: a glowing arrow travels from layer to layer; the active layer lights up, its description fades in."""

    def __init__(self, title=True):
        bw, gap, y = 190, 42, 520
        x0 = (W - (7 * bw + 6 * gap)) // 2
        self.boxes = [(860, 270, 860 + bw, 380)] + [(x0 + i * (bw + gap), y, x0 + i * (bw + gap) + bw, y + 120) for i in range(7)]
        self.boxes[0] = (self.boxes[1][0], 290, self.boxes[1][0] + 300, 400)
        self.paths = [None] + [((self.boxes[1][0] + bw // 2, 400), (self.boxes[1][0] + bw // 2, y))] + \
                     [((self.boxes[i][2], y + 60), (self.boxes[i + 1][0], y + 60)) for i in range(1, 7)]
        sl = Slide("How it works", "Architecture: from video and tracking data to the best tactic", 5 if title else None)
        base = sl.base()
        self.states = []
        for k in range(len(ARCH) + 1):
            im = base.copy()
            d = ImageDraw.Draw(im)
            for i, (name, _, _) in enumerate(ARCH):
                x0_, y0_, x1_, y1_ = self.boxes[i]
                on, done = i == k - 1, i < k - 1
                d.rounded_rectangle([x0_, y0_, x1_, y1_], 18, fill=(70, 50, 22) if on else CARD,
                                    outline=GOLD if on else (TEAL if done else LINE), width=4 if on else 2)
                d.text(((x0_ + x1_) // 2, y0_ + 16), f"{i}" if i else "VLM", font=font(22, True), fill=GOLD if on else MUTED, anchor="ma")
                wrap(d, name, (x0_ + 18, y0_ + 50), x1_ - x0_ - 30, font(27, True), WHITE, 4)
            for i, p in enumerate(self.paths):
                if p:
                    seg_col = TEAL if i < k else LINE
                    d.line([p[0], p[1]], fill=seg_col, width=4)
            if k:
                name, desc, _ = ARCH[k - 1]
                d.rounded_rectangle([200, 730, 1720, 900], 18, fill=CARD, outline=GOLD, width=2)
                d.text((240, 752), f"{'VLM' if k == 1 else k - 1}  ·  {name}", font=font(34, True), fill=GOLD)
                wrap(d, desc, (240, 808), 1440, font(30), WHITE, 8)
            self.states.append(np.asarray(im).copy())

    def frame(self, t, total):
        step = total / len(ARCH)
        k = min(int(t / step), len(ARCH) - 1)
        u = (t - k * step) / step
        cur = self.states[k + 1]
        prev = self.states[k]
        a = ease((t - k * step) / 0.6)
        f = cv2.addWeighted(cur, a, prev, 1 - a, 0) if a < 1 else cur.copy()
        p = self.paths[k]
        if p and u < 0.3:                                              # glowing arrow head runs along the link
            s = ease(u / 0.3)
            x = p[0][0] + (p[1][0] - p[0][0]) * s
            y = p[0][1] + (p[1][1] - p[0][1]) * s
            g = f.copy()
            disc(g, x, y, 16, GOLD)
            f = cv2.addWeighted(g, 0.45, f, 0.55, 0)
            disc(f, x, y, 8, (255, 250, 220))
        return f


# ----------------------------------------------------------------------------------------------- scenes
def scenes():
    V = vision_numbers()
    shots = [grab(O(f"whatif/J03WOH_3500_3610_att_as_{t}_grf_h264.mp4"), 7, os.path.join(WORK, f"snap_{t}.png"))
             for t in ("Argentina", "Croatia", "France")]
    det = O("vision_poc/rfdetr/SNGS-117_000301.jpg")
    det = det if os.path.exists(det) else sorted(glob.glob(O("vision_poc/rfdetr/*.jpg")))[0]
    S = []

    S.append(("anim", "intro", None,
              "Football is a game of space and timing. A team that moves together can close a gap, or open one. "
              "So here is our question. Can A I suggest better football tactics, and show them working in a game? "
              "First, we take a real attack, from tracking data. The A I reads the play as a graph: who is near whom, and where the ball is. "
              "Then it suggests a change. Here: close down the ball carrier. The threat of the attack drops. "
              "And finally, we test that change inside a football game simulator, to see it play out."))

    s = Slide("", "", None)
    s.step(lambda d, im: (d.text((120, 330), "DYNAMIC TACTICAL AI", font=font(30, True), fill=TEAL),
                          d.text((120, 380), "in Football", font=font(110, True), fill=WHITE),
                          wrap(d, "Learning tactics from real matches, and testing them in a game simulator", (124, 530), 1300, font(40), MUTED)))
    s.step(lambda d, im: (d.line([124, 650, 700, 650], fill=RED, width=4),
                          d.text((124, 680), "Srinivasan Dhakshanamoorthy", font=font(34, True), fill=WHITE),
                          d.text((124, 730), "Supervisor: Professor Ahmed Azough  ·  ESILV · De Vinci Research Center", font=font(28), fill=MUTED),
                          d.text((124, 780), "Progress video · November 2026", font=font(28), fill=MUTED)))
    S.append(("slide", s, None,
              "Welcome. This is Dynamic Tactical AI in Football, a research project at E S I L V and the De Vinci Research Center, "
              "by Srinivasan Dhakshanamoorthy, supervised by Professor Ahmed Azough. In the next ten minutes we show the idea, "
              "how it is built, the results so far, and the newest part: reading positions straight from video."))

    s = Slide("Our idea", "Learn from real matches. Suggest. Test in a game.", 2)
    for i, (n, t, sub, c) in enumerate([("1", "Learn", "Graph models learn how real teams move and pass, from tracking data.", TEAL),
                                        ("2", "Suggest", "Change one thing in a real moment: a player's position, the defensive line, a pass.", GOLD),
                                        ("3", "Test", "Play the change out in a football simulator and measure the threat it creates or removes.", GREEN)]):
        s.step(card(120 + i * 570, 270, 530, 470, n, t, sub, c, 110))
    s.step(banner(120, 790, 1680, "Goal: tactical advice a coach can trust: tested on real data, measured, and explained in plain numbers."))
    S.append(("slide", s, None,
              "Our idea has three steps. First, learn: graph models learn how real teams move and pass, from tracking data. "
              "Second, suggest: take a real moment and change one thing, for example a player's position, the height of the defensive line, or an earlier pass. "
              "Third, test: play that change out in a football simulator, and measure the threat it creates or removes. "
              "The goal is advice a coach can trust: tested, measured and explained."))

    s = Slide("Our research", "What others did, and the gap we fill", 3)
    for i, (y, t, sub, c) in enumerate([("2024", "TacticAI", "DeepMind and Liverpool: graph networks on corner kicks; experts preferred its suggestions.", TEAL),
                                        ("GNN", "Graphs over time", "Players as nodes in a graph that changes every moment: the right shape for team play.", BLUE),
                                        ("GRF", "Football simulator", "Google Research Football: a full 11 v 11 game where AI teams can play.", GREEN),
                                        ("2025", "Vision models", "Detectors and vision-language models read players straight from TV video.", GOLD)]):
        s.step(card(120 + i * 425, 260, 395, 440, y, t, sub, c, 64))
    s.step(banner(120, 750, 1680, "Our gap: open play, not just corners. Real data into a simulator, with what-if tests and honest uncertainty."))
    S.append(("slide", s, None,
              "We build on four lines of research. DeepMind's TacticAI, made with Liverpool, used graph networks on corner kicks, "
              "and expert analysts preferred its suggestions. Graph models over time treat players as nodes in a network that changes every moment. "
              "Google Research Football gives us a full eleven a side game where A I teams can play. "
              "And new vision models can read players straight from television video. The gap we fill is open play, "
              "not just corners: real data into a simulator, with what-if tests and honest uncertainty."))

    S.append(("anim", "arch", None,
              "Here is the architecture, layer by layer. A new branch starts from broadcast video: a vision model turns television frames into player positions. "
              "Layer one is the data: ninety-one real matches of tracking data. "
              "Layer two finds the key states, the moments that matter, like passes, shots and ball wins. "
              "Layer three turns each moment into a graph: twenty-two players and the ball, linked to their neighbours. "
              "Layer four holds the models: graph neural networks for movement and passing, plus expected goals and threat. "
              "Layer five is the what-if engine: change one thing, and predict what happens next. "
              "Layer six plays it out in the G R F game, with team models in real team styles. "
              "And the last layer ranks the changes by threat, so the coach sees the best tactic, with its uncertainty."))

    s = Slide("Datasets", "91 real matches, plus video", 6)
    for i, (n, t, sub, c) in enumerate([("7", "Bundesliga", "DFL open data. Optical tracking, 10 times a second.", TEAL),
                                        ("64", "World Cup 2022", "PFF. 32 national teams; Argentina, France, Croatia 7 matches each.", BLUE),
                                        ("20", "A-League", "SkillCorner broadcast tracking: a second, different source.", GREEN),
                                        ("49", "SoccerNet clips", "NEW: 30-second TV clips with labelled positions, to test video to positions.", GOLD)]):
        s.step(card(120 + i * 425, 260, 395, 470, n, t, sub, c, 110))
    s.step(banner(120, 780, 1680, "Real data is the reference. The simulator is judged against reality, never the other way round."))
    S.append(("slide", s, None,
              "The data. Seven Bundesliga matches from the open D F L dataset, tracked ten times a second. "
              "Sixty-four World Cup twenty twenty-two matches from P F F, with thirty-two national teams. "
              "Twenty A-League matches from SkillCorner, a second and very different source. "
              "And new for this phase: forty-nine SoccerNet television clips, where every player's true position is labelled, "
              "so we can measure how well video can replace tracking data. Real data is always the reference."))

    s = Slide("Key States", "Keep the moments that matter", 7)

    def timeline(d, im):
        x0, x1, y = 160, 1760, 400
        d.rounded_rectangle([x0, y, x1, y + 60], 12, fill=CARD)
        rng = np.random.default_rng(3)
        for k in range(140):
            x = x0 + 10 + k * (x1 - x0 - 20) / 140
            d.line([x, y + 14, x, y + 46], fill=LINE, width=3)
        for x, lab, c in [(330, "pass", TEAL), (520, "pass", TEAL), (700, "pressure", GOLD), (980, "ball won", GREEN),
                          (1180, "pass", TEAL), (1420, "shot", RED), (1600, "turnover", BLUE)]:
            d.rounded_rectangle([x - 9, y - 10, x + 9, y + 70], 6, fill=c)
            d.text((x, y + 86), lab, font=font(24, True), fill=c, anchor="ma")
        d.text((x0, y - 60), "One match = about 54,000 frames", font=font(30, True), fill=MUTED)
    s.step(timeline)
    s.step(bullets(160, 600, ["We keep only key states: passes, shots, ball wins, pressure and turnovers.",
                              "The models look at a window of 3 key states, the build-up to each moment.",
                              "Fewer, richer moments: faster to learn, and easy for a coach to read."], 1500, 32))
    S.append(("slide", s, None,
              "A full match has about fifty-four thousand frames, and most of them are routine. "
              "So we keep only the key states: passes, shots, ball wins, pressure and turnovers. "
              "The models look at a window of three key states, the build-up to each moment. "
              "Fewer but richer moments make learning faster, and they are easy for a coach to read."))

    s = Slide("Graphs", "Every moment becomes a network", 8)
    s.step(picture(O("graph_snapshot.png"), 120, 240, 980, 730))
    s.step(bullets(1160, 270, ["23 nodes: 22 players and the ball.",
                               "Links to nearby team-mates, nearby opponents and the ball (Delaunay).",
                               "Each player carries position, speed, role and distance to the ball.",
                               "The graph network reads it like a map of who can reach whom."], 620, 30))
    S.append(("slide", s, None,
              "Every key state becomes a network. There are twenty-three nodes: twenty-two players and the ball. "
              "Each player is linked to nearby team-mates, nearby opponents and the ball, using a Delaunay triangulation. "
              "Each node carries position, speed, role and distance to the ball. "
              "The graph network reads this like a map of who can reach whom, which is exactly what tactics are about."))

    s = Slide("Models", "Which model wins, and how we choose", 9)
    s.step(table(120, 250, ["Model", "Count", "Best result"],
                 [["Movement (graph network)", "77 trainings", "1.13 m error; 3-model ensemble 1.10 m"],
                  ["Pass choice", "6", "learned pass model beats the simple baseline"],
                  ["Expected goals (xG)", "12", "shot quality for every chance"],
                  ["Threat value", "6", "how dangerous each moment is"],
                  ["Team styles", "4", "Argentina, France, Croatia, Bundesliga"]], [520, 260, 900], 28, hi=1))
    s.step(banner(120, 640, 1680, "Fair test: every model is scored on a match it never saw (7 rounds). A new model wins only if it is "
                                  "better on average, the 95% interval excludes zero, and it wins at least 6 of the 7 matches. "
                                  "The 3-model ensemble won 7 of 7.", TEAL, 29))
    S.append(("slide", s, None,
              "The models. For movement we ran seventy-seven trainings of graph networks. The best single model predicts where players go "
              "with an error of one point one three metres, and an ensemble of three models brings it to one point one zero. "
              "We also have six pass models, twelve expected goals models, six threat models, and four team styles. "
              "Every model is tested on a match it never saw, seven times over. A new model only wins if it is better on average, "
              "its ninety-five percent interval excludes zero, and it wins at least six of the seven matches. The ensemble won all seven."))

    S.append(("clip", O("pass2_3min_GRF_h264.mp4"), (0, "Real match replayed in GRF  ·  Fortuna Düsseldorf v Jahn Regensburg  ·  real player names"),
              "This is a real Bundesliga moment, rebuilt inside the Google Research Football game, with the real player names. "
              "Every player starts where the tracking data says he was. From here, the game engine takes over, "
              "so we can replay real football and then change it. This is the bridge from real data to a game."))

    s = Slide("Model v model", "Team styles playing each other", 11)
    for i, (p, t) in enumerate(zip(shots, ("as Argentina", "as Croatia", "as France"))):
        s.step(picture(p, 120 + i * 570, 250, 540, 360, f"Same attack, played {t}"))
    s.step(bullets(160, 680, ["Each team model learned from that team's real World Cup matches.",
                              "Same starting moment, three styles: the passing and the runs change.",
                              "In the league, the Argentina and France models play full GRF matches."], 1500, 30))
    S.append(("slide", s, None,
              "Next, model against model. Each team model learned from that team's real World Cup matches. "
              "Here the same attack starts from the same moment, but is played in three styles: as Argentina, as Croatia, and as France. "
              "You can see the passing and the runs change with the style. In our small league, these team models play full matches."))

    S.append(("clip", O("team_league/Argentina_vs_France_h264.mp4"), (62, "Model v model in GRF  ·  Argentina model v France model"),
              "Here is the Argentina model playing the France model in G R F. Each side moves the way its real team tends to move: "
              "how high it presses, how quickly it passes forward, and where it attacks."))

    s = Slide("What-if", "5,180 simulated runs: what really helps the defence?", 13)
    s.step(picture(O("whatif/batch_chart.png"), 120, 240, 1020, 740))
    s.step(table(1180, 260, ["Change", "Threat"], [["Close the ball carrier", "-0.038"], ["Whole team 3 m closer", "-0.033"],
                                                   ["3 nearest 3 m closer", "-0.033"], ["Keeper off his line", "-0.015"],
                                                   ["Back four 3 m higher", "+0.024"]], [420, 200], 27, hi=1))
    s.step(banner(1180, 640, 620, "Backfires: a back four 3 m higher raises threat in 14 of 20 moments.", RED, 28))
    S.append(("slide", s, None,
              "Now the what-if engine. We tested twenty-four changes on twenty real moments, ten times each, five thousand one hundred and eighty runs in total. "
              "The best help for the defence is to close down the ball carrier: threat falls by zero point zero three eight. "
              "Moving the whole team three metres closer, or the three nearest players, comes next. "
              "And one change backfires: pushing the back four three metres higher raises the threat in fourteen of twenty moments."))

    S.append(("clip", O("whatif/J03WOH_3500_3610_line_higher_grf_h264.mp4"), (0, "What-if in GRF  ·  left: no change   ·   right: back four 3 m higher"),
              "Here it is in the game. On the left, the real moment with no change. On the right, the back four starts three metres higher. "
              "The space behind the line grows, and the attack finds it. The numbers and the game tell the same story."))

    s = Slide("Interactive viewer", "See it in the browser: real moment and what-if", None)
    s.step(picture(DEMO("viewer_real.png"), 120, 240, 820, 560, "Real moment, real names"))
    s.step(picture(DEMO("viewer_whatif.png"), 980, 240, 820, 560, "Same moment after a what-if change"))
    s.step(banner(120, 840, 1680, "A 3D web viewer (Three.js): play any key moment, switch between real and what-if, and read each player's name.", TEAL, 28))
    S.append(("slide", s, None,
              "To make this usable for a coach, we built an interactive viewer that runs in the browser. "
              "On the left is a real moment with the real player names. On the right is the same moment after a what-if change. "
              "A coach can play any key moment, switch between real and what-if, and see exactly who moved, and why it matters."))

    s = Slide("VLM: video to positions", "Can TV video replace paid tracking data?", 15)
    s.step(picture(det, 120, 240, 860, 520, "RF-DETR on a SoccerNet frame"))
    s.step(table(1020, 240, ["Test (490 frames)", "Result"], [["People found", "91%"], ["Boxes correct", "88%"], ["Team from shirt", "97%"],
                                                             ["Ball found", "50%"], ["Pitch position", "0.34 m"], ["Speed", "0.5 s / frame"]],
                 [430, 350], 27))
    s.step(banner(120, 800, 1680, f"Tracking full clips: {V['track']}.  Jersey numbers: {V['jersey']}.  Game situation: {V['moment']}.", TEAL, 26))
    S.append(("slide", s, None,
              "Now the professor's new idea: read the players straight from television video. We tested free, open-source models on four hundred and ninety SoccerNet frames. "
              "The R F DETR detector finds ninety-one percent of the people, with eighty-eight percent of boxes correct, at half a second per frame. "
              "Shirt colours give the right team ninety-seven percent of the time, and positions land within a third of a metre. "
              "The weak point is the ball, found only half the time, so it needs a small fine-tune. "
              "On full clips, the tracker follows most players, but identities still switch too often. "
              "A vision-language model reads jersey numbers better than plain text reading, but it is slow, so it helps only where a detector cannot."))

    s = Slide("Noise test", "How accurate must video positions be?", 16)
    s.step(picture(O("noise_test/noise_chart.png"), 120, 240, 1060, 740))
    s.step(table(1220, 260, ["Positions from", "Error"], [["True tracking", "1.14 m"], ["Good vision system", "1.49 m"],
                                                          ["Detector + our mapping", "1.98 m"], ["Vision-language model only", "2.80 m"]], [380, 200], 27))
    s.step(banner(1220, 620, 580, "Target: positions within 0.5 m. Jitter hurts most; missing players or ball hurt little.", GOLD, 27))
    S.append(("slide", s, None,
              "How good must the video be? We damaged the true tracking on purpose, and measured how much the movement model suffers. "
              "With clean data the error is one point one four metres. Position jitter hurts most: half a metre of jitter already lifts it to one point four four. "
              "Missing players, a missing ball or swapped identities hurt much less. A vision-language model alone gives two point eight metres, "
              "so the target is clear: positions within half a metre, from a detector, a tracker and a good pitch mapping, with the V L M as a helper."))

    s = Slide("Putting it together", "From broadcast video to the best tactic", 17)
    s.step(flow(140, 290, ["TV video", "Detector + tracker", "Pitch mapping", "Graph", "Graph models", "What-if in GRF", "Best tactic"]))
    s.step(card(140, 470, 800, 250, None, "Example result", "Close down the ball carrier: threat -0.038, the best of 24 changes. "
                                                            "Pushing the back four higher backfires (+0.024).", GOLD))
    s.step(card(980, 470, 800, 250, None, "Next steps", "Fine-tune the ball detector on a free GPU · stronger pitch mapping · "
                                                         "fewer identity switches · what-if on matches seen only on video.", TEAL))
    s.step(banner(140, 770, 1640, "Any match on TV could become tactical analysis: no paid tracking data needed.", GREEN, 32))
    S.append(("slide", s, None,
              "Putting it all together. Television video goes through a detector and tracker, then our pitch mapping, into the same graphs and models, "
              "and the what-if engine tests changes in the G R F game. The result is a ranked list of tactics, like: close down the ball carrier. "
              "Next, we fine-tune the ball detector on a free G P U, make the pitch mapping more robust, cut identity switches, "
              "and run what-if tests on matches seen only on video. The promise: any match on television could become tactical analysis."))

    s = Slide("", "", None)
    s.step(lambda d, im: (d.text((W // 2, 400), "Thank you", font=font(120, True), fill=WHITE, anchor="ma"),
                          d.text((W // 2, 560), "Questions and feedback welcome", font=font(40), fill=TEAL, anchor="ma"),
                          d.text((W // 2, 650), "Dynamic Tactical AI in Football  ·  ESILV · De Vinci Research Center", font=font(28), fill=MUTED, anchor="ma")))
    S.append(("slide", s, None, "Thank you for watching. We welcome your questions and feedback."))
    return S


# ----------------------------------------------------------------------------------------------- voice and render
def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def voice(k, text):
    mp3 = os.path.join(WORK, f"voice_{k:02d}.mp3")
    for attempt in range(8):
        if os.path.exists(mp3) and os.path.getsize(mp3) > 1000:
            return mp3
        try:
            import edge_tts
            asyncio.run(edge_tts.Communicate(text, VOICE, rate=RATE).save(mp3))
        except Exception as e:
            print(f"scene {k}: voice attempt {attempt + 1} failed ({type(e).__name__})", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"no voice for scene {k}")


def clip_reader(path, start, box, slow=1.0):
    x, y, w, h = box
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(start), "-i", path,
                          "-vf", f"setpts={slow:.4f}*PTS,fps={FPS},scale={w}:{h}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            p.kill()
            return
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)


def clip_box(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    cw, ch = map(int, r.stdout.strip().split(","))
    w = 1680
    h = int(round(w * ch / cw / 2) * 2)
    if h > 800:
        h = 800
        w = int(round(h * cw / ch / 2) * 2)
    return ((W - w) // 2, 120 + (800 - h) // 2, w, h)


def clip_bg(caption):
    s = Slide("", "", None)
    im = s.base()
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([60, 940, W - 60, 1000], 14, fill=CARD)
    d.rectangle([60, 948, 66, 992], fill=GOLD)
    d.text((96, 950), caption, font=font(32, True), fill=WHITE)
    return np.asarray(im).copy()


def scene_frames(sc, dur):
    """Yield every frame of one scene (exactly round(dur * FPS) frames)."""
    kind, obj, extra, _ = sc
    n = int(round(dur * FPS))
    if kind == "anim" and obj == "intro":
        a = Intro()
        for i in range(n):
            yield a.frame(i / FPS * 34.0 / max(dur, 34.0) if dur > 34 else i / FPS)
    elif kind == "anim":
        a = Arch()
        for i in range(n):
            yield a.frame(i / FPS, dur)
    elif kind == "slide":
        st = obj.states()
        lead = 0.8                                                     # first build visible this long before the next
        per = (dur - lead) / len(st)
        for i in range(n):
            t = i / FPS
            k = min(int(max(t - lead, 0) / per), len(st) - 1) if len(st) > 1 else 0
            k = min(int(t / per), len(st) - 1)
            into = t - k * per
            if k > 0 and into < XF:
                a = ease(into / XF)
                yield cv2.addWeighted(st[k], a, st[k - 1], 1 - a, 0)
            else:
                yield st[k]
    else:                                                              # clip on a slide background
        start, cap = extra
        bg = clip_bg(cap)
        x, y, w, h = clip_box(obj)
        slow = min(max(1.0, dur / max(duration(obj) - start - 0.2, 1)), 1.6)   # never loops: stretch a short clip a little
        rd = clip_reader(obj, start, (x, y, w, h), slow)
        c = None
        for i in range(n):
            f = bg.copy()
            nxt = next(rd, None)
            c = nxt if nxt is not None else c                          # past the end: hold the last frame (no repeat)
            if c is not None:
                f[y:y + h, x:x + w] = c
            yield f


def plan(S):
    out = []
    for k, sc in enumerate(S):
        mp3 = voice(k, sc[3])
        d = duration(mp3) + 1.2
        if sc[0] == "anim" and sc[1] == "intro":
            d = max(d, 32.0)
        out.append((sc, mp3, d))
    return out


def render(P, out):
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS),
                            os.path.join(WORK, "video_only.mp4")], stdin=subprocess.PIPE)
    prev, total = None, 0
    for k, (sc, mp3, d) in enumerate(P):
        t0 = time.time()
        for i, f in enumerate(scene_frames(sc, d)):
            if prev is not None and i < XF * FPS:                      # cross-fade from the previous scene's last frame
                a = ease(i / (XF * FPS))
                f = cv2.addWeighted(f, a, prev, 1 - a, 0)
            enc.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
            last = f
        prev = last
        total += d
        print(f"scene {k:2d} {sc[0]:5s} {d:5.1f} s  (rendered in {time.time() - t0:4.0f} s)  total {total / 60:4.1f} min", flush=True)
    enc.stdin.close()
    enc.wait()
    wavs = []                                                          # each scene's voice, 0.5 s in, padded to the scene length
    for k, (sc, mp3, d) in enumerate(P):
        w = os.path.join(WORK, f"a{k:02d}.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp3, "-af", f"adelay=500|500,apad,atrim=0:{d:.3f}", "-ar", "48000", "-ac", "2", w], check=True)
        wavs.append(w)
    lst = os.path.join(WORK, "audio.txt")
    open(lst, "w").write("".join(f"file '{w}'\n" for w in wavs))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "pcm_s16le", os.path.join(WORK, "voice.wav")], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(WORK, "video_only.mp4"), "-i", os.path.join(WORK, "voice.wav"),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out], check=True)
    print(f"wrote {out}  ({total / 60:.1f} min)")


def gif():
    a = Arch(title=True)
    total, fps = 20.0, 12
    frames = [Image.fromarray(cv2.resize(a.frame(i / fps, total), (960, 540), interpolation=cv2.INTER_AREA)) for i in range(int(total * fps))]
    frames += [frames[-1]] * fps * 2
    pal = frames[len(frames) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(os.path.join(PRES, "architecture_animation.gif"), save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", os.path.join(PRES, "architecture_animation.mp4")], stdin=subprocess.PIPE)
    for i in range(int((total + 2) * FPS)):
        enc.stdin.write(a.frame(min(i / FPS, total - 0.01), total).tobytes())
    enc.stdin.close()
    enc.wait()
    print("wrote architecture_animation.gif and .mp4")


def pptx(S):
    from pptx import Presentation
    from pptx.util import Emu
    pr = Presentation()
    pr.slide_width, pr.slide_height = Emu(12192000), Emu(6858000)
    sx = 12192000 / W
    for k, sc in enumerate(S):
        sl = pr.slides.add_slide(pr.slide_layouts[6])
        if sc[0] == "slide":
            img = sc[1].states()[-1]
        elif sc[1] == "arch":
            img = Arch().states[-1]
        elif sc[0] == "anim":
            img = Intro().frame(23.0)
        else:
            img = clip_bg(sc[2][1])
        p = os.path.join(WORK, f"slide_{k:02d}.png")
        Image.fromarray(img).save(p)
        sl.shapes.add_picture(p, 0, 0, pr.slide_width, pr.slide_height)
        if sc[0] == "clip":
            x, y, w, h = clip_box(sc[1])
            poster = grab(sc[1], sc[2][0] + 2, os.path.join(WORK, f"poster_{k:02d}.png"))
            sl.shapes.add_movie(sc[1], Emu(int(x * sx)), Emu(int(y * sx)), Emu(int(w * sx)), Emu(int(h * sx)), poster_frame_image=poster, mime_type="video/mp4")
        sl.notes_slide.notes_text_frame.text = sc[3]
    out = os.path.join(PRES, "Dynamic_Tactical_AI_Video_v2_slides.pptx")
    pr.save(out)
    print("wrote", out)


def stills(S):
    pics = []
    for k, sc in enumerate(S):
        if sc[0] == "slide":
            img = sc[1].states()[-1]
        elif sc[1] == "intro":
            a = Intro()
            img = np.hstack([np.vstack([a.frame(2.5), a.frame(9)]), np.vstack([a.frame(15), a.frame(23)])])
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        elif sc[1] == "arch":
            img = Arch().frame(4 * 6.0 + 1.5, 8 * 6.0)
        else:
            img = clip_bg(sc[2][1])
            x, y, w, h = clip_box(sc[1])
            c = next(clip_reader(sc[1], sc[2][0] + 3, (x, y, w, h)))
            img[y:y + h, x:x + w] = c
        Image.fromarray(img).save(os.path.join(WORK, f"still_{k:02d}.png"))
        pics.append(cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA))
    while len(pics) % 4:
        pics.append(np.zeros_like(pics[0]))
    sheet = np.vstack([np.hstack(pics[i:i + 4]) for i in range(0, len(pics), 4)])
    Image.fromarray(sheet).save(os.path.join(WORK, "sheet.png"))
    print("wrote", os.path.join(WORK, "sheet.png"), f"({len(S)} scenes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true", help="only one picture per scene (quick check)")
    ap.add_argument("--no-extras", action="store_true", help="skip the GIF and the PowerPoint")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    S = scenes()
    if a.stills:
        stills(S)
        return
    P = plan(S)
    print(f"voice ready: {sum(d for _, _, d in P) / 60:.1f} min in {len(P)} scenes", flush=True)
    render(P, OUT)
    if not a.no_extras:
        gif()
        pptx(S)


if __name__ == "__main__":
    main()
