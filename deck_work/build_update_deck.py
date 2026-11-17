"""7-slide De Vinci / ESILV deck: what is new since the 3 Nov meeting. Last slide: 8 s model v model clip.
Usage: python deck_work\\build_update_deck.py  ->  presentations/Dynamic_Tactical_AI_Update_17Nov_2026.pptx
"""
import os
import subprocess

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = lambda p: os.path.join(ROOT, "outputs", p)
RED, INK, GREY, PALE, GREEN, GOLD = "CF1748", "19191E", "6E6E76", "F5F6F7", "1E8C5A", "B7791F"
LOGO = os.path.join(ROOT, "deck_work", "demo", "esilv_logo.png")
OUT = os.path.join(ROOT, "presentations", "Dynamic_Tactical_AI_Update_17Nov_2026.pptx")

pr = Presentation()
pr.slide_width, pr.slide_height = Inches(13.333), Inches(7.5)


def rect(s, x, y, w, h, col, shape=MSO_SHAPE.RECTANGLE, line=None):
    r = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    r.fill.solid()
    r.fill.fore_color.rgb = RGBColor.from_string(col)
    if line:
        r.line.color.rgb = RGBColor.from_string(line)
    else:
        r.line.fill.background()
    r.shadow.inherit = False
    return r


def text(s, x, y, w, h, t, size=14, col=INK, bold=False, align=PP_ALIGN.LEFT):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(t.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, "Calibri"
        r.font.color.rgb = RGBColor.from_string(col)
    return tb


def slide(title, n):
    s = pr.slides.add_slide(pr.slide_layouts[6])
    rect(s, 0, 0, 0.09, 7.5, RED)
    rect(s, 0.46, 0, 0.02, 7.5, RED)
    text(s, 0.1, 6.9, 0.34, 0.3, str(n), 11, GREY, True, PP_ALIGN.CENTER)
    text(s, 1.1, 0.25, 11, 0.3, "DYNAMIC TACTICAL AI IN FOOTBALL", 11, GREY, True)
    if title:
        text(s, 1.1, 0.6, 11.3, 0.6, title, 26, INK, True)
        rect(s, 1.1, 1.32, 11.25, 0.015, "DCDCE1")
    if os.path.exists(LOGO):
        s.shapes.add_picture(LOGO, Inches(11.27), Inches(6.79), Inches(1.46), Inches(0.58))
    return s


def pic(s, path, x, y, w=None, h=None, cap=None):
    if os.path.exists(path):
        s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)
    if cap:
        text(s, x, y + (h or 3) + 0.05, w or 5, 0.3, cap, 11, GREY)


def table(s, x, y, w, rows, widths, size=12, hi=None):
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(0.36 * len(rows))).table
    for j, wd in enumerate(widths):
        t.columns[j].width = Inches(wd)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = str(v) or " "
            c.fill.solid()
            c.fill.fore_color.rgb = RGBColor.from_string(RED if i == 0 else ("FBF3E2" if i == hi else (PALE if i % 2 else "FFFFFF")))
            f = c.text_frame.paragraphs[0].runs[0].font
            f.size, f.name, f.bold = Pt(size), "Calibri", i == 0 or i == hi
            f.color.rgb = RGBColor.from_string("FFFFFF" if i == 0 else INK)


def note(s, x, y, w, h, head, body, col=GREEN, fill="E5F2EE"):
    rect(s, x, y, w, h, fill, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x + 0.15, y + 0.08, w - 0.3, 0.3, head, 13, col, True)
    text(s, x + 0.15, y + 0.42, w - 0.3, h - 0.5, body, 12, INK)


# 1 title
s = slide(None, 1)
text(s, 1.12, 1.7, 6, 0.3, "PROGRESS UPDATE", 13, RED, True)
text(s, 1.12, 2.15, 6.9, 1.6, "What is new since\nour 3 November meeting", 34, INK, True)
text(s, 1.16, 3.9, 6.8, 0.3, "Meeting  ·  17 November 2026", 14, GREY)
text(s, 1.16, 4.75, 6.5, 0.3, "Srinivasan Dhakshanamoorthy", 16, INK, True)
text(s, 1.16, 5.15, 7.3, 0.3, "MSc Data Engineering & AI  |  ESILV / DVRC", 13, GREY)
text(s, 1.16, 5.5, 6.8, 0.3, "Supervisor: Prof. Ahmed Azough", 13, GREY)
for i, (n, t) in enumerate([("77", "movement trainings"), ("24", "new xG + threat + pass models"), ("5,180", "what-if runs"), ("4", "video tests (Phase A)")]):
    rect(s, 8.2 + (i % 2) * 2.1, 1.9 + (i // 2) * 1.75, 1.95, 1.55, RED if i == 0 else PALE, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, 8.32 + (i % 2) * 2.1, 2.0 + (i // 2) * 1.75, 1.75, 0.7, n, 30, "FFFFFF" if i == 0 else RED, True)
    text(s, 8.32 + (i % 2) * 2.1, 2.75 + (i // 2) * 1.75, 1.75, 0.6, t, 12, "FFFFFF" if i == 0 else INK)

# 2 models
s = slide("1. Better models: a fair contest decides which model wins", 2)
table(s, 1.12, 1.6, 6.3, [["Model", "Count", "Result"],
                          ["Movement (graph network)", "77", "1.13 m error"],
                          ["3-model ensemble", "1", "1.10 m, best in 7 of 7"],
                          ["More training data", "7", "1.20 m (worse)"],
                          ["Pass choice", "6", "beats simple baseline"],
                          ["Expected goals (xG)", "12", "new"],
                          ["Threat value", "6", "new"]], [2.8, 0.9, 2.6], 12, hi=2)
pic(s, O("tuning/ladder_chart.png"), 7.7, 1.6, w=4.65)
note(s, 1.12, 4.45, 6.3, 1.9, "How we choose (simple rule)",
     "Each model is tested on a match it never saw, 7 times. A new model wins only if it is better on average, "
     "the 95% interval excludes zero, and it wins at least 6 of 7 matches. The 3-model ensemble won all 7.")

# 3 what-if
s = slide("2. What-if at scale: 5,180 simulated runs, 24 tactical changes", 3)
pic(s, O("whatif/batch_chart.png"), 1.12, 1.55, h=4.8)
table(s, 7.9, 1.6, 4.45, [["Change (defence)", "Threat"], ["Close the ball carrier", "-0.038"], ["Whole team 3 m closer", "-0.033"],
                          ["3 nearest 3 m closer", "-0.033"], ["Keeper off his line", "-0.015"], ["Back four 3 m higher", "+0.024"]],
      [3.2, 1.25], 12, hi=1)
note(s, 7.9, 3.95, 4.45, 2.4, "What it means",
     "24 changes x 20 real moments x 10 repeats. Pressing the ball carrier helps most. "
     "Pushing the back four higher backfires in 14 of 20 moments: it opens space behind.", RED, "FBE8EE")

# 4 VLM idea + noise test
s = slide("3. Professor's idea: positions from TV video. How accurate must they be?", 4)
pic(s, O("noise_test/noise_chart.png"), 1.12, 1.55, h=4.8)
table(s, 7.9, 1.6, 4.45, [["Positions from", "Model error"], ["True tracking", "1.14 m"], ["Good vision system", "1.49 m"],
                          ["Detector + pitch mapping", "1.98 m"], ["Vision-language model only", "2.80 m"]], [3.0, 1.45], 12)
note(s, 7.9, 3.6, 4.45, 2.75, "Noise test: what it shows",
     "We damaged true tracking on purpose. Position jitter hurts most (0.5 m jitter: 1.14 -> 1.44 m); "
     "missing players or ball hurt little. Target: video positions within 0.5 m. "
     "So: detector + tracker as the engine, VLM as a helper.", GOLD, "FBF3E2")

# 5 phase A
s = slide("4. Phase A done: from TV video to player positions (free, open-source)", 5)
det = O("vision_poc/rfdetr/SNGS-117_000301.jpg")
pic(s, det, 1.12, 1.55, w=5.4, cap="RF-DETR on a SoccerNet broadcast frame")
table(s, 6.85, 1.55, 5.5, [["Step", "Result"],
                           ["A1 Detector (490 frames)", "91% people found, 97% team, 0.5 s/frame"],
                           ["A1 Ball", "50% found: needs fine-tuning"],
                           ["A2 Our pitch mapping", "median 0.51 m, 74% of frames"],
                           ["A3 Tracker (3 full clips)", "85-92% followed; IDs switch too often"],
                           ["A4 VLM jersey numbers", "Qwen 28% vs OCR 15%"],
                           ["A4 VLM game situation", "1 of 15 from one frame"]], [2.4, 3.1], 11)
note(s, 1.12, 5.0, 11.23, 1.35, "In one line",
     "A free detector already finds players and teams well; the ball, the pitch mapping and identity switches are the next fixes. "
     "The VLM is slow and weak on its own: useful only for small jobs a detector cannot do.")

# 6 summary
s = slide("5. 3 November vs today, and what comes next", 6)
table(s, 1.12, 1.6, 6.6, [["", "3 Nov", "Today"], ["Movement error", "1.14 m", "1.10 m (ensemble)"], ["Pass models", "3", "6"],
                          ["xG / threat models", "0", "12 / 6"], ["What-if", "demo clips", "5,180 runs"],
                          ["Video -> positions", "not started", "Phase A done"], ["Presentation", "Nov deck", "video + GIF + deck"]],
      [2.3, 1.7, 2.6], 12)
note(s, 8.0, 1.6, 4.35, 4.75, "Next (Phase B)",
     "1. Fine-tune the ball detector on a free GPU (Kaggle)\n2. Stronger pitch mapping (reject bad frames, use the centre circle)\n"
     "3. Fewer identity switches (re-identification)\n4. Feed video positions into the graph models\n"
     "5. Run what-if on matches seen only on TV", RED, "FBE8EE")

# 7 video
s = slide("6. Model v model in the game: Argentina model v France model", 7)
clip = os.path.join(ROOT, "deck_work", "update_clip.mp4")
poster = os.path.join(ROOT, "deck_work", "update_clip.png")
src = O("team_league/Argentina_vs_France_h264.mp4")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "62", "-i", src, "-t", "8", "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", clip], check=True)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", clip, "-frames:v", "1", poster], check=True)
s.shapes.add_movie(clip, Inches(1.12), Inches(1.55), Inches(8.6), Inches(4.84), poster_frame_image=poster, mime_type="video/mp4")
note(s, 9.95, 1.55, 2.4, 4.84, "What you see",
     "Each team model learned from its real World Cup matches and plays its own style in GRF. "
     "Differences come mostly from shooting distance (Croatia shoots from far, Argentina closer).", GREEN)

pr.save(OUT)
print("wrote", OUT)
