"""Narrated project video (about 10 minutes) for the professor and management.

Visuals: the pitch deck and the November demo deck (exported to PNG by PowerPoint), four new cards drawn here, and
the GRF videos already in outputs/. Voice: Microsoft neural voice through edge-tts (needs internet).
Output: presentations/Dynamic_Tactical_AI_Project_Video.mp4

Usage :  python deck_work\\video\\build_video.py            (system Python with Pillow and edge-tts; ffmpeg on PATH)
"""
import asyncio
import os
import subprocess
import time

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORK = os.path.join(HERE, "build")
OUT = os.path.join(ROOT, "presentations", "Dynamic_Tactical_AI_Project_Video.mp4")
VOICE, RATE = "en-US-AndrewNeural", "-4%"
W, H, FPS = 1920, 1080, 25
RED, INK, GREY, PALE, GREEN = (207, 23, 72), (25, 25, 30), (110, 110, 118), (244, 245, 247), (30, 140, 90)
F = "C:/Windows/Fonts/"

P = lambda i: os.path.join(HERE, "pitch", f"p{i}.png")
D = lambda i: os.path.join(HERE, "demo", f"d{i}.png")
O = lambda p: os.path.join(ROOT, "outputs", p)
DEMO = lambda p: os.path.join(ROOT, "deck_work", "demo", p)

# each scene: narration, and the visuals shown one after another (share of the scene's time)
# a visual is ("img", path) or ("clip", path, start_s, caption)
SCENES = [
    ("Hello, and welcome. This is Dynamic Tactical AI, a research project at the De Vinci Research Center and ESILV, by "
     "Srinivasan Dhakshanamoorthy, supervised by Professor Ahmed Azough. In the next ten minutes, we will walk through why we "
     "started, what we built, what the results show, and what we propose next. The idea is simple: read football as a graph, "
     "find the moments that matter, and let a coach ask: what if we had played it differently?",
     [(1, ("img", P(1)))]),
    ("Let's start with the problem. Football tactics are about relationships. A press, or a passing option, is created by "
     "several players moving together, not by one player's statistics. Most tactical AI today stops at set pieces, like corner "
     "kicks, because they are short and well defined. Open play changes every few frames, and is much harder. Data also comes "
     "from many sources that disagree on format and quality. And coaches need to trust the output, so every prediction must "
     "be compared with a simple baseline, and shown with its uncertainty.",
     [(1, ("img", P(2)))]),
    ("We built on solid research. DeepMind's TacticAI, developed with Liverpool, showed that graph models can analyse corner "
     "kicks, and suggest alternatives that expert analysts found useful. Other work treats players as nodes in a graph that "
     "changes over time. The gap we address is open play: a reproducible pipeline that finds the key moments, tests graph "
     "models against strong baselines, and reports what-if scenarios with honest uncertainty.",
     [(1, ("img", P(3)))]),
    ("Our proposal is an AI coach assistant. It detects key tactical moments, turns each moment into a player interaction "
     "graph, learns how play develops over time, and then runs what-if scenarios: controlled changes to a real situation, "
     "compared with their uncertainty. The primary objective is to test whether graph structure and time really add value, "
     "and to present the answer honestly to analysts.",
     [(1, ("img", P(4)))]),
    ("Here is the architecture, in five layers. First, the data. Second, key tactical states. Third, the player graph. "
     "Fourth, the tactical models. And fifth, the agents, which play inside a football simulator called Google Research "
     "Football, or G R F. Each layer has a clear input and output, so a data source or a model can be swapped on its own. "
     "Real data, predictions and simulated scenarios are always kept, and labelled, separately.",
     [(1, ("img", P(5)))]),
    ("So what did we actually use? Real tracking data. Seven complete Bundesliga matches from the open D F L dataset, "
     "tracked ten times per second. Sixty-four matches from the 2022 FIFA World Cup, from P F F, with thirty-two national "
     "teams. And twenty A-League matches from SkillCorner, to check that the method works on other sources too. We also "
     "generated simulated matches in G R F, and compared them with the real data. One example: simulated players run more "
     "than twice as fast as real ones. So the simulator is judged against reality, never the other way round.",
     [(1, ("img", "card_data"))]),
    ("Layer two: key tactical states. Most frames in a match carry very little tactical change. So we mark a new key state "
     "only when the game has really moved: when the players, their positions relative to each other, and the ball, together "
     "shift by at least three metres. This gives the models a focused set of moments, and makes their errors easier to understand.",
     [(1, ("img", P(7)))]),
    ("Layer three: every key state becomes a graph. Each player is a node, with position, speed, team and role. Edges "
     "connect neighbouring teammates and opponents, and the passing lanes from the ball carrier. A sequence of these graphs "
     "over time feeds the learning models.",
     [(1, ("img", P(8)))]),
    ("Layer four: the models, and the first results. Our graph neural network predicts where every player will be next. On "
     "real matches it has never seen, its error is one point one four metres, against one point five one metres for the "
     "simple guess, keep the same speed. That is about twenty-five percent better. For passes, the model ranks the real "
     "receiver first forty-five percent of the time. Random guessing gets ten percent, and simply picking the nearest "
     "teammate gets twenty-seven. And it predicts whether a pass will arrive with an A U C of zero point eight four, against "
     "zero point six three when using the pass length alone.",
     [(1, ("img", "card_models"))]),
    ("Next, we wanted to see a real match inside the simulator. We start G R F from the real positions of all twenty-two "
     "players, steer each one along his real path, and press pass or shoot at the moment of the real kick. After our fixes, "
     "sudden jumps in the video dropped from fifty-three to eight, and G R F scores the real goal, in the same minute. "
     "Here is Fortuna Düsseldorf against Jahn Regensburg, replayed in G R F. The players stay within about one metre of the "
     "real game.",
     [(0.5, ("img", D(2))), (0.5, ("clip", O("pass2_3min_GRF_h264.mp4"), 25, "Real match replayed in GRF  ·  Fortuna Düsseldorf v Jahn Regensburg"))]),
    ("It was not easy. G R F only accepts button presses, so we cannot simply place a player on a spot. The ball leaves the "
     "foot half a second late. And small differences add up, so after a few minutes the game drifts twenty metres away from "
     "reality. Our answer was to restart G R F from the real data every few seconds, hide those restarts, and keep the "
     "simulator for what it does best: playing out alternatives.",
     [(1, ("img", D(3)))]),
    ("To watch a real match exactly as it was recorded, we built a three-dimensional viewer in the browser, with Three J S. "
     "It shows ninety-four real matches from three datasets, with five camera angles. And it can overlay the model's runs as "
     "ghost players, next to the real ones, so you see reality and the model side by side.",
     [(0.45, ("img", D(4))), (0.55, ("img", "card_viewer"))]),
    ("Now the team models: parts A and B. Using the World Cup data, we built a model for Argentina, France and Croatia. A "
     "movement model, that predicts where the players go, and a decision model, that chooses who receives the ball, when to "
     "shoot, and how long to keep it. Then we let these team brains control all eleven players in G R F, and organised a "
     "full round robin: eighty-eight complete matches, of about five minutes of play each. Here is Argentina's model against "
     "France's model, with the real players' names.",
     [(0.55, ("img", D(5))), (0.45, ("clip", O("team_league/Argentina_vs_France_h264.mp4"), 62, "Model v model in GRF  ·  Argentina model v France model (real names)"))]),
    ("What did we learn? Movement looks almost the same for every team: telling the model which team it is barely changes "
     "its accuracy. But decisions differ clearly. Croatia shoots from thirty-seven metres, Argentina from twenty-seven. And "
     "that habit decides the league: Croatia's model won eighteen of thirty-six matches, Argentina's only nine. To be honest, "
     "against G R F's own built-in A I, all our models still lose. They learned real football, not yet how to win this game.",
     [(1, ("img", D(6)))]),
    ("Parts C and D are the heart of the coach assistant: the what-if. We take a real moment, two point six seconds before a "
     "real goal. We change one thing: the defence higher or deeper, a quicker forward pass, or the attack playing in another "
     "team's style. Then we play it twenty times in G R F, with the same random seeds as the unchanged version, and count "
     "the goals, shots and chances. On the left, no change. On the right, the quick forward pass.",
     [(0.55, ("img", D(7))), (0.45, ("clip", O("whatif/J03WOH_3500_3610_early_pass_grf_h264.mp4"), 0, "What-if in GRF  ·  left: no change   ·   right: quick forward pass"))]),
    ("The result. With no change, the attack scored in five of twenty runs: twenty-five percent. With a quick forward pass, it "
     "scored in thirteen of twenty: sixty-five percent. That is plus zero point four goals per run, and even the low end of "
     "the confidence interval is above zero, so it is real, not luck. Moving the back four by three metres, or swapping the "
     "team's style, stayed within noise. This is one moment, and our own models, so it shows that the method works. It is not "
     "yet a rule for real football.",
     [(1, ("img", D(8)))]),
    ("A small change that matters for coaches and players: real names. G R F normally shows invented names, like Curie or "
     "Einstein. We now write the real names into the simulator: Messi, Mbappé, Di María, or the real Bundesliga players. And "
     "the same names appear in the 3D viewer. A player can recognise himself on the screen.",
     [(1, ("img", "card_names"))]),
    ("Every number in this project follows one rule: it is shown next to a simple baseline, such as a random choice, keep "
     "running, or no change, so anyone can judge how good it really is. Data quality is measured too. In the World Cup data, "
     "only fifty-seven percent of player positions are actually seen by the camera; the rest are estimated. We report that openly.",
     [(1, ("img", D(9)))]),
    ("These are our six commitments for trustworthy tactical A I. Baseline first. No leakage between training and test "
     "matches. Measured robustness. Honest what-ifs, always labelled as model output. The analyst stays in the loop: the "
     "system informs, it never directs players. And everything is reproducible, including the negative results.",
     [(1, ("img", P(14)))]),
    ("So, what do we propose next? First, a coach what-if: stop a past match at any second, move one chosen player to the "
     "position the coach advised, and replay it twenty times, to show the player what could have happened. Second, a model "
     "tuning ladder, with more data and ensembles, to show measurable gains in accuracy. Third, run the what-if on many "
     "moments, not just one, to turn single findings into general patterns. And fourth, partner with a club, for their data "
     "and their analysts' feedback, which is the real test of value.",
     [(1, ("img", "card_next"))]),
    ("To sum up. We can replay real matches, build team models that play full games, and test tactical changes with honest "
     "numbers. The foundation is in place. Thank you for watching. We welcome your questions, and a club or data partner, to "
     "take the next step with us.",
     [(1, ("img", P(15)))]),
]


def font(size, bold=False):
    return ImageFont.truetype(F + ("arialbd.ttf" if bold else "arial.ttf"), size)


def base(title, kicker="DYNAMIC TACTICAL AI IN FOOTBALL"):
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 14, H], fill=RED)
    d.rectangle([66, 0, 70, H], fill=RED)
    d.text((158, 42), kicker, font=font(22, True), fill=GREY)
    d.text((158, 105), title, font=font(52, True), fill=INK)
    d.line([158, 190, 1778, 190], fill=(220, 220, 225), width=2)
    try:
        logo = Image.open(DEMO("esilv_logo.png")).convert("RGBA")
        logo.thumbnail((170, 90))
        im.paste(logo, (1660, 970), logo)
    except OSError:
        pass
    return im, d


def wrap(d, text, xy, width, f, fill, gap=8):
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


def box(d, x, y, w, h, fill=PALE, outline=(225, 225, 230)):
    d.rounded_rectangle([x, y, x + w, y + h], 14, fill=fill, outline=outline, width=2)


def card_data():
    im, d = base("The data we actually used: real matches first")
    items = [("7", "Bundesliga matches", "DFL open data. Optical tracking, 10 times a second."),
             ("64", "World Cup 2022", "PFF matches. 32 national teams; Argentina, France, Croatia 7 each."),
             ("20", "A-League matches", "SkillCorner. Broadcast tracking: a second, different source."),
             ("12", "Simulated GRF matches", "Compared against real play: players run 2.3x faster.")]
    for k, (n, t, s) in enumerate(items):
        x = 158 + k * 410
        box(d, x, 250, 380, 470, fill=RED if k == 0 else PALE, outline=RED if k == 0 else (225, 225, 230))
        c = "white" if k == 0 else INK
        d.text((x + 30, 280), n, font=font(120, True), fill=c)
        d.text((x + 30, 440), t, font=font(32, True), fill=c)
        wrap(d, s, (x + 30, 500), 320, font(26), "white" if k == 0 else GREY)
    box(d, 158, 770, 1620, 120, fill=(253, 246, 232), outline=(240, 220, 180))
    wrap(d, "Real data is the reference. The simulator is judged against reality, never the other way round. "
            "In the World Cup data only 57% of player positions are seen by the camera, and we say so.",
         (190, 795), 1560, font(28), INK)
    return im


def card_models():
    im, d = base("First results: every number next to a simple baseline")
    d.text((780, 225), "OUR MODEL", font=font(22, True), fill=RED)
    d.text((1120, 225), "SIMPLE BASELINE", font=font(22, True), fill=GREY)
    rows = [("Where will each player be next?", "1.14 m", "1.51 m", "keep the same speed", "25% closer to reality"),
            ("Who receives the pass?", "45%", "10% / 27%", "random / nearest teammate", "real receiver ranked first"),
            ("Will the pass arrive?", "AUC 0.84", "AUC 0.63", "pass length only", "1.0 perfect, 0.5 coin toss"),
            ("Do team labels change movement?", "barely", "0.004 m", "same model, team label off", "style lives in decisions")]
    for k, (q, ours, b, bl, say) in enumerate(rows):
        y = 270 + k * 165
        box(d, 158, y, 1620, 145)
        wrap(d, q, (190, y + 30), 540, font(32, True), INK)
        d.text((780, y + 30), ours, font=font(56, True), fill=RED)
        d.text((1120, y + 30), b, font=font(44, True), fill=GREY)
        d.text((1120, y + 92), bl, font=font(22), fill=GREY)
        wrap(d, say, (1480, y + 40), 280, font(24), GREEN)
    d.text((158, 945), "Tested on matches the model never saw (leave one match out).", font=font(26), fill=GREY)
    return im


def card_viewer():
    im, d = base("Three.js viewer: real players solid, model runs as ghosts")
    for k, (p, cap) in enumerate([("viewer_real.png", "Real match, as recorded"), ("viewer_whatif.png", "Real match + model runs (ghosts)")]):
        pic = Image.open(DEMO(p)).convert("RGB").resize((790, 444))
        x = 158 + k * 830
        im.paste(pic, (x, 250))
        d.text((x, 715), cap, font=font(30, True), fill=INK)
    wrap(d, "94 real matches from three datasets  ·  five cameras  ·  player names  ·  each model run can be switched on and off",
         (158, 800), 1600, font(28), GREY)
    return im


def card_names():
    im, d = base("Real player names in GRF and in the 3D viewer")
    a = Image.open(DEMO("league_names.png")).convert("RGB").resize((960, 540))
    im.paste(a, (158, 240))
    b = Image.open(DEMO("whatif_names.png")).convert("RGB")
    b = b.resize((620, int(620 * b.height / b.width)))
    im.paste(b, (1158, 240))
    y = 240 + b.height + 40
    for t, s in [("Before", "Curie, Einstein, Turing: GRF's invented names"),
                 ("Now", "Messi, Mbappé, Di María; the real Bundesliga players"),
                 ("Where", "GRF videos, model matches, what-if, 3D viewer")]:
        d.text((1158, y), t, font=font(28, True), fill=RED)
        y = wrap(d, s, (1290, y), 490, font(26), INK) + 18
    d.text((158, 800), "Argentina model v France model in GRF, with the World Cup starting eleven", font=font(26), fill=GREY)
    return im


def card_next():
    im, d = base("What we propose next")
    steps = [("1", "Coach what-if", "Stop a past match, move one player where the coach advised, replay it 20 times, show the player the odds."),
             ("2", "Tuning ladder", "More data, mirrored pitches, ensembles: show how accuracy improves, step by step."),
             ("3", "Many moments", "Run what-if on hundreds of moments, so single findings become general patterns."),
             ("4", "Club partner", "Club data and analyst feedback: the real test of value for coaches.")]
    for k, (n, t, s) in enumerate(steps):
        x = 158 + k * 410
        hl = k == 0
        box(d, x, 250, 380, 520, fill=RED if hl else PALE, outline=RED if hl else (225, 225, 230))
        c = "white" if hl else INK
        d.text((x + 30, 275), n, font=font(90, True), fill=c)
        d.text((x + 30, 400), t, font=font(34, True), fill=c)
        wrap(d, s, (x + 30, 465), 320, font(27), "white" if hl else GREY)
    box(d, 158, 820, 1620, 100, fill=(253, 246, 232), outline=(240, 220, 180))
    d.text((190, 850), "Already working today: stop at any second, shift the back line, replay 20 times, compare with a 95% interval.",
           font=font(27), fill=INK)
    return im


def clip_frame(caption):
    im = Image.new("RGB", (W, H), (14, 16, 20))
    d = ImageDraw.Draw(im)
    d.rectangle([0, H - 110, W, H], fill=(24, 26, 32))
    d.rectangle([0, H - 110, 14, H], fill=RED)
    d.text((60, H - 78), caption, font=font(36, True), fill="white")
    return im


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


async def speak(text, path):
    import edge_tts
    await edge_tts.Communicate(text, VOICE, rate=RATE).save(path)


def segment(k, text, visuals):
    mp3 = os.path.join(WORK, f"s{k:02d}.mp3")
    for attempt in range(10):                                      # the voice service sometimes sends nothing; wait, try again
        if os.path.exists(mp3) and os.path.getsize(mp3) > 1000:
            break
        try:
            asyncio.run(speak(text, mp3))
        except Exception as e:
            print(f"scene {k}: voice attempt {attempt + 1} failed ({type(e).__name__})", flush=True)
            if os.path.exists(mp3):
                os.remove(mp3)
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"no voice for scene {k}")
    out = os.path.join(WORK, f"seg{k:02d}.mp4")
    if os.path.exists(out):                                        # built earlier; delete segNN.mp4 to rebuild a scene
        return out, duration(out)
    total = duration(mp3) + 1.0                                    # half a second of quiet before and after
    parts = []
    for j, (share, v) in enumerate(visuals):
        dur = round(total * share, 2)
        part = os.path.join(WORK, f"s{k:02d}_{j}.mp4")
        if v[0] == "img":
            src = v[1] if os.path.exists(v[1]) else os.path.join(WORK, v[1] + ".png")
            run(["ffmpeg", "-y", "-loop", "1", "-t", str(dur), "-i", src,
                 "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:white,"
                        f"zoompan=z='1+0.00012*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={FPS},format=yuv420p",
                 "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "20", part])
        else:
            _, path, start, cap = v
            bg = os.path.join(WORK, f"s{k:02d}_{j}_bg.png")
            clip_frame(cap).save(bg)
            run(["ffmpeg", "-y", "-loop", "1", "-t", str(dur), "-i", bg, "-stream_loop", "-1", "-ss", str(start), "-i", path,
                 "-filter_complex", f"[1:v]fps={FPS},scale=1740:-2[c];[0:v][c]overlay=(W-w)/2:(H-110-h)/2:shortest=1,format=yuv420p",
                 "-t", str(dur), "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "20", part])
        parts.append(part)
    lst = os.path.join(WORK, f"s{k:02d}_v.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    out = os.path.join(WORK, f"seg{k:02d}.mp4")
    vdur = sum(duration(p) for p in parts)
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", mp3,
         "-filter_complex", f"[0:v]fade=t=in:st=0:d=0.4,fade=t=out:st={vdur - 0.4:.2f}:d=0.4[v];"
                            f"[1:a]adelay=500|500,apad,atrim=0:{vdur:.2f},aresample=48000[a]",
         "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "160k", "-ac", "2", out])
    print(f"scene {k:2d}: {vdur:5.1f} s", flush=True)
    return out, vdur


def main():
    os.makedirs(WORK, exist_ok=True)
    for name in ("card_data", "card_models", "card_viewer", "card_names", "card_next"):
        globals()[name]().save(os.path.join(WORK, name + ".png"))
    segs, total = [], 0.0
    for k, (text, visuals) in enumerate(SCENES, 1):
        s, dur = segment(k, text, visuals)
        segs.append(s)
        total += dur
    lst = os.path.join(WORK, "all.txt")
    open(lst, "w").write("".join(f"file '{s}'\n" for s in segs))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", OUT])
    print(f"wrote {OUT}  ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
