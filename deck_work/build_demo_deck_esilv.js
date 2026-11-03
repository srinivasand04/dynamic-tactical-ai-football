// 11-slide demo deck in the style of Dynamic_Tactical_AI_ESILV_Progress_Oct_2026.pptx (ESILV template: red side bar, small and
// large ESILV logo, grey header, 24 pt title, light cards, red flow arrows). Real match in GRF, its challenges, the Three.js viewer,
// Parts A-D as flowcharts with results, the metrics explained, every demo command, and next steps.
const pptxgen = require(process.env.PPTXGENJS || "pptxgenjs");
const path = require("path");
const D = __dirname.replace(/\\/g, "/");
const I = D + "/demo/";
const OUT = path.join(D, "..", "presentations", "Dynamic_Tactical_AI_Demo_Nov_2026_v2.pptx");

// colours of the ESILV progress deck
const C = { red: "CF1748", text: "22272B", gray: "5C6268", line: "D9DDE0", card: "F5F6F7", green: "258068", greenT: "E5F2EE",
  blue: "397AA0", blueT: "E9F1F6", amber: "886119", amberT: "FBF3E2", light: "AEB8BE", white: "FFFFFF", redT: "FBE8ED" };
const F = "Arial", MONO = "Courier New";
const N = 11;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Dynamic Tactical AI: real matches, team models and what-if in GRF";
pres.author = "Srinivasan Dhakshanamoorthy";
pres.company = "ESILV / DVRC";

function txt(s, text, x, y, w, h, o = {}) {
  s.addText(text, { x, y, w, h, fontFace: o.font || F, fontSize: o.size || 13, bold: !!o.bold, color: o.color || C.text, align: o.align || "left",
    valign: o.valign || "top", margin: 0, isTextBox: true, fit: "none", italic: !!o.italic, paraSpaceAfter: o.after || 0, objectName: o.name });
}
function card(s, x, y, w, h, fill = C.card, border = C.line) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, rectRadius: 0.08, line: border ? { color: border, width: 0.75 } : { color: fill, width: 0 } });
}
function flowArrow(s, x, y) {                     // red line + triangle, as in the template's project flow
  s.addShape(pres.shapes.LINE, { x, y, w: 0.2, h: 0, line: { color: C.red, width: 1.5 } });
  s.addShape(pres.shapes.ISOSCELES_TRIANGLE, { x: x + 0.16, y: y - 0.075, w: 0.15, h: 0.15, rotate: 90, fill: { color: C.red }, line: { color: C.red, width: 0 } });
}
function downArrow(s, x, y) {
  s.addShape(pres.shapes.LINE, { x, y, w: 0, h: 0.16, line: { color: C.red, width: 1.5 } });
  s.addShape(pres.shapes.ISOSCELES_TRIANGLE, { x: x - 0.075, y: y + 0.13, w: 0.15, h: 0.13, rotate: 180, fill: { color: C.red }, line: { color: C.red, width: 0 } });
}
function frame(s, n, title) {
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.09, h: 7.5, fill: { color: C.red }, line: { color: C.red, width: 0 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.46, y: 0, w: 0.02, h: 7.5, fill: { color: C.red }, line: { color: C.red, width: 0 } });
  s.addImage({ path: I + "esilv_logo.png", x: 0.11, y: 0.27, w: 0.33, h: 0.13, altText: "ESILV" });
  txt(s, "DATA\nGRAPH\nAI", 0.1, 1.17, 0.32, 0.56, { size: 6, bold: true, color: C.gray, align: "center" });
  txt(s, String(n), 0.1, 6.93, 0.34, 0.21, { size: 9.75, bold: true, align: "center", name: "Page" });
  txt(s, "DYNAMIC TACTICAL AI IN FOOTBALL", 1.1, 0.29, 10.94, 0.21, { size: 9.75, bold: true, color: C.gray, name: "Header" });
  txt(s, title, 1.1, 0.66, 11.25, 0.58, { size: 24, bold: true, valign: "middle", name: "Title" });
  s.addShape(pres.shapes.LINE, { x: 1.1, y: 1.32, w: 11.25, h: 0, line: { color: C.line, width: 0.75 } });
  s.addImage({ path: I + "esilv_logo.png", x: 11.27, y: 6.79, w: 1.46, h: 0.58, altText: "ESILV logo" });
}
function label(s, t, x, y, w, color = C.red) { txt(s, t, x, y, w, 0.28, { size: 13, bold: true, color }); }
function chip(s, t, x, y, w, fill, color) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.34, fill: { color: fill }, rectRadius: 0.06, line: { color: fill, width: 0 } });
  txt(s, t, x + 0.12, y, w - 0.24, 0.34, { size: 11, bold: true, color, valign: "middle" });
}
// flow box in the template style: number in red, caps title, grey line of text
function flowBox(s, n, x, y, w, h, title, body, hi = false) {
  card(s, x, y, w, h, hi ? C.red : C.card, hi ? C.red : C.line);
  txt(s, String(n), x + 0.15, y + 0.12, 0.4, 0.25, { size: 11.25, bold: true, color: hi ? C.white : C.red });
  txt(s, title, x + 0.15, y + 0.42, w - 0.3, 0.42, { size: 12.75, bold: true, color: hi ? C.white : C.text });
  txt(s, body, x + 0.15, y + 0.88, w - 0.3, h - 0.95, { size: 11, color: hi ? C.white : C.gray });
}
function img(s, file, x, y, w, h, alt) {
  s.addImage({ path: I + file, x, y, w, h, altText: alt });
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { type: "none" }, line: { color: C.line, width: 0.75 } });
}
// a small table drawn as rows (header row in grey caps)
function rows(s, x, y, widths, data, o = {}) {
  const rh = o.rh || 0.36;
  data.forEach((r, i) => {
    let cx = x;
    if (i > 0 && i % 2 === 1) s.addShape(pres.shapes.RECTANGLE, { x, y: y + i * rh, w: widths.reduce((a, b) => a + b, 0), h: rh, fill: { color: C.card }, line: { color: C.card, width: 0 } });
    r.forEach((v, j) => {
      const hl = o.hl && o.hl(i, j);
      txt(s, String(v), cx + 0.08, y + i * rh, widths[j] - 0.12, rh, { size: i === 0 ? 9.5 : (o.size || 12), bold: i === 0 || j === 0 || hl,
        color: i === 0 ? C.gray : hl ? C.red : C.text, valign: "middle", align: j === 0 ? "left" : (o.align || "center") });
      cx += widths[j];
    });
  });
}

// ---------------------------------------------------------------- 1 title
{
  const s = pres.addSlide();
  frame(s, 1, "From real tracking data to football what-if tests");
  label(s, "DEMO", 1.12, 1.71, 5.1);
  txt(s, "Can our AI replay a real match,\nplay it, and test a different\ndecision?", 1.12, 2.16, 6.9, 1.6, { size: 30, bold: true });
  txt(s, "Demo  •  4 November 2026", 1.16, 3.95, 6.77, 0.31, { size: 15, color: C.gray });
  txt(s, "Srinivasan Dhakshanamoorthy", 1.16, 4.75, 6.46, 0.32, { size: 15, bold: true });
  txt(s, "MSc Data Engineering & AI  |  ESILV / DVRC", 1.16, 5.16, 7.29, 0.29, { size: 13.5, color: C.gray });
  txt(s, "Supervisor: Prof. Ahmed Azough", 1.16, 5.51, 6.77, 0.28, { size: 12, color: C.gray });
  img(s, "grf_pass_frame.png", 8.2, 2.4, 3.9, 2.19, "A real Bundesliga 2 match replayed inside Google Research Football");
  txt(s, "REAL MATCH REPLAYED IN GRF  •  58:03", 8.2, 4.75, 3.9, 0.3, { size: 10.5, bold: true, color: C.gray, align: "center" });
  s.addNotes("Today: four things. One, a real match replayed inside the football game GRF, and why that is hard. Two, a 3D viewer that shows the real match exactly as recorded. Three, models of Argentina, France and Croatia that play full matches against each other. Four, what-if tests: take a real moment, change one decision, and measure what happens.");
}

// ---------------------------------------------------------------- 2 real match in GRF
{
  const s = pres.addSlide();
  frame(s, 2, "Real match in GRF: we steer all 22 players along the real game");
  const st = [["REAL DATA", "Every player and the ball, 10 times a second"], ["START GRF", "From the real positions of all 22 players"],
    ["STEER", "Each player follows his real path"], ["KICK", "Pass or shoot at the moment of the real kick"], ["VIDEO", "GRF draws it, with the real match clock"]];
  st.forEach(([t, b], i) => {
    const x = 1.12 + i * 2.27;
    flowBox(s, i + 1, x, 1.6, 1.95, 1.62, t, b, i === 4);
    if (i < 4) flowArrow(s, x + 1.98, 2.41);
  });
  img(s, "grf_pass_frame.png", 1.12, 3.55, 5.0, 2.81, "GRF replay of the real match with the real match clock");
  txt(s, "Fortuna Düsseldorf v Jahn Regensburg, 58th minute, played by GRF", 1.12, 6.42, 5.0, 0.3, { size: 10.5, color: C.gray });
  label(s, "WHAT CHANGED AFTER OUR FIXES (3-MINUTE REPLAY)", 6.45, 3.55, 5.9);
  const ch = [["Jumps in the video", "53", "8", "fewer sudden cuts"], ["Restarts you can notice", "100%", "28%", "72% now invisible"],
    ["Ball leaves the foot", "0.5 s late", "on time", "button pressed early"], ["GRF scores the real goal", "no", "yes", "same goal, same minute"]];
  ch.forEach(([what, before, after, note], i) => {
    const y = 3.98 + i * 0.66;
    card(s, 6.45, y, 5.9, 0.56);
    txt(s, what, 6.6, y, 2.35, 0.56, { size: 12.5, bold: true, valign: "middle" });
    txt(s, before, 8.9, y, 0.95, 0.56, { size: 14, bold: true, color: C.gray, valign: "middle", align: "center" });
    txt(s, "→", 9.82, y, 0.3, 0.56, { size: 14, bold: true, color: C.red, valign: "middle", align: "center" });
    txt(s, after, 10.1, y, 0.95, 0.56, { size: 14, bold: true, color: C.green, valign: "middle", align: "center" });
    txt(s, note, 11.05, y, 1.25, 0.56, { size: 9.5, color: C.gray, valign: "middle" });
  });
  txt(s, "BEFORE", 8.9, 6.6, 0.95, 0.2, { size: 8.5, bold: true, color: C.gray, align: "center" });
  txt(s, "AFTER", 10.1, 6.6, 0.95, 0.2, { size: 8.5, bold: true, color: C.green, align: "center" });
  s.addNotes("GRF is a game: you can only place players at the start, after that you can only press buttons. So we start GRF from the real positions and steer every player along his real path; when the real player kicks, our player presses pass or shot at that moment. The right side shows what our fixes changed in a 3-minute replay: sudden jumps in the video went from 53 to 8; most restarts can no longer be noticed; the ball used to leave the foot half a second late, now we press the button early so it leaves on time; and GRF now scores the real goal, in the same minute.");
}

// ---------------------------------------------------------------- 3 challenges
{
  const s = pres.addSlide();
  frame(s, 3, "Why a real match is hard to play in GRF, and how we handled it");
  txt(s, "CHALLENGE", 1.12, 1.6, 3.5, 0.25, { size: 10.5, bold: true, color: C.gray });
  txt(s, "WHAT WENT WRONG", 4.75, 1.6, 3.6, 0.25, { size: 10.5, bold: true, color: C.gray });
  txt(s, "OUR FIX", 8.6, 1.6, 3.6, 0.25, { size: 10.5, bold: true, color: C.red });
  const r = [["GRF only takes button presses", "No way to put a player at a spot once the game runs", "Steer each player towards his real spot every 0.1 s"],
    ["The ball leaves the foot late", "Half a second after the button, so passes came too late", "Press the button 0.7 s before the real kick"],
    ["A long game drifts away", "Small differences add up over the minutes", "Restart GRF from the real data every few seconds"],
    ["Restarts were visible", "Camera jumped, clock reset, players stood still", "Hidden warm-up, real clock banner, restart only when in step"],
    ["Player names are invented", "GRF shows names like 'Curie'; they are not the real players", "Use shirt numbers from the real match instead"]];
  r.forEach(([a, b, c], i) => {
    const y = 1.93 + i * 0.72;
    card(s, 1.12, y, 3.45, 0.6, C.card);
    txt(s, a, 1.27, y, 3.2, 0.6, { size: 12.5, bold: true, valign: "middle" });
    txt(s, b, 4.75, y, 3.6, 0.6, { size: 11.5, color: C.gray, valign: "middle" });
    flowArrow(s, 8.3, y + 0.3);
    card(s, 8.6, y, 3.75, 0.6, C.greenT, null);
    txt(s, c, 8.75, y, 3.5, 0.6, { size: 11.5, bold: true, color: C.green, valign: "middle" });
  });
  card(s, 1.12, 5.62, 11.23, 1.05, C.amberT);
  txt(s, "WHY NOT ONE CONTINUOUS REAL MATCH?", 1.32, 5.72, 6, 0.26, { size: 11, bold: true, color: C.amber });
  txt(s, "GRF's physics is not real physics. Left alone for a few minutes, GRF's ball ends up far from the real ball (20 m on average), so it is no longer the real match. That is why we restart often, and why we built a 3D viewer for the full real match (next slide).", 1.32, 6.0, 10.8, 0.62, { size: 12 });
  s.addNotes("These are the difficulties, each with what went wrong and what we did. The key point for the professor: a fully continuous real match in GRF is not possible, because GRF's physics drifts away from reality within a few minutes; the ball ends up about 20 metres from where the real ball was. So we restart from the real data every few seconds and hide those restarts, and for the full real match we use the 3D viewer.");
}

// ---------------------------------------------------------------- 4 Three.js viewer
{
  const s = pres.addSlide();
  frame(s, 4, "Three.js viewer: the real match exactly as recorded, in 3D");
  img(s, "viewer_real.png", 1.12, 1.6, 7.2, 4.05, "Three.js viewer: real players with shirt numbers and GRF ghost players");
  txt(s, "Solid players = real match.  Players with a ring = a GRF or model run of the same moment, linked to the real player.", 1.12, 5.75, 7.2, 0.5, { size: 11, color: C.gray });
  const pts = [["NO GRF LIMITS", "Shows the data as it was: no drift, no restarts, no invented names."],
    ["COMPARE RUNS", "GRF replays and model runs appear as ghost players next to the real ones."],
    ["FIVE CAMERAS", "Broadcast, top, behind the goal, follow the ball, free camera."],
    ["ANY MATCH", "94 real matches from three datasets, in any browser."]];
  pts.forEach(([t, b], i) => {
    const y = 1.6 + i * 1.03;
    card(s, 8.6, y, 3.75, 0.9);
    txt(s, t, 8.78, y + 0.1, 3.4, 0.25, { size: 11.5, bold: true, color: C.red });
    txt(s, b, 8.78, y + 0.38, 3.45, 0.5, { size: 11, color: C.text });
  });
  chip(s, "Optional in the demo: commands on slide 10", 8.6, 5.83, 3.75, C.amberT, C.amber);
  s.addNotes("Because GRF cannot show a whole real match, we built a viewer with Three.js. It draws the tracking data as recorded, in 3D, in a normal browser. Model runs and GRF runs of the same moment appear as ghost players next to the real players, so you can see where the model goes differently. Five cameras, a timeline, and it works for all 94 matches. The commands are on slide 10.");
}

// ---------------------------------------------------------------- 5 Parts A + B flow
{
  const s = pres.addSlide();
  frame(s, 5, "Part A and B: build a model for each team, then let them play");
  label(s, "PART A  •  TEAM MODELS", 1.12, 1.55, 5);
  label(s, "PART B  •  LEAGUE IN GRF", 7.0, 1.55, 5);
  const a = [["WORLD CUP DATA", "64 matches. Argentina, France, Croatia: 7 matches each."],
    ["MOVEMENT MODEL", "Guesses where every player will be in 2 seconds. It is also told the team."],
    ["DECISION MODEL", "Who gets the pass, when to shoot, how long to keep the ball."]];
  const b = [["TEAM BRAIN IN GRF", "Each model controls its 11 GRF players, 10 times a second."],
    ["ROUND ROBIN", "3 teams + 1 mixed model. Every pair 12 times, both halves: 88 matches."],
    ["SCORES", "Wins, draws, losses, goals, shots, possession."]];
  a.forEach(([t, x], i) => {
    const y = 1.95 + i * 1.6;
    flowBox(s, i + 1, 1.12, y, 5.4, 1.32, t, x);
    if (i < 2) downArrow(s, 3.82, y + 1.36);
  });
  b.forEach(([t, x], i) => {
    const y = 1.95 + i * 1.6;
    flowBox(s, i + 4, 7.0, y, 5.35, 1.32, t, x, i === 2);
    if (i < 2) downArrow(s, 9.67, y + 1.36);
  });
  flowArrow(s, 6.62, 2.61);
  s.addNotes("Part A: from World Cup tracking data we train a model that guesses where every player will be two seconds later; it is also told which team the player belongs to. A second model learns each team's habits: who gets the pass, when the team shoots, how long it keeps the ball. Part B: these models control eleven GRF players each. Every team plays every team twelve times, plus a mixed model that is not told any team: 88 matches in total, and we count wins, draws, losses, goals and shots.");
}

// ---------------------------------------------------------------- 6 Parts A + B results (redesigned)
{
  const s = pres.addSlide();
  frame(s, 6, "Results A and B: team habits decide the league");
  // panel 1
  card(s, 1.12, 1.55, 3.45, 3.95);
  txt(s, "1", 1.3, 1.68, 0.3, 0.3, { size: 13, bold: true, color: C.red });
  txt(s, "DO TEAMS MOVE DIFFERENTLY?", 1.6, 1.68, 2.9, 0.3, { size: 11, bold: true });
  txt(s, "The test: we gave Argentina's unseen match the label 'France' on purpose, and the same for the others.", 1.3, 2.1, 3.1, 0.95, { size: 12 });
  txt(s, "The result: the model's guesses were just as good with the wrong label.", 1.3, 3.1, 3.1, 0.75, { size: 12 });
  chip(s, "No: movement looks alike", 1.3, 4.0, 3.1, C.amberT, C.amber);
  txt(s, "7 matches per team is too few to learn a team's way of moving.", 1.3, 4.5, 3.1, 0.75, { size: 10.5, color: C.gray });
  // panel 2
  card(s, 4.75, 1.55, 3.6, 3.95);
  txt(s, "2", 4.93, 1.68, 0.3, 0.3, { size: 13, bold: true, color: C.red });
  txt(s, "DO TEAMS DECIDE DIFFERENTLY?", 5.23, 1.68, 3.05, 0.3, { size: 11, bold: true });
  rows(s, 4.88, 2.1, [1.05, 0.8, 0.8, 0.75], [["TEAM", "SHOOTS FROM", "LONG PASSES", "HOLDS BALL"],
    ["Argentina", "27 m", "25%", "1.2 s"], ["France", "29 m", "24%", "1.2 s"], ["Croatia", "37 m", "29%", "1.3 s"]],
    { rh: 0.42, size: 12, hl: (i, j) => i === 3 && j === 1 });
  chip(s, "Yes: Croatia shoots from far", 4.93, 4.0, 3.25, C.greenT, C.green);
  txt(s, "Learned from about 2,300 real passes per team.", 4.93, 4.5, 3.25, 0.75, { size: 10.5, color: C.gray });
  // panel 3
  card(s, 8.53, 1.55, 3.82, 3.95);
  txt(s, "3", 8.71, 1.68, 0.3, 0.3, { size: 13, bold: true, color: C.red });
  txt(s, "WHO WINS THE GRF LEAGUE?", 9.01, 1.68, 3.2, 0.3, { size: 11, bold: true });
  rows(s, 8.66, 2.1, [1.06, 0.62, 0.62, 0.62, 0.66], [["TEAM", "WON", "DRAW", "LOST", "SHOTS"],
    ["Mixed", 21, 8, 7, 23], ["Croatia", 18, 6, 12, 26], ["France", 12, 6, 18, 16], ["Argentina", 9, 4, 23, 9]],
    { rh: 0.42, size: 12, hl: (i, j) => i === 1 && j === 1 });
  txt(s, "36 matches each. Shots = per match.", 8.71, 4.25, 3.5, 0.3, { size: 10.5, color: C.gray });
  chip(s, "More shots, more wins", 8.71, 4.6, 3.45, C.greenT, C.green);
  // takeaway
  card(s, 1.12, 5.68, 11.23, 1.0, C.blueT, null);
  txt(s, "HOW TO SAY IT", 1.32, 5.78, 3, 0.25, { size: 11, bold: true, color: C.blue });
  txt(s, "The models move like real players, but not yet like one particular team. What makes the teams different is their habits, mostly when they shoot, and that decides the league. Against GRF's own built-in AI, all our models still lose.", 1.32, 6.05, 10.9, 0.6, { size: 12 });
  s.addNotes("Three questions. One: do the teams move differently? We gave each team's unseen match the wrong team label on purpose; the guesses stayed just as good, so the model has not learned a team-specific way of moving; seven matches per team is too few. Two: do they decide differently? Yes. Croatia shoots from 37 metres on average, Argentina waits until 27 metres; Croatia also plays more long passes. Three: who wins? The mixed model wins 21 of its 36 matches, Croatia 18, France 12, Argentina 9. The teams that shoot more win more. Honest limit: against GRF's own built-in AI all our models lose, about 8 to 10 goals conceded per match.");
}

// ---------------------------------------------------------------- 7 Parts C + D flow
{
  const s = pres.addSlide();
  frame(s, 7, "Part C and D: change one thing, play it 20 times, compare");
  label(s, "PART C  •  WHAT-IF", 1.12, 1.55, 5);
  const st = [["REAL MOMENT", "58:50, 2.6 s before a real goal"], ["ONE CHANGE", "Defence 3 m deeper or higher, a quick forward pass, or another team's style"],
    ["PLAY IT 20 TIMES", "Team models play on for 12 s in GRF"], ["SAME 20 SEEDS", "Each run has a 'no change' twin with the same luck"], ["COUNT", "Goals, shots and chances, change vs no change"]];
  st.forEach(([t, b], i) => {
    const x = 1.12 + i * 2.27;
    flowBox(s, i + 1, x, 1.95, 1.95, 1.75, t, b, i === 4);
    if (i < 4) flowArrow(s, x + 1.98, 2.82);
  });
  label(s, "PART D  •  SEE IT IN THREE WAYS", 1.12, 3.95, 6);
  img(s, "whatif_grf.png", 1.12, 4.35, 3.6, 1.11, "GRF side by side: no change on the left, what-if on the right");
  img(s, "whatif_drawn.png", 4.92, 4.35, 3.6, 1.38, "Real data next to the what-if, drawn on a pitch");
  img(s, "viewer_whatif.png", 8.72, 4.35, 3.0, 1.69, "Three.js viewer with what-if ghost layers");
  txt(s, "GRF video: no change (left) and what-if (right)", 1.12, 5.55, 3.6, 0.5, { size: 10.5, color: C.gray });
  txt(s, "Drawn pitch: real match (left) and what-if (right)", 4.92, 5.82, 3.6, 0.5, { size: 10.5, color: C.gray });
  txt(s, "3D viewer: real match plus 7 runs to switch on and off", 8.72, 6.12, 3.0, 0.5, { size: 10.5, color: C.gray });
  s.addNotes("Part C: we start GRF from a real moment, 2.6 seconds before a real goal. We change one thing, then the team models play on for twelve seconds. GRF has luck in it, so each change is played twenty times, always with the same twenty seeds as the 'no change' runs. Each what-if run therefore has a twin with the same luck, and the difference between the twins is the effect of the change. Part D shows it: a GRF video side by side, the real match next to the what-if on a drawn pitch, and the 3D viewer.");
}

// ---------------------------------------------------------------- 8 Parts C + D results
{
  const s = pres.addSlide();
  frame(s, 8, "Results C: a quick forward pass lifts goals from 25% to 65%");
  label(s, "RUNS THAT END IN A GOAL (OUT OF 20)", 1.12, 1.55, 7);
  const r = [["No change", 25, C.gray], ["Quick forward pass", 65, C.red], ["Back four 3 m deeper", 15, C.light], ["Back four 3 m higher", 30, C.light],
    ["Attack plays as Croatia", 30, C.light], ["Attack plays as France", 30, C.light], ["Attack plays as Argentina", 15, C.light]];
  r.forEach(([name, v, col], i) => {                    // bars drawn as in the template's results slide
    const y = 2.0 + i * 0.56;
    txt(s, name, 1.12, y, 2.75, 0.42, { size: 12.5, bold: i < 2, valign: "middle" });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 3.95, y, w: 3.9, h: 0.42, fill: { color: C.card }, rectRadius: 0.05, line: { color: C.card, width: 0 } });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 3.95, y, w: 3.9 * v / 100, h: 0.42, fill: { color: col }, rectRadius: 0.05, line: { color: col, width: 0 } });
    txt(s, `${v / 5} of 20  (${v}%)`, 3.95 + 3.9 * v / 100 + 0.1, y, 1.6, 0.42, { size: 12, bold: true, valign: "middle" });
  });
  txt(s, "65% =", 8.55, 1.6, 1.3, 0.6, { size: 26, bold: true, color: C.red });
  txt(s, "13 of 20 runs score, against 5 of 20 with no change.", 8.55, 2.2, 3.8, 0.6, { size: 12.5 });
  card(s, 8.55, 2.95, 3.8, 1.45, C.greenT, null);
  txt(s, "IS IT REAL OR LUCK?", 8.72, 3.05, 3.5, 0.25, { size: 11, bold: true, color: C.green });
  txt(s, "Gain: +0.40 goals per run, give or take 0.26. Even the low end (+0.14) is above zero, so it is a real effect.", 8.72, 3.35, 3.5, 1.0, { size: 11.5 });
  card(s, 8.55, 4.55, 3.8, 1.45);
  txt(s, "THE OTHER CHANGES", 8.72, 4.65, 3.5, 0.25, { size: 11, bold: true, color: C.gray });
  txt(s, "Moving the defence or swapping the team's style changes 1 or 2 goals in 20. That is within luck: no real effect.", 8.72, 4.95, 3.5, 1.0, { size: 11.5 });
  card(s, 1.12, 6.05, 7.0, 0.62, C.amberT, null);
  txt(s, "One moment and our own models: it shows the method works, not a rule for real football yet.", 1.27, 6.05, 6.8, 0.62, { size: 11.5, color: C.amber, valign: "middle" });
  s.addNotes("Out of twenty runs with no change, five end in a goal. If the player on the ball passes forward at once, thirteen of twenty end in a goal. Is that luck? The gain is plus 0.40 goals per run, with a margin of 0.26 either way; even the low end is above zero, so the effect is real. Moving the back four three metres, or making the attack play like Croatia, France or Argentina, changes the count by one or two goals in twenty, which is within luck. Limit: one moment, our own models.");
}

// ---------------------------------------------------------------- 9 metrics explained
{
  const s = pres.addSlide();
  frame(s, 9, "How we measured: each number, how we got it, and how to say it");
  const head = ["QUESTION", "HOW WE GOT THE NUMBER", "RESULT", "HOW TO SAY IT"];
  const W = [2.3, 3.9, 1.6, 3.43];
  const xs = W.reduce((a, w, i) => (a.push(i ? a[i - 1] + W[i - 1] : 1.12), a), []);
  head.forEach((h, j) => txt(s, h, xs[j] + 0.1, 1.5, W[j] - 0.15, 0.25, { size: 9.5, bold: true, color: j === 2 ? C.red : C.gray }));
  const m = [
    ["1  Is the data good enough?", "PFF marks every position as seen by a camera or estimated. We count the seen ones.", "57% seen", "More than half of all positions were really seen; the rest is filled in."],
    ["2  Can we predict movement?", "Hide the next 2 s, let the model guess, measure how far the guess is from the real spot. Compare with 'keep running straight'.", "0.44 m vs 0.75 m", "Our guess is 41% closer than the simple guess: about half a step off."],
    ["3  Do teams move differently?", "Give each team's unseen match the wrong team label and see if the guesses get worse.", "Right label best 1 of 3", "The label hardly matters: team movement style is not learned yet."],
    ["4  Do teams decide differently?", "From every real pass and shot: who received it, and from how far the team shot.", "26% vs 10%\n37 m vs 27 m", "The model finds the real receiver 2.6 times more often than chance; Croatia shoots 10 m further out."],
    ["5  Does it change results?", "88 GRF matches, every pair, both halves. Count wins, draws, losses.", "21 vs 9 wins\n(of 36 each)", "The mixed model wins 21 of 36, Argentina 9: shooting habits decide."],
    ["6  Can we test tactics?", "Same moment 20 times with and without one change, same seeds. Count goals.", "5 → 13 goals\n(of 20 runs)", "A quick forward pass more than doubles the goal chance, and it is not luck."],
  ];
  m.forEach((r, i) => {
    const y = 1.8 + i * 0.8;
    if (i % 2 === 0) s.addShape(pres.shapes.RECTANGLE, { x: 1.12, y, w: 11.23, h: 0.76, fill: { color: C.card }, line: { color: C.card, width: 0 } });
    r.forEach((v, j) => txt(s, v, xs[j] + 0.1, y, W[j] - 0.18, 0.76, { size: j === 2 ? 12.5 : 10.5, bold: j === 0 || j === 2, color: j === 2 ? C.red : C.text, valign: "middle" }));
  });
  txt(s, "Every result is a number, next to a simple baseline (chance, 'keep running', 'no change'), so the audience can judge how good it is.", 1.12, 6.68, 10, 0.3, { size: 10.5, color: C.gray });
  s.addNotes("How to explain each value. 1: PFF marks each position as seen or estimated; 57% were seen. 2: we hide the next two seconds, the model guesses, and we measure how far the guess is from the real spot; our model is off by 0.44 m, the simple guess 'keep running straight' by 0.75 m, so ours is 41% closer, about half a step. 3: we give each team's unseen match the wrong label; if the model had learned a team style, the wrong label would hurt; it doesn't, the right label was best for only one team of three. 4: the model picks the real pass receiver 26% of the time; picking one of ten teammates at random gives 10%, so 2.6 times better than chance; Croatia shoots from 37 m, Argentina from 27 m. 5: 88 GRF matches; the mixed model wins 21 of 36, Argentina 9 of 36. 6: the same moment twenty times with and without the change, same luck each time; no change scores 5 of 20, a quick forward pass 13 of 20; the margin of error stays above zero, so it is real.");
}

// ---------------------------------------------------------------- 10 commands
{
  const s = pres.addSlide();
  frame(s, 10, "Demo commands: scores, GRF, Three.js and videos");
  txt(s, "Run every command from:  cd C:\\Users\\Srinivasan\\grf-prototype", 1.12, 1.48, 11.2, 0.3, { size: 11.5, bold: true, color: C.gray, font: MONO });
  const g = [
    ["1  SCORES", [["League table (models vs models)", "python scripts\\team_league.py --report"],
      ["What-if results table", "type outputs\\whatif\\J03WOH_3500_3610_summary.csv"], ["What-if chart", "start outputs\\whatif\\J03WOH_3500_3610_chart.png"]]],
    ["2  GRF LIVE (opens the game window)", [["Real match in GRF", ".venv\\Scripts\\python.exe scripts\\grf_pass.py --segment data\\real\\processed\\J03WOH_segment_3500.npz --goal-s 3612.6 --render"],
      ["Model vs model match", ".venv\\Scripts\\python.exe scripts\\team_league.py --video Argentina France"],
      ["What-if runs", ".venv\\Scripts\\python.exe scripts\\whatif.py --video-only"]]],
    ["3  THREE.JS VIEWER", [["Start the viewer", "python -m http.server 8765 --directory viewer"],
      ["Real match", "http://localhost:8765/?file=J03WOH_full.json"], ["Real match + model runs", "http://localhost:8765/?file=J03WOH_whatif_3610.json"]]],
    ["4  VIDEOS (ready to play)", [["Real match in GRF", "start outputs\\pass2_3min_GRF_h264.mp4"], ["Model vs model", "start outputs\\team_league\\Argentina_vs_France_h264.mp4"],
      ["What-if in GRF", "start outputs\\whatif\\J03WOH_3500_3610_early_pass_grf_h264.mp4"]]],
  ];
  g.forEach(([h, items], k) => {
    const x = 1.12 + (k % 2) * 5.68, y = 1.88 + Math.floor(k / 2) * 2.45, w = 5.55;
    card(s, x, y, w, 2.33);
    txt(s, h, x + 0.17, y + 0.1, w - 0.3, 0.27, { size: 11.5, bold: true, color: C.red });
    items.forEach(([lab, cmd], i) => {
      const yy = y + 0.45 + i * 0.62;
      txt(s, lab, x + 0.17, yy, w - 0.3, 0.2, { size: 10, bold: true });
      txt(s, cmd, x + 0.17, yy + 0.21, w - 0.3, 0.38, { size: 8.5, color: C.text, font: MONO });
    });
  });
  s.addNotes("All commands run from the project folder. Scores: the league report prints the table; the what-if summary and chart are ready files. GRF live: these open the game window; the model match plays a full 5-minute GRF match and records it, the what-if command records all what-if runs again. Three.js: start the small web server, then open the two links in a browser; Ctrl+C stops it. Videos: ready to play, no waiting.");
}

// ---------------------------------------------------------------- 11 where we are
{
  const s = pres.addSlide();
  frame(s, 11, "Where we are: what works, what doesn't yet, next steps");
  const cols = [["WHAT WORKS", C.green, C.greenT, ["A real match replayed in GRF that even scores the real goal", "3D viewer for 94 real matches", "Team models play full GRF matches with scores", "What-if tests that find a real effect"]],
    ["NOT YET", C.red, C.redT, ["One fully continuous real match in GRF", "Team style in movement (too few matches per team)", "Beating GRF's own built-in AI", "What-if on more than one moment"]],
    ["NEXT STEPS", C.blue, C.blueT, ["Run what-if on many moments: goals, chances, lost balls", "Add club data for stronger team models", "Train the models inside GRF", "Which what-if should we test first?"]]];
  cols.forEach(([h, col, tint, items], i) => {
    const x = 1.12 + i * 3.81;
    card(s, x, 1.6, 3.6, 4.7, tint, null);
    txt(s, h, x + 0.22, 1.75, 3.2, 0.3, { size: 14, bold: true, color: col });
    items.forEach((t, k) => {
      const y = 2.3 + k * 0.95;
      s.addShape(pres.shapes.OVAL, { x: x + 0.22, y: y + 0.08, w: 0.14, h: 0.14, fill: { color: col }, line: { color: col, width: 0 } });
      txt(s, t, x + 0.48, y, 2.95, 0.85, { size: 12.5 });
    });
  });
  txt(s, "Discussion: which what-if question matters most for the project, and should we add club data next?", 1.12, 6.45, 9.9, 0.3, { size: 13, bold: true, color: C.red });
  s.addNotes("To close: we can replay a real match in GRF closely, show any match in 3D, have team models play each other with clear scores, and run what-if tests that find a real effect. Not yet: a fully continuous real match in GRF, team style in movement, and beating GRF's own AI. Next: run the what-if on many moments, add club data, and train inside GRF. Question for you: which what-if should we test first?");
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote", f));
