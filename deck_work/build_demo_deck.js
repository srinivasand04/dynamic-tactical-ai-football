// 10-slide demo deck (DVRC / ESILV style, as build_deck.js): real match in GRF, its challenges, the Three.js viewer,
// Parts A-D (team models, GRF league, what-if, videos) as flowcharts with results, the metrics, and next steps.
const pptxgen = require(process.env.PPTXGENJS || "pptxgenjs");
const path = require("path");
const D = __dirname.replace(/\\/g, "/");
const I = D + "/demo/";
const OUT = path.join(D, "..", "presentations", "Dynamic_Tactical_AI_Demo_Nov_2026.pptx");

const C = { text: "26333D", gray: "63717B", dark: "0D5864", green: "27836F", blue: "397CA0", teal: "147D8A", orange: "D9822B", navy: "15324A",
  red: "B8473A", line: "D7E0E4", light: "F4F7F8", greenT: "E8F3EF", blueT: "EAF2F7", tealT: "E7F3F4", orangeT: "FBF1E5", redT: "F8EAE7", white: "FFFFFF" };
const F = "Arial";
const N = 10;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Dynamic Tactical AI: real matches, team models and what-if in GRF";
pres.author = "Srinivasan Dhakshanamoorthy";
pres.company = "ESILV / DVRC";

function txt(s, text, x, y, w, h, o = {}) {
  s.addText(text, { x, y, w, h, fontFace: F, fontSize: o.size || 13.5, bold: !!o.bold, color: o.color || C.text, align: o.align || "left",
    valign: o.valign || "top", margin: 0, isTextBox: true, fit: "none", italic: !!o.italic, paraSpaceAfter: o.after || 0, objectName: o.name });
}
function box(s, x, y, w, h, fill, o = {}) {
  s.addShape(o.round === false ? pres.shapes.RECTANGLE : pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, rectRadius: o.r === undefined ? 0.1 : o.r,
    line: o.border ? { color: o.border, width: 0.75 } : { color: fill, width: 0 }, objectName: o.name });
}
function arrow(s, x1, y1, x2, y2, color = C.teal) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipV: y2 < y1, flipH: x2 < x1,
    line: { color, width: 1.75, endArrowType: "triangle" } });
}
function frame(s, n, section, title) {
  s.background = { color: C.white };
  txt(s, section, 0.65, 0.29, 9.9, 0.26, { size: 11.25, bold: true, color: C.gray, valign: "middle", name: "Section" });
  txt(s, title, 0.65, 0.64, 10.9, 0.6, { size: 24, bold: true, color: C.dark, valign: "middle", name: "Title" });
  s.addImage({ path: D + "/logo.png", x: 11.72, y: 0.24, w: 1.08, h: 0.43, altText: "ESILV logo" });
  s.addShape(pres.shapes.LINE, { x: 0.65, y: 1.36, w: 12.02, h: 0, line: { color: C.line, width: 0.75 } });
  s.addShape(pres.shapes.LINE, { x: 0.65, y: 7.12, w: 12.02, h: 0, line: { color: C.line, width: 0.75 } });
  txt(s, "DYNAMIC TACTICAL AI  ·  DVRC / ESILV", 0.65, 7.19, 7.29, 0.19, { size: 8.25, color: C.gray, valign: "middle", name: "Footer" });
  txt(s, `${n} / ${N}`, 12.1, 7.19, 0.57, 0.19, { size: 8.25, bold: true, color: C.gray, align: "right", valign: "middle", name: "Page" });
}
// a flowchart step: number circle, title, one line of explanation
function step(s, n, x, y, w, h, title, body, col, tint) {
  box(s, x, y, w, h, tint);
  s.addShape(pres.shapes.OVAL, { x: x + 0.15, y: y + 0.15, w: 0.42, h: 0.42, fill: { color: col }, line: { color: col, width: 0 } });
  txt(s, String(n), x + 0.15, y + 0.15, 0.42, 0.42, { size: 13, bold: true, color: C.white, align: "center", valign: "middle" });
  txt(s, title, x + 0.68, y + 0.13, w - 0.8, 0.5, { size: 13.5, bold: true, color: col, valign: "middle" });
  txt(s, body, x + 0.17, y + 0.72, w - 0.32, h - 0.82, { size: 12, color: C.text });
}
// a big number with a label and a short explanation
function stat(s, x, y, w, h, big, label, body, col, fill = C.light) {
  box(s, x, y, w, h, fill, { border: fill === C.light ? C.line : undefined });
  txt(s, big, x + 0.2, y + 0.12, w - 0.4, 0.55, { size: 26, bold: true, color: col, valign: "middle" });
  txt(s, label, x + 0.2, y + 0.7, w - 0.4, 0.3, { size: 13, bold: true });
  txt(s, body, x + 0.2, y + 1.02, w - 0.4, h - 1.1, { size: 11.5, color: C.gray });
}
function img(s, file, x, y, w, h, alt) {
  s.addImage({ path: I + file, x, y, w, h, altText: alt });
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { type: "none" }, line: { color: C.line, width: 0.75 } });
}

// ---------------------------------------------------------------- 1 title
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.19, h: 7.5, fill: { color: C.teal }, line: { color: C.teal, width: 0 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.19, y: 0, w: 0.04, h: 7.5, fill: { color: C.orange }, line: { color: C.orange, width: 0 } });
  s.addImage({ path: D + "/logo.png", x: 10.94, y: 0.43, w: 1.77, h: 0.71, altText: "ESILV logo" });
  txt(s, "DEMO  ·  NOVEMBER 2026", 0.81, 0.97, 7.29, 0.31, { size: 12.75, bold: true, color: C.orange, valign: "middle" });
  txt(s, [{ text: "Dynamic", options: { breakLine: true } }, { text: "Tactical AI" }], 0.81, 1.67, 7.66, 1.56, { size: 43.5, bold: true, color: C.navy, valign: "middle", name: "Title" });
  txt(s, "Real matches in GRF, team models that play each other, and what-if replays", 0.85, 3.49, 7.6, 0.95, { size: 22.5, bold: true, color: C.dark });
  s.addShape(pres.shapes.LINE, { x: 0.85, y: 4.7, w: 7.71, h: 0, line: { color: C.line, width: 0.75 } });
  txt(s, "GRF replay  ·  3D viewer  ·  A Team models  ·  B League  ·  C What-if  ·  D Videos", 0.85, 4.98, 7.9, 0.6, { size: 14, color: C.gray });
  txt(s, "Srinivasan Dhakshanamoorthy", 0.85, 6.19, 4.79, 0.31, { size: 14.25, bold: true, valign: "middle" });
  txt(s, "MSc Data Engineering & AI  |  DVRC / ESILV  |  Supervisor: Prof. Ahmed Azough", 0.85, 6.52, 8.12, 0.26, { size: 11.25, color: C.gray, valign: "middle" });
  img(s, "grf_pass_frame.png", 8.95, 2.2, 3.75, 2.11, "A real Bundesliga 2 match replayed inside Google Research Football");
  box(s, 8.95, 4.5, 3.75, 0.32, C.tealT, { r: 0.08 });
  txt(s, "REAL TRACKING  →  GRF  →  WHAT-IF", 9.08, 4.5, 3.5, 0.32, { size: 10.5, bold: true, color: C.dark, valign: "middle" });
  s.addNotes("Today I show four things. First, how we put a real match inside Google Research Football (GRF) and what made that hard. Second, the 3D viewer that shows the real match exactly as recorded. Third, team models for Argentina, France and Croatia that play full matches against each other in GRF. Fourth, what-if replays: we take a real moment, change one decision, and measure what happens.");
}

// ---------------------------------------------------------------- 2 real match in GRF
{
  const s = pres.addSlide();
  frame(s, 2, "01 · REAL MATCH IN GRF", "We replay a real match inside GRF by steering all 22 players");
  const steps = [
    ["Real tracking", "Every player and the ball, 10 times a second (DFL, Bundesliga 2).", C.teal, C.tealT],
    ["Start GRF from reality", "GRF is placed at the real positions of all 22 players and the ball.", C.blue, C.blueT],
    ["Steer, pass, shoot", "Each player walks his real path; at a real kick he presses pass or shot.", C.green, C.greenT],
    ["Video with real clock", "GRF draws the game; a banner shows the real match time.", C.orange, C.orangeT],
  ];
  steps.forEach(([t, b, col, tint], i) => {
    const x = 0.65 + i * 3.08;
    step(s, i + 1, x, 1.62, 2.8, 1.62, t, b, col, tint);
    if (i < 3) arrow(s, x + 2.82, 2.43, x + 3.06, 2.43, C.gray);
  });
  img(s, "grf_pass_frame.png", 0.65, 3.5, 6.1, 3.43, "GRF replay of the real match with the real match clock banner");
  stat(s, 7.05, 3.5, 2.7, 1.62, "0.54 m", "players, goal clip", "3 minutes around a real goal; GRF also scored the goal", C.teal);
  stat(s, 9.97, 3.5, 2.7, 1.62, "1.43 m", "ball, goal clip", "average distance from the real ball", C.teal);
  stat(s, 7.05, 5.31, 2.7, 1.62, "1.15 m", "players, with passes", "3 minutes, 23 real kicks replayed by GRF", C.blue);
  stat(s, 9.97, 5.31, 2.7, 1.62, "3.94 m", "ball, with passes", "the ball is GRF's: physics decides its flight", C.orange);
  s.addNotes("How it works: GRF only lets you place players at the start, so we start GRF from the real positions and then steer every player along his real path. When the real player kicks the ball, our player presses pass or shot at the same moment. Results: in the 3-minute clip around a real goal, GRF players are on average 0.54 m from the real players and the ball 1.43 m; GRF scored the goal as well. With real passes the ball error is larger, about 4 m, because the ball flight is decided by GRF's physics, not by us.");
}

// ---------------------------------------------------------------- 3 challenges
{
  const s = pres.addSlide();
  frame(s, 3, "01 · REAL MATCH IN GRF", "Why it is hard, and what we did about it");
  const rows = [
    ["GRF takes only button presses", "There is no 'put player here' after kick-off.", "Steer each player to his real position every 0.1 s.", C.teal, C.tealT],
    ["The ball leaves the foot late", "About 0.5 s after the pass button, so the ball was 5–8 m behind.", "Press the button 0.7 s before the real kick.", C.blue, C.blueT],
    ["One long game drifts away", "Small errors grow: after a few minutes the ball is 20 m off on average.", "Restart GRF from reality in short windows.", C.orange, C.orangeT],
    ["Restarts are visible", "Each restart jumped the camera, reset the clock, froze players.", "Hidden warm-up, real-clock banner, restart only when in sync.", C.green, C.greenT],
  ];
  txt(s, "CHALLENGE", 0.65, 1.55, 3.6, 0.25, { size: 10.5, bold: true, color: C.gray });
  txt(s, "WHAT WENT WRONG", 4.35, 1.55, 3.6, 0.25, { size: 10.5, bold: true, color: C.gray });
  txt(s, "OUR FIX", 8.1, 1.55, 3.6, 0.25, { size: 10.5, bold: true, color: C.gray });
  rows.forEach(([c, w, f, col, tint], i) => {
    const y = 1.88 + i * 0.86;
    box(s, 0.65, y, 3.55, 0.74, tint);
    txt(s, c, 0.82, y, 3.3, 0.74, { size: 13, bold: true, color: col, valign: "middle" });
    txt(s, w, 4.35, y, 3.55, 0.74, { size: 12, valign: "middle" });
    arrow(s, 7.88, y + 0.37, 8.06, y + 0.37, C.gray);
    box(s, 8.1, y, 4.57, 0.74, C.light, { border: C.line });
    txt(s, f, 8.27, y, 4.3, 0.74, { size: 12, bold: true, valign: "middle" });
  });
  stat(s, 0.65, 5.4, 3.85, 1.55, "53 → 8", "sudden cuts in 3 minutes", "after the fixes, most restarts can't be seen", C.green);
  stat(s, 4.74, 5.4, 3.85, 1.55, "72%", "restarts invisible", "the replay stays in step with the real match", C.teal);
  stat(s, 8.82, 5.4, 3.85, 1.55, "20 m", "ball drift in one long game", "why a fully continuous real match in GRF is not possible", C.red);
  s.addNotes("The honest part. GRF is a game: you control it with buttons, not coordinates. So we steer. The ball leaves the foot about half a second after the button, so we press early. A single continuous game drifts: after a few minutes the ball is 20 m away from reality, so we restart from the real data in short windows. Restarts used to be very visible; with a hidden warm-up, a real-clock banner and restarting only when GRF is already in step, sudden cuts went from 53 to 8 in three minutes, and 72% of the restarts cannot be seen. Note: GRF's player names are invented; that is why we show shirt numbers.");
}

// ---------------------------------------------------------------- 4 Three.js viewer
{
  const s = pres.addSlide();
  frame(s, 4, "02 · THREE.JS VIEWER (OPTIONAL)", "The real match exactly as recorded, in 3D in the browser");
  img(s, "viewer_real.png", 0.65, 1.6, 8.0, 4.5, "Three.js viewer: real players with shirt numbers and GRF ghost players");
  txt(s, "Solid players = the real match.  Ringed ghosts = a GRF run of the same moment, with a line to the real player.", 0.65, 6.25, 8.0, 0.6, { size: 11.5, color: C.gray });
  const pts = [
    ["No GRF limits", "Draws the data as it was: no drift, no restarts, no fake names.", C.teal],
    ["Compare runs", "Any GRF replay or what-if appears as ghost players, with the live distance in metres.", C.blue],
    ["Five cameras", "Broadcast, top, behind goal, follow ball, free; timeline and speed.", C.green],
    ["Any match", "94 matches from 3 datasets; runs in any browser, 4–7 ms per frame.", C.orange],
  ];
  pts.forEach(([t, b, col], i) => {
    const y = 1.6 + i * 1.13;
    box(s, 8.9, y, 3.77, 1.0, C.light, { border: C.line });
    txt(s, t, 9.08, y + 0.1, 3.45, 0.3, { size: 13.5, bold: true, color: col });
    txt(s, b, 9.08, y + 0.42, 3.45, 0.55, { size: 11.5 });
  });
  box(s, 8.9, 6.15, 3.77, 0.75, C.tealT);
  txt(s, "python -m http.server 8765 --directory viewer", 9.05, 6.15, 3.5, 0.75, { size: 10.5, bold: true, color: C.dark, valign: "middle" });
  s.addNotes("Because GRF can't show a real match continuously, we built a second view: a Three.js page that draws the tracking data exactly as recorded, in 3D, in any browser. GRF runs and what-if runs of the same moment appear as ghost players next to the real ones, with a line to the matching real player and the live distance in metres. It works for all 94 matches. This is the main tool for showing what-if results to coaches.");
}

// ---------------------------------------------------------------- 5 Parts A + B flow
{
  const s = pres.addSlide();
  frame(s, 5, "03 · PART A TEAM MODELS  +  PART B LEAGUE IN GRF", "From World Cup data to team models that play each other");
  box(s, 0.65, 1.55, 6.0, 0.36, C.tealT, { r: 0.08 });
  txt(s, "PART A  ·  BUILD A MODEL FOR EACH TEAM", 0.8, 1.55, 5.8, 0.36, { size: 11.5, bold: true, color: C.dark, valign: "middle" });
  box(s, 6.82, 1.55, 5.85, 0.36, C.orangeT, { r: 0.08 });
  txt(s, "PART B  ·  LET THE MODELS PLAY IN GRF", 6.97, 1.55, 5.6, 0.36, { size: 11.5, bold: true, color: C.orange, valign: "middle" });
  const top = [
    ["World Cup data", "64 matches (PFF FC). Argentina, France, Croatia: 7 matches each.", C.teal, C.tealT],
    ["Movement model", "Graph network: 'where will each player be in 2 s?', told the team.", C.blue, C.blueT],
    ["Decision model", "Who gets the pass, when to shoot, how long to hold the ball.", C.green, C.greenT],
  ];
  top.forEach(([t, b, col, tint], i) => {
    const y = 2.1 + i * 1.6;
    step(s, i + 1, 0.65, y, 6.0, 1.38, t, b, col, tint);
    if (i < 2) arrow(s, 3.65, y + 1.39, 3.65, y + 1.59, C.gray);
  });
  arrow(s, 6.66, 1.73, 6.81, 1.73, C.gray);                       // Part A feeds Part B
  const bot = [
    ["A team 'brain' in GRF", "Each team model controls its 11 GRF players, every 0.1 s.", C.orange, C.orangeT],
    ["Round robin", "3 teams + a pooled model; every pair 12 matches, sides swapped; 88 matches.", C.navy, C.light],
    ["Scores", "Points, goal difference with a 95% interval, shots, xG, passes, possession.", C.red, C.redT],
  ];
  bot.forEach(([t, b, col, tint], i) => {
    const y = 2.1 + i * 1.6;
    step(s, i + 4, 6.82, y, 5.85, 1.38, t, b, col, tint);
    if (i < 2) arrow(s, 9.75, y + 1.39, 9.75, y + 1.59, C.gray);
  });
  s.addNotes("Part A: from the World Cup tracking data we train one graph network that predicts where every player will be two seconds later. It is also told which team the player belongs to, so the same network becomes a model of Argentina, France or Croatia. On top, a decision model learns each team's habits: who receives the pass, when the team shoots, how long it keeps the ball. Part B: a team brain uses both models to control eleven GRF players. All teams play all teams, plus a pooled model that is not told any team, 88 matches in total, with proper scores and 95% intervals.");
}

// ---------------------------------------------------------------- 6 Parts A + B results
{
  const s = pres.addSlide();
  frame(s, 6, "03 · PART A + B RESULTS", "Teams differ in how they decide, not in how they move");
  stat(s, 0.65, 1.6, 2.9, 1.7, "0.44 m", "movement error at 2 s", "the simple guess 'keep running' is 0.75 m off: 41% better", C.teal);
  stat(s, 3.75, 1.6, 2.9, 1.7, "1 of 3", "right team label best", "telling the model the team hardly changes its error", C.orange);
  stat(s, 0.65, 3.5, 2.9, 1.7, "37 vs 27 m", "shot distance", "Croatia shoots from far out, Argentina waits to get close", C.green);
  stat(s, 3.75, 3.5, 2.9, 1.7, "26% vs 10%", "right pass receiver", "Argentina's model picks the real receiver; a random pick gets 10%", C.blue);
  s.addChart(pres.charts.BAR, [{ name: "Points per game", labels: ["Pooled", "Croatia", "France", "Argentina"], values: [1.97, 1.67, 1.17, 0.86] }], {
    x: 6.95, y: 1.55, w: 5.72, h: 3.75, barDir: "bar", chartColors: [C.navy, C.green, C.blue, C.teal], invertedColors: [C.navy],
    showTitle: true, title: "Points per game, 72 matches between the models", titleFontFace: F, titleFontSize: 12, titleColor: C.text,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelFontFace: F, dataLabelColor: C.text, dataLabelFormatCode: "0.00",
    catAxisLabelFontFace: F, catAxisLabelFontSize: 12, catAxisLabelColor: C.text, catAxisOrientation: "maxMin", valAxisHidden: true, valAxisMaxVal: 2.4, valAxisMinVal: 0,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 45 });
  box(s, 0.65, 5.42, 6.0, 1.5, C.tealT);
  txt(s, "WHAT IT MEANS", 0.85, 5.55, 5.6, 0.25, { size: 11, bold: true, color: C.dark });
  txt(s, "The models learned to move like real players. What separates the teams is when they shoot: Croatia shoots 26 times a game, Argentina 9.", 0.85, 5.85, 5.65, 1.0, { size: 12.5 });
  box(s, 6.95, 5.42, 5.72, 1.5, C.redT);
  txt(s, "HONEST BASELINE", 7.15, 5.55, 5.3, 0.25, { size: 11, bold: true, color: C.red });
  txt(s, "Against GRF's own built-in AI every model loses: 0 goals scored, 8–10 conceded per game. GRF's AI is tuned to GRF, ours to real football.", 7.15, 5.85, 5.35, 1.0, { size: 12.5 });
  s.addNotes("Results. Movement: 0.44 m error after two seconds, against 0.75 m for keep-velocity. But when we give Argentina's held-out match the label France or Croatia, the error barely moves; the correct label is best for only one team of three. So team style is not in the movement with seven matches per team. Decisions are different: Croatia shoots from 37 m on average, Argentina from 27 m. In the league this decides the table: pooled 1.97 points per game, Croatia 1.67, France 1.17, Argentina 0.86. Pooled versus Argentina is a clear gap: the 95% intervals don't overlap. And honestly, all our models lose heavily to GRF's own AI.");
}

// ---------------------------------------------------------------- 7 Parts C + D flow
{
  const s = pres.addSlide();
  frame(s, 7, "04 · PART C WHAT-IF  +  PART D VIDEOS", "Take a real moment, change one thing, play it 20 times");
  const st = [
    ["Real moment", "58:50, Düsseldorf v Regensburg, 2.6 s before a real goal.", C.teal, C.tealT],
    ["Change one thing", "Defence 3 m deeper or higher, a quick forward pass, or another team's style.", C.blue, C.blueT],
    ["Play it in GRF", "Team models play on for 12 s; 20 runs per change, same 20 seeds.", C.green, C.greenT],
    ["Compare pairs", "Each run against its 'no change' twin: goals, shots, xG, with a 95% interval.", C.orange, C.orangeT],
  ];
  st.forEach(([t, b, col, tint], i) => {
    const x = 0.65 + i * 3.08;
    step(s, i + 1, x, 1.6, 2.8, 1.75, t, b, col, tint);
    if (i < 3) arrow(s, x + 2.82, 2.47, x + 3.06, 2.47, C.gray);
  });
  box(s, 0.65, 3.55, 12.02, 0.36, C.orangeT, { r: 0.08 });
  txt(s, "PART D  ·  SEE IT: THE SAME WHAT-IF IN THREE VIEWS", 0.8, 3.55, 11.8, 0.36, { size: 11.5, bold: true, color: C.orange, valign: "middle" });
  img(s, "whatif_grf.png", 0.65, 4.05, 3.85, 1.185, "GRF side by side: no change on the left, what-if on the right");
  img(s, "whatif_drawn.png", 4.74, 4.05, 3.85, 1.48, "Real data next to the what-if, drawn on a pitch");
  img(s, "viewer_whatif.png", 8.82, 4.05, 3.85, 2.17, "Three.js viewer with what-if ghost layers");
  txt(s, "GRF video: no change (left) vs what-if (right), same start and seed", 0.65, 5.35, 3.85, 0.6, { size: 11, color: C.gray });
  txt(s, "Drawn pitch: real match (left) vs what-if (right), shirt numbers", 4.74, 5.65, 3.85, 0.6, { size: 11, color: C.gray });
  txt(s, "3D viewer: the real match with 7 ghost layers to switch on and off", 8.82, 6.3, 3.85, 0.6, { size: 11, color: C.gray });
  s.addNotes("Part C is the what-if engine. We start GRF from a real moment, 2.6 seconds before a real goal. We change one thing: the defensive line, the pass decision, or the team style. Then the team models play on for twelve seconds. Because GRF is random, each change is played twenty times, always with the same twenty seeds as the no-change runs, so each what-if run has a twin and the difference is the effect of the change. Part D shows it three ways: GRF video side by side, real against what-if on a drawn pitch, and the 3D viewer with ghost layers.");
}

// ---------------------------------------------------------------- 8 Parts C + D results
{
  const s = pres.addSlide();
  frame(s, 8, "04 · PART C + D RESULTS", "A quick forward pass raises the goal chance from 25% to 65%");
  s.addChart(pres.charts.BAR, [{ name: "Goal in the run", labels: ["No change", "Quick forward pass", "Back four 3 m deeper", "Back four 3 m higher", "Attack plays as Croatia", "Attack plays as France", "Attack plays as Argentina"],
    values: [25, 65, 15, 30, 30, 30, 15] }], {
    x: 0.65, y: 1.55, w: 7.2, h: 4.2, barDir: "bar", chartColors: [C.gray, C.green, C.blue, C.blue, C.teal, C.teal, C.teal],
    showTitle: true, title: "Runs that end in a goal (%), 20 GRF runs each", titleFontFace: F, titleFontSize: 12, titleColor: C.text,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelFontFace: F, dataLabelColor: C.text, dataLabelFormatCode: "0\"%\"",
    catAxisLabelFontFace: F, catAxisLabelFontSize: 12, catAxisLabelColor: C.text, catAxisOrientation: "maxMin", valAxisHidden: true, valAxisMaxVal: 80, valAxisMinVal: 0,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 40 });
  stat(s, 8.1, 1.6, 4.57, 1.75, "+0.40 ± 0.26", "goals per run, quick forward pass", "the interval doesn't include 0, so the effect is real; xG also rises: +0.33 ± 0.31", C.green, C.greenT);
  stat(s, 8.1, 3.55, 4.57, 1.75, "≈ 0", "defence shift and style swap", "every interval includes 0: in this moment, these changes make no measurable difference", C.gray);
  box(s, 0.65, 5.95, 12.02, 0.98, C.tealT);
  txt(s, "HOW TO READ IT", 0.85, 6.05, 3, 0.25, { size: 11, bold: true, color: C.dark });
  txt(s, "Same moment, same random seeds, one change: the gap between the twins is the effect. 140 GRF runs in 15 minutes on a laptop. One moment and our own models, so this shows the method works, not a rule for real football yet.", 0.85, 6.33, 11.6, 0.6, { size: 12 });
  s.addNotes("Results of 140 GRF runs. With no change, 25% of runs end in a goal. If the player on the ball passes forward at once, 65% do: plus 0.40 goals per run, with a 95% interval of plus or minus 0.26, so it does not include zero and the effect is real. Chance quality, the xG proxy, rises too. Moving the back four three metres, or making the attack play like Croatia, France or Argentina, gives differences that are all inside the noise. Limits: one moment and our own models.");
}

// ---------------------------------------------------------------- 9 metrics
{
  const s = pres.addSlide();
  frame(s, 9, "05 · HOW WE MEASURE", "Six numbers, each answering a bigger question");
  const m = [
    ["Is the data good enough?", "Data quality", "57%", "of World Cup positions really seen; the rest is estimated", C.gray],
    ["Can we predict movement?", "Error after 2 s", "0.44 m", "about half a stride; keep-velocity 0.75 m", C.teal],
    ["Do teams move differently?", "Team-label test", "1 of 3", "right label best for one team only: no clear style", C.orange],
    ["Do teams decide differently?", "Pass receiver, shot distance", "26% · 37 m", "vs 10% chance; Croatia 37 m vs Argentina 27 m", C.blue],
    ["Does it change results?", "League points per game", "1.97 → 0.86", "pooled best, Argentina last, 88 GRF matches", C.navy],
    ["Can we test tactics?", "What-if effect", "25% → 65%", "goal chance with a quick forward pass, ± 26 points", C.green],
  ];
  m.forEach(([q, metric, big, body, col], i) => {
    const y = 1.55 + i * 0.9;
    s.addShape(pres.shapes.OVAL, { x: 0.65, y: y + 0.17, w: 0.44, h: 0.44, fill: { color: col }, line: { color: col, width: 0 } });
    txt(s, String(i + 1), 0.65, y + 0.17, 0.44, 0.44, { size: 13, bold: true, color: C.white, align: "center", valign: "middle" });
    box(s, 1.3, y, 11.37, 0.78, i % 2 ? C.white : C.light, { border: C.line });
    txt(s, q, 1.5, y, 3.6, 0.78, { size: 13.5, bold: true, valign: "middle" });
    txt(s, metric, 5.1, y, 2.6, 0.78, { size: 12, color: C.gray, valign: "middle" });
    txt(s, big, 7.7, y, 1.95, 0.78, { size: 17, bold: true, color: col, valign: "middle" });
    txt(s, body, 9.7, y, 2.85, 0.78, { size: 11.5, valign: "middle" });
    if (i < 5) arrow(s, 0.87, y + 0.62, 0.87, y + 1.06, C.line);
  });
  s.addNotes("This is the chain of measurements in order. Each result is a number, compared with a simple baseline, and where there is randomness with a 95% interval. Data quality first, then movement accuracy, then whether the team label carries style, then the decisions, then the league result, then the what-if effect.");
}

// ---------------------------------------------------------------- 10 summary + next
{
  const s = pres.addSlide();
  frame(s, 10, "06 · WHERE WE ARE", "What works, what doesn't yet, and the next steps");
  const cols = [
    ["WORKS", C.green, C.greenT, ["Real match replayed in GRF: players within 0.5–1.2 m", "3D viewer for any of 94 real matches", "Team models play full GRF matches with scores", "What-if engine: one change, 20 runs, a real effect found"]],
    ["NOT YET", C.red, C.redT, ["A fully continuous real match in GRF (the ball drifts 20 m)", "Team style in movement: 7 matches per team is too few", "Our models lose to GRF's own AI", "What-if tested on one moment only"]],
    ["NEXT", C.blue, C.blueT, ["Run what-if on many moments: goals, chances, lost balls", "Add club data (SkillCorner, more PFF) for team style", "Train the models inside GRF to compete with GRF's AI", "Which what-if do you want to test first?"]],
  ];
  cols.forEach(([h, col, tint, items], i) => {
    const x = 0.65 + i * 4.07;
    box(s, x, 1.6, 3.88, 3.45, tint);
    txt(s, h, x + 0.25, 1.75, 3.4, 0.35, { size: 15, bold: true, color: col });
    s.addText(items.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < items.length - 1 } })),
      { x: x + 0.2, y: 2.25, w: 3.5, h: 2.7, fontFace: F, fontSize: 13, color: C.text, valign: "top", margin: 0, isTextBox: true, paraSpaceAfter: 10 });
  });
  box(s, 0.65, 5.3, 12.02, 1.62, C.light, { border: C.line });
  txt(s, "IN THE DEMO", 0.85, 5.42, 4, 0.25, { size: 11, bold: true, color: C.dark });
  const demo = [["Real match in GRF", "outputs/goal_3min_GRF_h264.mp4"], ["Team models' match", "outputs/team_league/Argentina_vs_France_h264.mp4"],
    ["What-if videos", "outputs/whatif/*_grf_h264.mp4"], ["3D viewer", "localhost:8765/?file=J03WOH_whatif_3610.json"]];
  demo.forEach(([a, b], i) => {
    const x = 0.85 + (i % 2) * 5.95, y = 5.75 + Math.floor(i / 2) * 0.55;
    txt(s, a, x, y, 2.1, 0.45, { size: 12, bold: true, valign: "middle" });
    txt(s, b, x + 2.1, y, 3.75, 0.45, { size: 11, color: C.gray, valign: "middle" });
  });
  s.addNotes("To close: we can replay real matches in GRF closely, show any match in 3D, make team models play each other with measurable scores, and run what-if experiments that find a real effect. What doesn't work yet: one fully continuous real match in GRF, team style in movement, and beating GRF's own AI. Next steps: run the what-if engine on many moments, add more team data, and train the models inside GRF. Question for you: which what-if should we test first?");
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote", f));
