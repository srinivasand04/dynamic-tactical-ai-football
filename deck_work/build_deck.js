// 8-slide ESILV deck: Key States, graph types, ball-owner (BO) graph, integration, datasets, current run.
// Style tokens (colours, sizes, positions) are copied from the supplied template, Dynamic_Tactical_AI_Pitch_6_Slides.pptx.
const SCRATCH = "C:/Users/Srinivasan/AppData/Local/Temp/claude/C--Users-Srinivasan-grf-prototype/31840941-a7c1-49f0-9a98-770a29f36e04/scratchpad/node_modules/pptxgenjs";
const pptxgen = require(SCRATCH);
const fs = require("fs");
const path = require("path");
const D = __dirname.replace(/\\/g, "/");
const S = JSON.parse(fs.readFileSync(path.join(D, "stats.json"), "utf8"));

const C = { text: "26333D", gray: "63717B", dark: "0D5864", green: "27836F", blue: "397CA0", teal: "147D8A", orange: "D9822B", navy: "15324A",
  line: "D7E0E4", light: "F4F7F8", greenT: "E8F3EF", blueT: "EAF2F7", tealT: "E7F3F4", orangeT: "FBF1E5", white: "FFFFFF", pale: "8DD3D6" };
const F = "Arial";
const N = 8;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.33 x 7.5 in, as the template
pres.title = "Dynamic Tactical AI: Key States and graphs";
pres.author = "Srinivasan Dhakshanamoorthy";
pres.company = "ESILV / DVRC";

function txt(s, text, x, y, w, h, o = {}) {
  s.addText(text, { x, y, w, h, fontFace: F, fontSize: o.size || 13.5, bold: !!o.bold, color: o.color || C.text, align: o.align || "left",
    valign: o.valign || "top", margin: 0, isTextBox: true, fit: "none", italic: !!o.italic, paraSpaceAfter: o.after || 0, charSpacing: o.cs, objectName: o.name });
}
function box(s, x, y, w, h, fill, o = {}) {
  s.addShape(o.round === false ? pres.shapes.RECTANGLE : pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, rectRadius: o.r === undefined ? 0.1 : o.r,
    line: o.border ? { color: o.border, width: 0.75 } : { color: fill, width: 0 }, objectName: o.name });
}
function arrow(s, x1, y1, x2, y2, color = C.teal) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipV: y2 < y1, flipH: x2 < x1,
    line: { color, width: 1.5, endArrowType: "triangle" } });
}
function frame(s, n, section, title) {
  txt(s, section, 0.65, 0.29, 9.9, 0.26, { size: 11.25, bold: true, color: C.gray, valign: "middle", name: "Section" });
  txt(s, title, 0.65, 0.64, 11.0, 0.6, { size: 25.5, bold: true, color: C.dark, valign: "middle", name: "Title" });
  s.addImage({ path: D + "/logo.png", x: 11.72, y: 0.24, w: 1.08, h: 0.43, altText: "ESILV logo" });
  s.addShape(pres.shapes.LINE, { x: 0.65, y: 1.36, w: 12.02, h: 0, line: { color: C.line, width: 0.75 } });
  s.addShape(pres.shapes.LINE, { x: 0.65, y: 7.12, w: 12.02, h: 0, line: { color: C.line, width: 0.75 } });
  txt(s, "DYNAMIC TACTICAL AI  ·  DVRC / ESILV", 0.65, 7.19, 7.29, 0.19, { size: 8.25, color: C.gray, valign: "middle", name: "Footer" });
  txt(s, `${n} / ${N}`, 12.1, 7.19, 0.57, 0.19, { size: 8.25, bold: true, color: C.gray, align: "right", valign: "middle", name: "Page" });
}
function chip(s, label, x, y, w, color, fill = C.white) {
  box(s, x, y, w, 0.32, fill, { r: 0.08 });
  txt(s, label, x + 0.1, y, w - 0.2, 0.32, { size: 10.5, bold: true, color, valign: "middle" });
}

// ---------------------------------------------------------------- 1 title
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.19, h: 7.5, fill: { color: C.teal }, line: { color: C.teal, width: 0 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.19, y: 0, w: 0.04, h: 7.5, fill: { color: C.orange }, line: { color: C.orange, width: 0 } });
  s.addImage({ path: D + "/logo.png", x: 10.94, y: 0.43, w: 1.77, h: 0.71, altText: "ESILV logo" });
  txt(s, "RESEARCH PITCH  ·  OCTOBER 2026", 0.81, 0.97, 7.29, 0.31, { size: 12.75, bold: true, color: C.orange, valign: "middle" });
  txt(s, [{ text: "Dynamic", options: { breakLine: true } }, { text: "Tactical AI" }], 0.81, 1.67, 7.66, 1.56, { size: 43.5, bold: true, color: C.navy, valign: "middle", name: "Title" });
  txt(s, "Key States and player graphs: from the radius graph to the ball-owner graph", 0.85, 3.49, 7.6, 0.95, { size: 22.5, bold: true, color: C.dark, valign: "top" });
  s.addShape(pres.shapes.LINE, { x: 0.85, y: 4.7, w: 7.71, h: 0, line: { color: C.line, width: 0.75 } });
  txt(s, "Key States  ·  Graph types  ·  Ball-owner graph  ·  Integration  ·  New datasets", 0.85, 4.98, 8.4, 0.6, { size: 15, color: C.gray });
  txt(s, "Srinivasan Dhakshanamoorthy", 0.85, 6.19, 4.79, 0.31, { size: 14.25, bold: true, color: C.text, valign: "middle" });
  txt(s, "MSc Data Engineering & AI  |  DVRC / ESILV  |  Supervisor: Prof. Ahmed Azough", 0.85, 6.52, 8.12, 0.26, { size: 11.25, color: C.gray, valign: "middle" });
  s.addImage({ path: D + "/title_graph.png", x: 9.01, y: 2.42, w: 3.54, h: 2.24, altText: "Player graph drawn on a football pitch" });
  box(s, 9.08, 5.06, 3.12, 0.32, C.tealT, { r: 0.08 });
  txt(s, "REAL + SIMULATED TRACKING", 9.19, 5.06, 2.92, 0.32, { size: 10.5, bold: true, color: C.dark, valign: "middle" });
  s.addNotes("Opening. Today: how a football match is turned into Key States, which graph structures we tested, the new ball-owner (BO) graph from the whiteboard specification, how it joins the radius graph, what is running now, and the datasets we want to add.");
}

// ---------------------------------------------------------------- 2 key states
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  frame(s, 2, "01 · KEY STATES", "A Key State is a frame where play has moved by 3 m or more");
  box(s, 0.65, 1.57, 12.02, 0.72, C.tealT);
  txt(s, "KEY STATE RULE", 0.85, 1.67, 4, 0.23, { size: 10.5, bold: true, color: C.dark });
  txt(s, "Keep a frame when 0.4 × player movement + 0.4 × shape change + 0.2 × ball movement ≥ 3 m", 0.85, 1.94, 11.4, 0.28, { size: 14.25, bold: true });

  const cards = [
    ["PLAYER MOVEMENT · WEIGHT 0.4", "D_abs", "How far every player has moved since the last Key State, matched by player id.", C.green, C.greenT],
    ["SHAPE CHANGE · WEIGHT 0.4", "D_rel", "How much the arrangement of the team changed, seen relative to its own centre.", C.blue, C.blueT],
    ["BALL MOVEMENT · WEIGHT 0.2", "D_ball", "How far the ball travelled since the last Key State.", C.orange, C.orangeT],
  ];
  cards.forEach(([lab, big, body, col, tint], i) => {
    const x = 0.65 + i * 4.07;
    box(s, x, 2.5, 3.88, 1.65, tint);
    txt(s, lab, x + 0.22, 2.64, 3.5, 0.25, { size: 11, bold: true, color: col });
    txt(s, big, x + 0.22, 2.92, 3.4, 0.45, { size: 24, bold: true, color: col });
    txt(s, body, x + 0.22, 3.45, 3.45, 0.62, { size: 12, color: C.text });
  });

  const stats = [
    ["≈ 3,400", "Key States per real match", "about 6% of all frames; the simulator keeps 12%", C.teal],
    ["2 types", "open play and stoppage", "stable across matches: agreement 0.91, and 0.99 when a match is left out", C.green],
    ["31% vs 0.4%", "stoppage scenes", "in real matches versus the simulator: the biggest structural gap we found", C.orange],
  ];
  stats.forEach(([big, lab, body, col], i) => {
    const x = 0.65 + i * 4.07;
    box(s, x, 4.4, 3.88, 1.95, C.light, { border: C.line });
    txt(s, big, x + 0.22, 4.52, 3.5, 0.6, { size: 28, bold: true, color: col, valign: "middle" });
    txt(s, lab, x + 0.22, 5.15, 3.5, 0.3, { size: 14, bold: true });
    txt(s, body, x + 0.22, 5.5, 3.45, 0.75, { size: 11.5, color: C.gray });
  });
  txt(s, "Everything is measured in metres on a 105 × 68 m pitch at 10 frames per second. Boundaries between sequences come from geometry only; football events describe them afterwards.", 0.65, 6.55, 12.02, 0.45, { size: 11.5, color: C.gray });
  s.addNotes("Key State rule: D_KS = 0.4 D_abs + 0.4 D_rel + 0.2 D_ball, threshold 3 m. About 3,400 Key States per real match (23,852 in total). Sequences have boundaries from geometry only (5 s window); football events (shot, foul, ball out) are used only to describe them. Only two types are stable: open play and stoppage. Stoppage scenes are 31% in real matches and 0.4% in the simulator.");
}

// ---------------------------------------------------------------- 3 graph types
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  frame(s, 3, "02 · GRAPH TYPES", "The graph helps a lot; the exact graph type matters little");
  txt(s, "RADIUS GRAPH", 1.0, 1.57, 4.5, 0.24, { size: 11.25, bold: true, color: C.dark });
  txt(s, "team: 5 nearest mates  ·  opponents within 8 m  ·  ball: 3 nearest", 1.0, 1.82, 4.6, 0.21, { size: 9.75, color: C.gray });
  txt(s, "DELAUNAY GRAPH", 5.75, 1.57, 4.0, 0.24, { size: 11.25, bold: true, color: C.dark });
  txt(s, "geometric neighbours  ·  no radius cutoff", 5.75, 1.82, 4.2, 0.21, { size: 9.75, color: C.gray });
  s.addImage({ path: D + "/radius_delaunay.png", x: 0.88, y: 2.1, w: 9.4, h: 3.026, altText: "Radius graph and Delaunay graph on the same Key State" });
  box(s, 10.5, 1.6, 2.17, 3.5, C.blueT);
  txt(s, "DELAUNAY", 10.7, 1.75, 1.8, 0.25, { size: 11, bold: true, color: C.blue });
  txt(s, "1.17 m", 10.7, 2.05, 1.8, 0.45, { size: 22, bold: true, color: C.blue });
  txt(s, "against 1.13 m for the radius graph on the same held-out match: no gain, so the radius graph is kept.", 10.7, 2.6, 1.8, 1.7, { size: 11.5 });
  txt(s, "Early test, 3 training matches", 10.7, 4.55, 1.8, 0.4, { size: 9.75, color: C.gray });

  txt(s, "PLAYER ERROR ON HELD-OUT REAL MATCHES · 3 SHARED MATCHES · KEEP-VELOCITY BASELINE 1.52 m", 0.65, 5.38, 12, 0.25, { size: 11, bold: true, color: C.gray });
  const rows = [["No graph (control)", "1.21 m", "20% better than keep-velocity", C.gray, C.light], ["Radius graph (ours)", "1.11 m", "27% better", C.teal, C.tealT],
    ["GCN", "1.11 m", "27% better", C.blue, C.blueT], ["GAT", "1.11 m", "27% better", C.orange, C.orangeT]];
  rows.forEach(([name, val, sub, col, tint], i) => {
    const x = 0.65 + i * 3.053;
    box(s, x, 5.72, 2.86, 1.2, tint);
    txt(s, val, x + 0.2, 5.8, 2.5, 0.55, { size: 26, bold: true, color: col, valign: "middle" });
    txt(s, name, x + 0.2, 6.36, 2.5, 0.26, { size: 12.5, bold: true });
    txt(s, sub, x + 0.2, 6.62, 2.5, 0.22, { size: 10, color: C.gray });
  });
  s.addNotes("We tested four graph types on the same held-out real matches. Without any graph the error is 1.21 m; with the radius graph 1.11 m, a 27% improvement over keep-velocity. GCN and GAT are equal to our edge-type-aware layer, and the Delaunay mesh gave no gain. Conclusion: the graph matters, the exact type matters little. The matches shared by all four runs are 3.");
}

// ---------------------------------------------------------------- 4 BO graph
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  frame(s, 4, "03 · BALL-OWNER (BO) GRAPH", "The BO graph turns a pass into a weighted, explainable choice");
  s.addImage({ path: D + "/bo_graph_slide_crop.png", x: 0.65, y: 1.6, w: 7.45, h: 4.37, altText: "Ball-owner graph on a real match frame" });
  txt(s, "Real match, one Key State. Ring = ball owner. Green = the best five pass options, with their weight. Orange × = a blocking opponent. Red dashed with a minus = offside, size kept. Dashed vertical line = offside line.", 0.65, 6.1, 7.45, 0.75, { size: 11, color: C.gray });

  txt(s, "EDGE WEIGHT: FOUR INGREDIENTS", 8.45, 1.55, 4.25, 0.25, { size: 11.25, bold: true, color: C.dark });
  const ing = [["1", "Free space", "distance to his nearest opponent", C.green], ["2", "Pass length", "shorter is easier", C.blue],
    ["3", "Goal closeness", "to the centre of the goal", C.teal], ["4", "Open lane", "a blocker on the line counts less when far from the ball (lofted pass)", C.dark]];
  ing.forEach(([n, head, body, col], i) => {
    const y = 1.92 + i * 0.86;
    box(s, 8.45, y, 4.22, 0.74, C.light, { border: C.line });
    box(s, 8.6, y + 0.15, 0.44, 0.44, col, { r: 0.08 });
    txt(s, n, 8.6, y + 0.15, 0.44, 0.44, { size: 15, bold: true, color: C.white, align: "center", valign: "middle" });
    txt(s, head, 9.2, y + 0.07, 3.35, 0.25, { size: 12.5, bold: true, color: col });
    txt(s, body, 9.2, y + 0.33, 3.35, 0.38, { size: 10.5, color: C.text });
  });
  box(s, 8.45, 5.4, 4.22, 1.2, C.orangeT);
  txt(s, "OFFSIDE KEEPS ITS INFORMATION", 8.62, 5.5, 3.9, 0.24, { size: 10.5, bold: true, color: C.orange });
  txt(s, "The weight is multiplied by −1, not by 0: the size stays, so we can ask what the pass would have been worth.", 8.62, 5.78, 3.9, 0.78, { size: 12 });
  s.addNotes("This is the whiteboard specification: the ball owner is connected to every teammate, and to opponents within a threshold. The weight combines free space around the teammate, pass length, closeness to the centre of the goal and an open passing lane (a far blocker blocks less, for lofted passes). Offside teammates keep their weight but with a minus sign. The numbers in the picture use the distances learned from real passes.");
}

// ---------------------------------------------------------------- 5 validation
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  frame(s, 5, "03 · BALL-OWNER (BO) GRAPH", "On 3,987 real passes the BO graph beats chance 4.5 times over");
  txt(s, "REAL RECEIVER RANKED FIRST (%) · MATCHES NOT USED FOR LEARNING", 0.65, 1.55, 6.2, 0.25, { size: 11, bold: true, color: C.gray });
  s.addChart(pres.charts.BAR, [{ name: "Receiver ranked first (%)", labels: ["Chance", "Nearest teammate", "Hand-set weight", "Learned weight", "+ movement and timing"], values: [10.1, 27.4, 35.2, 41.5, 45.1] }], {
    x: 0.55, y: 1.85, w: 6.4, h: 3.2, barDir: "col", chartColors: ["B4C2C9", "B4C2C9", "8DB7C9", "397CA0", "147D8A"], barGapWidthPct: 45,
    showValue: true, dataLabelFormatCode: "0.0", dataLabelFontFace: F, dataLabelFontSize: 11, dataLabelColor: C.text, dataLabelPosition: "outEnd",
    catAxisLabelFontFace: F, catAxisLabelFontSize: 10, catAxisLabelColor: C.gray, valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 55,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, showTitle: false });
  txt(s, "SHOT WITHIN 10 s OF THE PASS (%) · BY QUARTER OF THE PASS GAIN", 7.3, 1.55, 5.4, 0.25, { size: 11, bold: true, color: C.gray });
  s.addChart(pres.charts.BAR, [{ name: "Shot within 10 s (%)", labels: ["Lowest", "Low", "High", "Highest"], values: [0.3, 0.7, 2.8, 16.3] }], {
    x: 7.2, y: 1.85, w: 5.5, h: 3.2, barDir: "col", chartColors: ["B4C2C9", "B4C2C9", "8DB7C9", "D9822B"], barGapWidthPct: 45,
    showValue: true, dataLabelFormatCode: "0.0", dataLabelFontFace: F, dataLabelFontSize: 11, dataLabelColor: C.text, dataLabelPosition: "outEnd",
    catAxisLabelFontFace: F, catAxisLabelFontSize: 10, catAxisLabelColor: C.gray, valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 20,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, showTitle: false });
  const st = [["0.84", "AUC: will the pass arrive?", "pass length alone: 0.62", C.teal, C.tealT], ["99%", "ball owner from the events", "the old nearest-player guess was the real passer", C.green, C.greenT],
    ["0.87 vs 0.83", "AUC: pass leads to a shot", "location plus gain against location alone", C.orange, C.orangeT]];
  st.forEach(([big, lab, sub, col, tint], i) => {
    const x = 0.65 + i * 4.07;
    box(s, x, 5.3, 3.88, 1.45, tint);
    txt(s, big, x + 0.22, 5.38, 3.5, 0.55, { size: 25, bold: true, color: col, valign: "middle" });
    txt(s, lab, x + 0.22, 5.97, 3.5, 0.26, { size: 12.5, bold: true });
    txt(s, sub, x + 0.22, 6.25, 3.5, 0.42, { size: 10.5, color: C.gray });
  });
  s.addNotes("Evidence that the BO graph carries real information. Left: with the real passer taken from the match events, the weight ranks the real receiver first in 45% of passes, against 10% for chance and 27% for the nearest teammate; the learned numbers add 6 points over the hand-set ones. Right: passes with a high expected gain are followed by a shot within 10 seconds in 16% of cases, against 0.3% for the lowest quarter. All values are measured on matches that the model never saw (leave-one-match-out). The 'gain' is the chance the pass arrives times the danger of the place it reaches.");
}

// ---------------------------------------------------------------- 6 integration
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  frame(s, 6, "04 · INTEGRATION", "The BO graph joins the radius graph as a fourth edge type");
  box(s, 0.65, 1.58, 6.6, 0.5, C.light, { border: C.line });
  txt(s, "Key State: 23 slots (22 players + ball), position and velocity", 0.85, 1.58, 6.3, 0.5, { size: 12.5, bold: true, valign: "middle" });
  const ed = [["Team edges", "5 nearest mates", C.green, C.greenT], ["Opponent edges", "within 8 m", C.blue, C.blueT], ["Ball edges", "3 nearest players", C.teal, C.tealT], ["Pass edges · new", "owner to teammates", C.orange, C.orangeT]];
  ed.forEach(([h, b, col, tint], i) => {
    const x = 0.65 + i * 1.7;
    arrow(s, x + 0.75, 2.08, x + 0.75, 2.36, col);
    box(s, x, 2.36, 1.5, 0.95, tint);
    txt(s, h, x + 0.08, 2.44, 1.34, 0.4, { size: 10.5, bold: true, color: col, align: "center", valign: "middle" });
    txt(s, b, x + 0.08, 2.86, 1.34, 0.38, { size: 9.75, color: C.text, align: "center", valign: "top" });
    arrow(s, x + 0.75, 3.31, x + 0.75, 3.56, col);
  });
  box(s, 0.65, 3.56, 6.6, 0.55, C.teal);
  txt(s, "Edge-aware message passing · one rule per edge type", 0.85, 3.56, 6.3, 0.55, { size: 12.5, bold: true, color: C.white, valign: "middle" });
  arrow(s, 3.95, 4.11, 3.95, 4.36, C.teal);
  box(s, 0.65, 4.36, 6.6, 0.55, C.dark);
  txt(s, "GRU over the last 3 Key States, with the time gaps", 0.85, 4.36, 6.3, 0.55, { size: 12.5, bold: true, color: C.white, valign: "middle" });
  const out = [["Movement", "where players and ball go next"], ["Embeddings", "tactical states, novelty"], ["Pass analysis", "gain, offside cost"]];
  out.forEach(([h, b], i) => {
    const x = 0.65 + i * 2.25;
    arrow(s, x + 1.0, 4.91, x + 1.0, 5.16, C.teal);
    box(s, x, 5.16, 2.1, 0.78, C.light, { border: C.line });
    txt(s, h, x + 0.1, 5.2, 1.9, 0.28, { size: 11.5, bold: true, color: C.dark, align: "center" });
    txt(s, b, x + 0.1, 5.5, 1.9, 0.4, { size: 9.75, color: C.gray, align: "center" });
  });

  box(s, 7.65, 1.58, 5.02, 4.36, C.blueT);
  txt(s, "PLAYER ERROR ON HELD-OUT REAL MATCHES", 7.87, 1.7, 4.6, 0.25, { size: 11, bold: true, color: C.blue });
  const pe = S.player;
  const mn = (v) => v.toFixed(3).replace("-", "−");
  const rows = [["No graph (3 matches)", "1.21 m", C.gray], ["Radius graph", pe.ours.toFixed(3) + " m", C.teal], ["+ pass edges, real numbers", pe.pass_raw.toFixed(3) + " m", C.green],
    ["+ pass edges, shuffled control", pe.pass_shuffle.toFixed(3) + " m", C.orange]];
  rows.forEach(([lab, val, col], i) => {
    const y = 2.05 + i * 0.5;
    txt(s, lab, 7.87, y, 3.2, 0.4, { size: 12.5, bold: i === 2, valign: "middle" });
    txt(s, val, 11.0, y, 1.45, 0.4, { size: 13.5, bold: true, color: col, align: "right", valign: "middle" });
    s.addShape(pres.shapes.LINE, { x: 7.87, y: y + 0.44, w: 4.58, h: 0, line: { color: C.line, width: 0.75 } });
  });
  const pd = (k) => S.paired.find((r) => r.metric === "player error (m)" && r.comparison.startsWith(k));
  const a = pd("B pass edges minus A"), c = pd("B pass edges minus C");
  txt(s, [{ text: "Interim, run still going: ", options: { bold: true, color: C.orange } },
    { text: `pass edges against the radius graph ${mn(a.mean_diff)} m (interval ${mn(a.ci_low)} to ${mn(a.ci_high)}), better in ${a.better_in_matches} matches. Against the shuffled control ${mn(c.mean_diff)} m (${mn(c.ci_low)} to ${mn(c.ci_high)}), ${c.better_in_matches} matches: the interval only just excludes zero.` }],
    7.87, 4.15, 4.6, 1.2, { size: 11.5 });
  txt(s, `7 held-out matches, ${S.done} of ${S.total} runs finished. For the ball the gain is shared with the control.`, 7.87, 5.4, 4.6, 0.45, { size: 10, color: C.gray });

  box(s, 0.65, 6.15, 12.02, 0.8, C.navy);
  txt(s, "WHY IT IS USEFUL", 0.9, 6.22, 3, 0.22, { size: 10.5, bold: true, color: C.pale });
  txt(s, "Explains each pass option  ·  keeps offside information  ·  compares real and simulated passing", 0.9, 6.5, 11.5, 0.38, { size: 13, bold: true, color: C.white, valign: "middle" });
  s.addNotes("Integration: the radius network already treats each edge type separately, so the BO graph is added as a fourth edge type between the ball owner and each teammate, in both directions, with the pass numbers on the edge. Control: the same edges with shuffled numbers tell us whether the gain comes from pass information or only from a bigger network. Interim result with 7 matches and 2 of 3 seeds: for players, pass edges beat both the radius graph and the shuffled control, with an interval that excludes zero but only just; for the ball the shuffled control gets most of the gain, so that part is mostly extra capacity. We will report the final numbers when the run ends.");
}

// ---------------------------------------------------------------- 7 datasets
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  frame(s, 7, "05 · NEXT STEPS · DATA", "Next: add real datasets to go from 7 matches to about 70");
  const ds = [
    ["PFF FC · WORLD CUP 2022", "64", "matches", "Tracking and events, one file per match; described as free. Licence and access still to confirm.", "FIRST CHOICE", C.green, C.greenT],
    ["SKILLCORNER OPEN DATA", "10", "A-League matches", "Broadcast tracking: not every frame shows all 22 players, so we use it as a robustness test.", "SECOND", C.blue, C.blueT],
    ["STATSBOMB OPEN DATA", "events", "+ 360 freeze frames", "Positions around events only, no continuous tracking: useful for pass analysis, not for movement.", "OPTIONAL", C.orange, C.orangeT],
  ];
  ds.forEach(([name, big, unit, body, tag, col, tint], i) => {
    const x = 0.65 + i * 4.07;
    box(s, x, 1.62, 3.88, 3.45, tint);
    txt(s, name, x + 0.22, 1.78, 3.5, 0.27, { size: 11.5, bold: true, color: col });
    txt(s, big, x + 0.22, 2.12, 3.4, 0.7, { size: big.length > 3 ? 30 : 40, bold: true, color: col, valign: "middle" });
    txt(s, unit, x + 0.22, 2.85, 3.4, 0.3, { size: 14, bold: true });
    txt(s, body, x + 0.22, 3.3, 3.45, 1.1, { size: 12 });
    chip(s, tag, x + 0.22, 4.58, 1.75, col);
  });
  box(s, 0.65, 5.3, 12.02, 1.65, C.navy);
  txt(s, "WHAT IT TAKES", 0.9, 5.42, 3, 0.25, { size: 11, bold: true, color: C.pale });
  txt(s, "Converter to the common format  →  add to settings.json  →  run the pipeline  →  train on the 7 DFL matches, test on a new league", 0.9, 5.74, 11.5, 0.7, { size: 16, bold: true, color: C.white });
  txt(s, "7 to about 71 matches: roughly 3 times narrower confidence intervals, and a cross-league test of generalisation. Nothing is downloaded yet.", 0.9, 6.52, 11.5, 0.3, { size: 11.5, color: C.pale });
  s.addNotes("Why more data: all results so far come from 7 real matches of one league, so the confidence intervals are wide. PFF's World Cup 2022 release has 64 matches with tracking and events and is described as free, but licence and access still need checking. SkillCorner has 10 A-League matches of broadcast tracking, which is a good robustness test but not always 22 players. StatsBomb has events plus 360 freeze frames, useful for passes only. Plan: write a converter to our common format, register the matches in settings.json, rerun the pipeline, train on DFL and test on the new league.");
}

// ---------------------------------------------------------------- 8 currently running
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  frame(s, 8, "06 · RUNNING NOW", "Running now: a firmer test that the pass edges really help");
  box(s, 0.65, 1.58, 6.35, 3.95, C.tealT);
  txt(s, "FIRMER PASS-EDGE TEST", 0.9, 1.72, 5.8, 0.25, { size: 11.5, bold: true, color: C.dark });
  txt(s, `${S.done} / ${S.total}`, 0.9, 2.02, 3.0, 0.75, { size: 40, bold: true, color: C.teal, valign: "middle" });
  txt(s, "training runs finished\n7 matches × 3 networks × 3 seeds", 3.5, 2.12, 3.4, 0.7, { size: 12, color: C.text, valign: "middle" });
  const arms = [["A", "Radius graph", C.teal], ["B", "+ pass edges, real numbers", C.green], ["C", "+ pass edges, shuffled numbers (control)", C.orange]];
  arms.forEach(([k, lab, col], i) => {
    const y = 3.0 + i * 0.5;
    box(s, 0.9, y, 0.38, 0.38, col, { r: 0.08 });
    txt(s, k, 0.9, y, 0.38, 0.38, { size: 12, bold: true, color: C.white, align: "center", valign: "middle" });
    txt(s, lab, 1.42, y, 5.4, 0.38, { size: 12.5, valign: "middle" });
  });
  txt(s, [{ text: "Interim result: ", options: { bold: true, color: C.orange } },
    { text: "for players the pass edges beat both A and the control C; for the ball the control gets most of the gain, so it is mostly extra capacity." }], 0.9, 4.6, 5.9, 0.85, { size: 12 });

  box(s, 7.35, 1.58, 5.32, 3.95, C.blueT);
  txt(s, "CURRENT STATUS", 7.6, 1.72, 4.8, 0.25, { size: 11.5, bold: true, color: C.blue });
  const st = [["Key States and sequences", "Implemented", C.green], ["Radius-graph network", "Implemented", C.green], ["BO graph on real passes", "Validated", C.green],
    ["BO edges inside the network", "Testing", C.teal], ["Tactical states", "Weak signal", C.orange], ["New datasets", "Planned", C.gray]];
  st.forEach(([lab, val, col], i) => {
    const y = 2.07 + i * 0.57;
    txt(s, lab, 7.6, y, 2.55, 0.5, { size: 12, bold: true, valign: "middle" });
    txt(s, val, 10.1, y, 2.35, 0.5, { size: 11, bold: true, color: col, align: "right", valign: "middle" });
    s.addShape(pres.shapes.LINE, { x: 7.6, y: y + 0.52, w: 4.85, h: 0, line: { color: C.line, width: 0.75 } });
  });

  txt(s, "NEXT RESEARCH STEPS", 0.65, 5.78, 6, 0.25, { size: 11.5, bold: true, color: C.orange });
  txt(s, "Finish the runs and report the verdict  ·  richer tactical context (possession, pressure, shape)  ·  add PFF World Cup data", 0.65, 6.12, 12.02, 0.7, { size: 14, bold: true });
  s.addNotes(`Status as of ${S.asof}: ${S.done} of ${S.total} runs finished. The three networks are compared on the same 7 held-out matches with several random starts; the shuffled control separates pass information from extra capacity. The first version of the tactics layer shows that the four tactical states add nothing to the ball position when predicting shots, so the next step is a richer, interpretable context. New datasets come after the run.`);
}

pres.writeFile({ fileName: D + "/ESILV_Key_States_and_Graphs_8_Slides.pptx" }).then((f) => console.log("saved", f));
