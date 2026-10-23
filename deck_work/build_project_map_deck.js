// 10-slide project map in the ESILV style: every important file, where it lives, what it did, with quantified results.
const SCRATCH = "C:/Users/Srinivasan/AppData/Local/Temp/claude/C--Users-Srinivasan-grf-prototype/31840941-a7c1-49f0-9a98-770a29f36e04/scratchpad/node_modules/pptxgenjs";
const pptxgen = require(SCRATCH);
const fs = require("fs");
const path = require("path");
const D = __dirname.replace(/\\/g, "/");
const S = JSON.parse(fs.readFileSync(path.join(D, "stats.json"), "utf8"));

const C = { text: "26333D", gray: "63717B", dark: "0D5864", green: "27836F", blue: "397CA0", teal: "147D8A", orange: "D9822B", navy: "15324A",
  line: "D7E0E4", light: "F4F7F8", greenT: "E8F3EF", blueT: "EAF2F7", tealT: "E7F3F4", orangeT: "FBF1E5", white: "FFFFFF", pale: "8DD3D6" };
const F = "Arial";
const N = 10;
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Dynamic Tactical AI: project map";
pres.author = "Srinivasan Dhakshanamoorthy";
pres.company = "ESILV / DVRC";

function txt(s, text, x, y, w, h, o = {}) {
  s.addText(text, { x, y, w, h, fontFace: o.font || F, fontSize: o.size || 13.5, bold: !!o.bold, color: o.color || C.text, align: o.align || "left",
    valign: o.valign || "top", margin: 0, isTextBox: true, fit: "none", paraSpaceAfter: o.after || 0 });
}
function box(s, x, y, w, h, fill, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, rectRadius: o.r === undefined ? 0.1 : o.r, line: o.border ? { color: o.border, width: 0.75 } : { color: fill, width: 0 } });
}
function arrow(s, x1, y1, x2, y2, color = C.teal) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipV: y2 < y1, flipH: x2 < x1, line: { color, width: 1.5, endArrowType: "triangle" } });
}
function frame(s, n, section, title) {
  txt(s, section, 0.65, 0.29, 9.9, 0.26, { size: 11.25, bold: true, color: C.gray, valign: "middle" });
  txt(s, title, 0.65, 0.64, 11.0, 0.6, { size: 25.5, bold: true, color: C.dark, valign: "middle" });
  s.addImage({ path: D + "/logo.png", x: 11.72, y: 0.24, w: 1.08, h: 0.43, altText: "ESILV logo" });
  s.addShape(pres.shapes.LINE, { x: 0.65, y: 1.36, w: 12.02, h: 0, line: { color: C.line, width: 0.75 } });
  s.addShape(pres.shapes.LINE, { x: 0.65, y: 7.12, w: 12.02, h: 0, line: { color: C.line, width: 0.75 } });
  txt(s, "DYNAMIC TACTICAL AI  ·  DVRC / ESILV", 0.65, 7.19, 7.29, 0.19, { size: 8.25, color: C.gray, valign: "middle" });
  txt(s, `${n} / ${N}`, 12.1, 7.19, 0.57, 0.19, { size: 8.25, bold: true, color: C.gray, align: "right", valign: "middle" });
}
const bottom = [{ type: "none" }, { type: "none" }, { type: "solid", color: C.line, pt: 0.75 }, { type: "none" }];
function cell(text, o = {}) {
  return { text, options: { fontFace: o.font || F, fontSize: o.size || 10.5, bold: !!o.bold, color: o.color || C.text, valign: "middle", border: bottom, margin: [0.04, 0.08, 0.04, 0.08], fill: o.fill ? { color: o.fill } : undefined } };
}
// rows: [file(s), what it did, saved in]
function fileTable(s, rows, y, headFirst = "FILE (scripts/)", colW = [2.75, 5.0, 4.27], rowH = 0.56) {
  const head = [headFirst, "WHAT IT DID", "OUTPUT SAVED IN"].map((h) => cell(h, { size: 10, bold: true, color: C.dark, fill: C.tealT }));
  const body = rows.map(([f, w, o]) => [cell(f, { bold: true, color: C.dark }), cell(w), cell(o, { color: C.gray })]);
  s.addTable([head, ...body], { x: 0.65, y, w: colW.reduce((a, b) => a + b, 0), colW, rowH: [0.32, ...rows.map(() => rowH)], fontFace: F });
}
function stat(s, big, lab, sub, x, y, w, h, col, tint) {
  box(s, x, y, w, h, tint);
  txt(s, big, x + 0.2, y + 0.08, w - 0.4, 0.5, { size: 22, bold: true, color: col, valign: "middle" });
  txt(s, lab, x + 0.2, y + 0.6, w - 0.4, 0.26, { size: 11.5, bold: true });
  if (sub) txt(s, sub, x + 0.2, y + 0.86, w - 0.4, h - 0.9, { size: 10, color: C.gray });
}

// ---------------------------------------------------------------- 1 title
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.19, h: 7.5, fill: { color: C.teal }, line: { color: C.teal, width: 0 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.19, y: 0, w: 0.04, h: 7.5, fill: { color: C.orange }, line: { color: C.orange, width: 0 } });
  s.addImage({ path: D + "/logo.png", x: 10.94, y: 0.43, w: 1.77, h: 0.71, altText: "ESILV logo" });
  txt(s, "PROJECT DOCUMENTATION  ·  OCTOBER 2026", 0.81, 0.97, 7.29, 0.31, { size: 12.75, bold: true, color: C.orange, valign: "middle" });
  txt(s, [{ text: "Dynamic", options: { breakLine: true } }, { text: "Tactical AI" }], 0.81, 1.67, 7.66, 1.56, { size: 43.5, bold: true, color: C.navy, valign: "middle" });
  txt(s, "Project map: every important file, what it does and where its output is saved", 0.85, 3.49, 7.6, 0.95, { size: 22.5, bold: true, color: C.dark });
  s.addShape(pres.shapes.LINE, { x: 0.85, y: 4.7, w: 7.71, h: 0, line: { color: C.line, width: 0.75 } });
  txt(s, "Folders  ·  Scripts  ·  Outputs  ·  Quantified results  ·  GitHub", 0.85, 4.98, 8.4, 0.6, { size: 15, color: C.gray });
  txt(s, "Srinivasan Dhakshanamoorthy", 0.85, 6.19, 4.79, 0.31, { size: 14.25, bold: true, valign: "middle" });
  txt(s, "MSc Data Engineering & AI  |  DVRC / ESILV  |  Supervisor: Prof. Ahmed Azough", 0.85, 6.52, 8.12, 0.26, { size: 11.25, color: C.gray, valign: "middle" });
  s.addImage({ path: D + "/title_graph.png", x: 9.01, y: 2.42, w: 3.54, h: 2.24, altText: "Player graph on a football pitch" });
  box(s, 8.75, 5.06, 4.1, 0.32, C.tealT, { r: 0.08 });
  txt(s, "46 SCRIPTS  ·  7 REAL MATCHES  ·  36 SIMULATED GAMES", 8.85, 5.06, 3.9, 0.32, { size: 10, bold: true, color: C.dark, valign: "middle" });
  s.addNotes("This deck is the map of the repository: for every stage, the scripts, what each one did, the files it wrote and the numbers it produced. The last slide shows how the code is published to GitHub.");
}

// ---------------------------------------------------------------- 2 layout
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  frame(s, 2, "01 · PROJECT LAYOUT", "The project in one picture: settings, scripts, data, outputs");
  const rows = [
    ["settings.json", "the one place for every number and every match list; scripts read it", "2 KB"],
    ["scripts/", "all code, 46 Python files, run from the project root", "0.4 MB"],
    ["data/real/raw/", "downloaded DFL files per match: tracking, events, players, meta", "198 MB · 28 files"],
    ["data/real/raw_xml/", "the original XML files (resumable download)", "1.09 GB · 9 files"],
    ["data/real/processed/", "common-format CSV and every file derived from it, per match", "189 MB · 145 files"],
    ["data/simulated/", "36 Google Football games in four settings, same format", "208 MB · 475 files"],
    ["outputs/", "pictures, tables, model weights, the run log (80 experiments)", "7.7 MB · 75 files"],
    ["docs/ · grf-setup/", "presentations · setup notes for the simulator", "small"],
  ];
  const head = ["PATH", "WHAT IS INSIDE", "SIZE"].map((h) => cell(h, { size: 10, bold: true, color: C.dark, fill: C.tealT }));
  const body = rows.map(([p, w, z]) => [cell(p, { bold: true, color: C.dark }), cell(w), cell(z, { color: C.gray })]);
  s.addTable([head, ...body], { x: 0.65, y: 1.6, w: 8.5, colW: [2.2, 4.4, 1.9], rowH: [0.32, ...rows.map(() => 0.52)], fontFace: F });
  stat(s, "7", "real DFL matches", "401,060 frames · 668 minutes at 10 frames/s", 9.45, 1.6, 3.22, 1.35, C.green, C.greenT);
  stat(s, "36", "simulated games", "stochastic 12 · easy 8 · hard 8 · competition 8", 9.45, 3.1, 3.22, 1.35, C.blue, C.blueT);
  stat(s, "1.7 GB", "data stays off GitHub", "the scripts rebuild it: run_pipeline.py", 9.45, 4.6, 3.22, 1.35, C.orange, C.orangeT);
  txt(s, "Per match, the processed folder holds about 20 files named <match>_<what>.npz or .csv (for example J03WMX_graphs.npz); outputs/ holds 24 pictures, 22 tables, 18 text logs, 8 model checkpoints, 2 json files and 1 html overview.", 0.65, 6.25, 12.02, 0.6, { size: 11.5, color: C.gray });
  s.addNotes("Everything is driven by settings.json. Real data lives under data/real, simulated games under data/simulated. All derived files are named after the match id. Outputs holds everything we show in slides.");
}

// ---------------------------------------------------------------- 3 data
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  frame(s, 3, "02 · DATA AND PIPELINE", "Stage 1: raw tracking becomes one common format");
  fileTable(s, [
    ["download_real.py\nsave_meta.py", "download one DFL match (tracking and events) in a resumable way; save teams, players, pitch size, direction, score", "data/real/raw/<match>_tracking.csv.gz, _events.csv.gz, _players.csv, _meta.json"],
    ["convert_real.py", "25 to 10 frames/s, metres, home always attacks to x = 105, one row per player per frame, red cards allowed", "data/real/processed/<match>.csv.gz"],
    ["register_matches.py", "one row per real match, keyed by the DFL match id", "data/real/matches.csv"],
    ["export_tracking.py", "run one Google Football game and export it in the same layout (in grf-setup/)", "data/simulated/<setting>/<game>.csv"],
    ["run_pipeline.py", "runs download, convert, Key States, sequences, descriptions, types and graphs; skips what exists", "all files of stages 1 to 3"],
    ["settings.py · runlog.py", "read settings.json; append every experiment to the run log", "outputs/run_log.csv (80 rows)"],
  ], 1.6);
  stat(s, "401,060", "real frames", "7 matches, 668 minutes, 10 frames/s", 0.65, 5.75, 3.88, 1.2, C.green, C.greenT);
  stat(s, "36 × 3,000", "simulated frames", "5 minutes per game, four settings", 4.72, 5.75, 3.88, 1.2, C.blue, C.blueT);
  stat(s, "105 × 68 m", "one common format", "metres, 23 slots: 22 players + ball", 8.79, 5.75, 3.88, 1.2, C.orange, C.orangeT);
  s.addNotes("The pipeline script runs every data step and skips what already exists, so adding a match to settings.json is enough to bring it through all stages.");
}

// ---------------------------------------------------------------- 4 key states
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  frame(s, 4, "03 · KEY STATES AND SEQUENCES", "Stage 2: Key States and sequences, with their output files");
  fileTable(s, [
    ["extract_keystates.py\nkeystates.py", "keep a frame when 0.4 × player + 0.4 × shape + 0.2 × ball movement reaches 3 m (keystates.py: first tiny test)", "data/real/processed/<match>_adaptive_ks.csv"],
    ["find_sequences.py", "boundaries between sequences from geometry only (5 s window, one scale for all matches)", "<match>_sequences.csv, <match>_ks_features.csv"],
    ["describe_sequences.py", "describe each boundary with real events (shot, foul, ball out) and compare with random boundaries", "<match>_boundary_events.csv, <match>_sequence_desc.csv"],
    ["cluster_sequences.py", "sequence types by k-means; keep only the k that is stable", "<match>_sequence_types.csv, outputs/sequence_types_8matches.png"],
    ["plot_timeline.py\nplot_state_share.py", "Key State timeline and share per state, real against simulated", "outputs/key_state_timeline.png, key_state_share.png"],
  ], 1.6, "FILE (scripts/)", [2.75, 5.0, 4.27], 0.6);
  stat(s, "23,852", "Key States in 7 real matches", "about 3,400 per match, 5.9% of frames (simulator: 358 per game)", 0.65, 5.2, 3.88, 1.6, C.teal, C.tealT);
  stat(s, "2 types", "open play and stoppage", "stable across matches: agreement 0.91, and 0.99 with a match left out", 4.72, 5.2, 3.88, 1.6, C.green, C.greenT);
  stat(s, "31% vs 0.4%", "stoppage scenes", "real matches against the simulator, the largest structural gap", 8.79, 5.2, 3.88, 1.6, C.orange, C.orangeT);
  s.addNotes("Key States are defined by geometry only; events describe them afterwards. 23,852 Key States in total over the 7 real matches.");
}

// ---------------------------------------------------------------- 5 graphs
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  frame(s, 5, "04 · GRAPHS AND NETWORK", "Stage 3: one graph per Key State, one network over time");
  fileTable(s, [
    ["build_graphs.py", "one graph per Key State: 23 nodes; team (5 nearest), opponent (within 8 m), ball (3 nearest) edges; time features", "<match>_graphs.npz, <match>_graphs_delaunay.npz"],
    ["train_gnn.py", "edge-type-aware message passing + GRU over 3 Key States; predicts the next movement; archs ours, gcn, gat, mlp, pass_*", "outputs/gnn_model.pt, <match>_embeddings.npz, run_log.csv"],
    ["run_folds.py", "hold each real match out in turn and train on the other six", "outputs/gnn_model_fold_<match>.pt (7), embeddings per fold"],
    ["compare_archs.py\nsummarise_gnn.py", "one table comparing graph layers on the same held-out matches", "outputs/compare_archs.csv"],
    ["plot_graph.py", "draw a graph on the pitch and four consecutive Key States", "outputs/graph_snapshot.png, graph_time.png"],
  ], 1.6, "FILE (scripts/)", [2.75, 5.0, 4.27], 0.6);
  stat(s, "1.11 m", "player error, unseen match", "keep-velocity 1.51 m: 26% lower. Ball 3.29 m against 6.19 m", 0.65, 5.2, 3.88, 1.6, C.teal, C.tealT);
  stat(s, "1.21 m", "no graph (control)", "7 points less gain than with the graph (20% against 27%)", 4.72, 5.2, 3.88, 1.6, C.gray, C.light);
  stat(s, "1.11 m", "GCN and GAT", "equal to our layer: the exact graph type matters little", 8.79, 5.2, 3.88, 1.6, C.blue, C.blueT);
  s.addNotes("Results are leave-one-match-out: the error is measured on a real match the network never saw. The no-graph and GCN/GAT numbers use the 3 matches shared by all runs.");
}

// ---------------------------------------------------------------- 6 BO graph
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  frame(s, 6, "05 · BALL-OWNER GRAPH", "Stage 4: the ball-owner graph and its real-pass validation");
  fileTable(s, [
    ["bo_graph.py", "the BO graph: weight from free space, pass length, goal closeness and open lane; offside weight × −1", "library used by the scripts below"],
    ["extract_passes.py", "real passer and receiver from the events; a snapshot of the pitch at the moment of each pass", "data/real/processed/<match>_passes.npz"],
    ["receiver_check.py", "does the weight rank the real receiver first? hand-set against learned numbers", "outputs/receiver_check.csv, receiver_check.png"],
    ["fit_bo_weights.py", "learn the five distances and the importance of each ingredient from real passes", "outputs/bo_params_learned.json, bo_fit.csv, bo_fit.png"],
    ["plot_bo_graph.py", "draw BO graphs on the pitch, real and simulated, with blockers and offside", "outputs/bo_graph*.png, bo_graph_edges.csv"],
  ], 1.6, "FILE (scripts/)", [2.75, 5.0, 4.27], 0.6);
  stat(s, "6,062 → 3,987", "pass events to usable passes", "4,838 matched to tracking; 3,987 in open play with a receiver (3,546 completed)", 0.65, 5.2, 3.88, 1.6, C.teal, C.tealT);
  stat(s, "99%", "ball owner from events", "the old nearest-player guess was the real passer in 99% of passes", 4.72, 5.2, 3.88, 1.6, C.green, C.greenT);
  stat(s, "35% → 41.5%", "real receiver ranked first", "hand-set against learned numbers; chance 10%, nearest teammate 27%", 8.79, 5.2, 3.88, 1.6, C.orange, C.orangeT);
  s.addNotes("The whiteboard specification is implemented in bo_graph.py. The real passer and receiver come from the match events, so the receiver check is a real test: the weight is only credited when the actual receiver is the top-ranked teammate.");
}

// ---------------------------------------------------------------- 7 pass models
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  frame(s, 7, "06 · PASS MODELS", "Stage 5: pass models, expected gain and simulator comparison");
  fileTable(s, [
    ["pass_features.py", "movement and timing numbers for every pass option: arrival point, lane time margin, running speed", "<match>_passfeat.npz (real and simulated)"],
    ["fit_pass_model.py", "three models: who is picked, will the pass arrive, how dangerous is the place (leave-one-match-out)", "outputs/pass_model.json, pass_model.csv, pass_model_completion.csv"],
    ["build_pass_edges.py", "pass edges for every Key State, as network input", "<match>_passedges.npz"],
    ["decision_quality.py\nspeed_check.py", "real against simulated passing decisions; is the gap only the simulator's speed?", "outputs/decision_quality.csv, speed_check.csv"],
    ["validate_gain.py", "does the expected gain predict a shot in the next 10 s?", "outputs/validate_gain.csv, validate_gain_matches.csv"],
  ], 1.6, "FILE (scripts/)", [2.75, 5.0, 4.27], 0.6);
  stat(s, "45%", "real receiver ranked first", "still picture 42%, with movement 45%; AUC for 'will it arrive' 0.84", 0.65, 5.2, 3.88, 1.6, C.teal, C.tealT);
  stat(s, "0.3% → 16.3%", "shot within 10 s", "lowest to highest quarter of the pass gain", 4.72, 5.2, 3.88, 1.6, C.orange, C.orangeT);
  stat(s, "17.7 vs 12 m", "pass length, real vs simulated", "not a speed effect: slowing the simulator changed the gap from 16.7% to 15.5%", 8.79, 5.2, 3.88, 1.6, C.blue, C.blueT);
  s.addNotes("Three small models, all judged on matches they never saw. The simulator passes shorter and more forward than real players, and that does not disappear when the simulated players are slowed to real speed.");
}

// ---------------------------------------------------------------- 8 evidence
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  frame(s, 8, "07 · EVIDENCE AND TACTICS", "Evidence and tactics: scoreboard, realism and the proof run");
  fileTable(s, [
    ["scoreboard.py", "every quantifier in one table with 95% intervals over held-out matches", "outputs/scoreboard.csv, scoreboard.png"],
    ["realism_scorecard.py", "gap of each simulator setting to real football on nine measures", "outputs/realism_scorecard.csv, realism_scorecard.png"],
    ["run_pass_proof.py\nprove_pass_edges.py", "three networks × 7 matches × seeds, paired statistics: do the pass edges help?", "outputs/prove_pass_edges.csv, prove_pass_edges.png"],
    ["tactical_states.py\ntactics_profile.py", "four tactical states; playbook, team style, unused best option, validity", "<match>_tactical_state.csv, outputs/tactics_*.csv, tactics_profile.png"],
    ["novelty_holdout.py\nlearned_distance.py", "how unusual a match is; learned against hand-made Key State distance", "outputs/novelty_holdout.csv, learned_distance.csv, combined_distance.csv"],
  ], 1.6, "FILE (scripts/)", [2.75, 5.0, 4.27], 0.6);
  const pa = S.paired.find((r) => r.metric === "player error (m)" && r.comparison.startsWith("B pass edges minus A"));
  stat(s, "64–68%", "simulator gap to real", "passing gaps 13–50%, structural gaps 90–100% (nine measures)", 0.65, 5.2, 3.88, 1.6, C.orange, C.orangeT);
  stat(s, `${pa.mean_diff.toFixed(3).replace("-", "−")} m`, "pass edges, interim", `against the radius network, better in ${pa.better_in_matches} matches (${S.done} of ${S.total} runs done)`, 4.72, 5.2, 3.88, 1.6, C.teal, C.tealT);
  stat(s, "−0.003", "tactical state, AUC", "adds nothing beyond the ball's place for 'shot within 10 s'", 8.79, 5.2, 3.88, 1.6, C.blue, C.blueT);
  s.addNotes("The scoreboard is the one-page answer to 'how do we know we are doing well'. The proof run is still finishing; the tactics layer is a first version and its states carry little information beyond the ball position.");
}

// ---------------------------------------------------------------- 9 run order
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  frame(s, 9, "08 · RUN ORDER AND OUTPUTS", "Run order: six steps from raw data to the scoreboard");
  const steps = [
    ["1 · DATA", "run_pipeline.py", "writes data/real/processed and graphs", C.green, C.greenT],
    ["2 · NETWORK", "run_folds.py", "writes gnn_model_fold_*.pt and embeddings", C.green, C.greenT],
    ["3 · PASSES", "extract_passes.py\npass_features.py", "write _passes.npz, _passfeat.npz", C.teal, C.tealT],
    ["4 · MODELS", "fit_pass_model.py\nbuild_pass_edges.py", "write pass_model.json, _passedges.npz", C.teal, C.tealT],
    ["5 · CHECKS", "decision_quality.py\nvalidate_gain.py\ntactics_profile.py", "write the CSV tables", C.blue, C.blueT],
    ["6 · EVIDENCE", "scoreboard.py\nprove_pass_edges.py", "write scoreboard and proof tables", C.orange, C.orangeT],
  ];
  steps.forEach(([h, cmd, out, col, tint], i) => {
    const x = 0.65 + i * 2.03;
    box(s, x, 1.75, 1.85, 2.7, tint);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.75, w: 1.85, h: 0.5, fill: { color: col }, rectRadius: 0.1, line: { color: col, width: 0 } });
    txt(s, h, x, 1.75, 1.85, 0.5, { size: 11.5, bold: true, color: C.white, align: "center", valign: "middle" });
    txt(s, cmd, x + 0.1, 2.4, 1.65, 1.4, { size: 10.5, bold: true, color: C.text, align: "center", valign: "top" });
    txt(s, out, x + 0.1, 3.75, 1.65, 0.65, { size: 10, color: C.gray, align: "center", valign: "top" });
    if (i < 5) arrow(s, x + 1.87, 3.1, x + 2.01, 3.1, C.teal);
  });
  box(s, 0.65, 4.75, 12.02, 2.15, C.navy);
  txt(s, "EVERY RUN IS LOGGED", 0.9, 4.9, 4, 0.25, { size: 11, bold: true, color: C.pale });
  txt(s, "outputs/run_log.csv: 80 experiment rows with parameters, errors and minutes", 0.9, 5.25, 11.5, 0.35, { size: 15, bold: true, color: C.white });
  txt(s, "outputs/ holds 75 files: 24 pictures, 22 tables, 18 text logs, 8 model checkpoints, 2 json files and 1 html overview. Each script prints its table and names the files it saved; commands are in the README.", 0.9, 5.85, 11.5, 0.9, { size: 12.5, color: C.pale });
  s.addNotes("The order matters: the pass scripts need the Key States and passes first, and the scoreboard reads the tables written by the earlier steps. Each script skips nothing silently: it prints what it saved.");
}

// ---------------------------------------------------------------- 10 github
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  frame(s, 10, "09 · GITHUB", "Publishing: code and results go to GitHub, raw data stays out");
  box(s, 0.65, 1.58, 6.05, 1.15, C.tealT);
  txt(s, "REPOSITORY NAME", 0.9, 1.7, 5.5, 0.23, { size: 10.5, bold: true, color: C.dark });
  txt(s, "dynamic-tactical-ai-football", 0.9, 2.0, 5.6, 0.45, { size: 24, bold: true, color: C.dark, font: "Courier New", valign: "middle" });
  txt(s, "private by default; public with -Visibility public", 0.9, 2.45, 5.6, 0.25, { size: 10.5, color: C.gray });
  box(s, 0.65, 2.95, 2.95, 3.0, C.greenT);
  txt(s, "GOES IN · ABOUT 140 FILES, 8 MB", 0.85, 3.07, 2.6, 0.4, { size: 10.5, bold: true, color: C.green });
  txt(s, "scripts/ · 46 files\noutputs/ · 75 files\nsettings.json, README\nrequirements files\ndocs/ · presentations\ngrf-setup/", 0.85, 3.5, 2.6, 2.3, { size: 12, after: 4 });
  box(s, 3.75, 2.95, 2.95, 3.0, C.orangeT);
  txt(s, "STAYS OUT", 3.95, 3.07, 2.6, 0.4, { size: 10.5, bold: true, color: C.orange });
  txt(s, "data/ · 1.7 GB, rebuilt by run_pipeline.py\n.venv folders\nold decks, PDFs, logo", 3.95, 3.5, 2.6, 2.3, { size: 12, after: 6 });
  box(s, 6.95, 1.58, 5.72, 4.37, C.navy);
  txt(s, "RUN ONCE", 7.2, 1.72, 5, 0.25, { size: 11, bold: true, color: C.pale });
  txt(s, "gh auth login", 7.2, 2.02, 5.2, 0.35, { size: 14, bold: true, color: C.white, font: "Courier New", valign: "middle" });
  txt(s, "THEN", 7.2, 2.62, 5, 0.25, { size: 11, bold: true, color: C.pale });
  txt(s, "powershell -ExecutionPolicy Bypass -File .\\push_to_github.ps1", 7.2, 2.92, 5.3, 0.7, { size: 12.5, bold: true, color: C.white, font: "Courier New" });
  txt(s, "PUBLIC REPOSITORY INSTEAD", 7.2, 3.85, 5, 0.25, { size: 11, bold: true, color: C.pale });
  txt(s, "... -File .\\push_to_github.ps1 -Visibility public", 7.2, 4.15, 5.3, 0.7, { size: 12.5, bold: true, color: C.white, font: "Courier New" });
  txt(s, "Run it again any time: it commits only what changed and pushes to the same repository.", 7.2, 5.1, 5.2, 0.7, { size: 12, color: C.pale });
  box(s, 0.65, 6.15, 12.02, 0.78, C.light, { border: C.line });
  txt(s, "The script checks that no file is larger than 50 MB before it commits, copies the presentations into docs/, and stops with a clear message if you are not logged in.", 0.9, 6.15, 11.5, 0.78, { size: 12, valign: "middle" });
  s.addNotes("The push script is push_to_github.ps1 in the project root. It creates the repository with the GitHub CLI after you log in once. The data folder is excluded by .gitignore because it is large and fully rebuilt by the pipeline.");
}

pres.writeFile({ fileName: D + "/Project_Files_Map_10_Slides.pptx" }).then((f) => console.log("saved", f));
