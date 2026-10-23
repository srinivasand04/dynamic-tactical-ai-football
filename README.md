# Dynamic Tactical AI for football

A five-layer tactics AI built on real football tracking data (DFL / Sportec open data) and the Google Research Football (GRF) simulator.
MSc Data Engineering & AI, DVRC / ESILV. Supervisor: Prof. Ahmed Azough.

| Layer | What | Status |
|---|---|---|
| 1 Data | 7 real DFL matches and 36 simulated GRF games in one common format (105 x 68 m, 10 frames/s) | done |
| 2 Key States | frames where play moved by at least 3 m, sequences and their types | done |
| 3 Graphs | radius graph, ball-owner (BO) pass graph, GNN + GRU over Key States | done, pass-edge proof running |
| 4 Tactics | tactical states, pass decisions, expected gain | first version |
| 5 Agents | actions in the simulator | not started |

## Key results (leave-one-match-out on the 7 real matches)

| Quantity | Result | Baseline |
|---|---|---|
| Player movement error on an unseen match | 1.11 m | 1.51 m (keep velocity) |
| Ball movement error | 3.29 m | 6.19 m |
| Real receiver ranked first by the BO graph | 45% | 10% chance, 27% nearest teammate |
| Will the pass arrive (AUC) | 0.84 | 0.62 (pass length only) |
| Shot within 10 s, lowest to highest pass-gain quarter | 0.3% to 16.3% | |
| Simulator gap to real football (realism scorecard) | 64-68% | 0% = real |

Every result is measured on matches the model never saw, with 95% intervals over matches in `outputs/scoreboard.csv`.

## Folder map

```
settings.json         the one place for every number and every match list
scripts/              all code (46 Python files), run from the project root
data/                 NOT in the repository (about 1.7 GB), rebuilt by scripts/run_pipeline.py
  real/raw/             downloaded DFL files (tracking, events, players, meta)
  real/processed/       common-format CSV and every derived file per match
  simulated/            GRF games in four settings (stochastic, easy, hard, competition)
outputs/              figures, tables, model weights and the run log (outputs/run_log.csv)
docs/                 presentations
grf-setup/            Docker / setup notes for the GRF environment
```

## Scripts, in the order the data flows

| Stage | Scripts | Main outputs |
|---|---|---|
| Data | `download_real.py`, `save_meta.py`, `convert_real.py`, `register_matches.py`, `export_tracking.py`, `run_pipeline.py` | `data/real/processed/<match>.csv.gz`, `data/real/matches.csv` |
| Key States | `extract_keystates.py`, `find_sequences.py`, `describe_sequences.py`, `cluster_sequences.py` | `<match>_adaptive_ks.csv`, `_sequences.csv`, `_sequence_types.csv` |
| Graphs and network | `build_graphs.py`, `train_gnn.py`, `run_folds.py`, `compare_archs.py` | `<match>_graphs.npz`, `outputs/gnn_model*.pt`, `outputs/compare_archs.csv` |
| Ball-owner graph | `bo_graph.py`, `extract_passes.py`, `receiver_check.py`, `fit_bo_weights.py`, `plot_bo_graph.py` | `<match>_passes.npz`, `outputs/bo_params_learned.json`, `outputs/receiver_check.csv` |
| Pass models | `pass_features.py`, `fit_pass_model.py`, `build_pass_edges.py`, `decision_quality.py`, `validate_gain.py`, `speed_check.py` | `<match>_passfeat.npz`, `<match>_passedges.npz`, `outputs/pass_model.json` |
| Evidence | `scoreboard.py`, `realism_scorecard.py`, `run_pass_proof.py`, `prove_pass_edges.py` | `outputs/scoreboard.csv`, `outputs/realism_scorecard.csv`, `outputs/prove_pass_edges.csv` |
| Tactics | `tactical_states.py`, `tactics_profile.py` | `outputs/tactics_*.csv` |

## Reproduce

```bash
python -m venv .venv-analysis
.venv-analysis\Scripts\pip install -r requirements-analysis.txt      # Windows; use bin/ on Linux
.venv-analysis\Scripts\python scripts\run_pipeline.py                # downloads and converts the real matches, builds Key States and graphs
.venv-analysis\Scripts\python scripts\run_folds.py                   # leave-one-match-out network training
.venv-analysis\Scripts\python scripts\extract_passes.py
.venv-analysis\Scripts\python scripts\pass_features.py
.venv-analysis\Scripts\python scripts\fit_pass_model.py
.venv-analysis\Scripts\python scripts\scoreboard.py
```

The simulated games need the GRF environment (Python 3.10, see `grf-setup/README.md` and `requirements-grf.txt`).

## Data

The real match data is the DFL / Sportec open dataset (IDSSE). It is not redistributed here: `scripts/download_real.py` fetches it
under the dataset's own licence. Simulated games come from Google Research Football.
