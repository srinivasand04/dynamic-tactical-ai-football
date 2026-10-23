# Google Research Football (GRF) — Setup for the Prototype

Covers Phase 1 (environment running) and the start of Phase 2 (telemetry in tracking format) of the staged plan.

## Status to know first

- **The GitHub repository was archived on 19 Aug 2026.** It still works, but there will be no more fixes. Pin a known-good commit and treat the environment as frozen.
- **Last PyPI release: 2.10.2 (Jan 2022)**, with prebuilt wheels for Windows only and Python 3.6–3.10. On Linux the C++ engine compiles from source.
- **Use Python 3.10.** Python 3.12 fails: GRF pins `gym<=0.21.0`, and gym 0.21 can't install on 3.12.

## What was verified (Linux, 25 Sep 2026)

Built from source and tested with Python 3.10, pinned commit `3d9e754`, gym 0.21.0 and numpy 2.2.6:

| Check | Result |
|---|---|
| Engine compile | ✅ ~3.5 min on 2 CPU cores |
| 11 vs 11 headless, raw observations | ✅ 300 steps in 2.1 s (≈ 20 s per full 3000-step match) |
| All 11 left players controlled (multi-agent) | ✅ obs `(11, 115)`, action space `MultiDiscrete([19]*11)` |
| Academy scenario (3 vs 1 with keeper) | ✅ |
| Rendered frames + `.avi` video under a virtual display | ✅ (`xvfb-run`) |
| Export to tracking CSV (metres, 10 fps) | ✅ speeds realistic (median 3.7 m/s, max 7.8 m/s) |

## Recommended routes

| Your machine | Route | Notes |
|---|---|---|
| **Windows** | **WSL2 with Ubuntu 22.04** → `setup_ubuntu22.sh` | Most reliable. Install WSL with `wsl --install -d Ubuntu-22.04`. |
| Linux (Ubuntu 22.04) | `setup_ubuntu22.sh` | Python 3.10 and Boost.Python 3.10 match by default. |
| Any machine with Docker | `docker build -t grf .` | Same steps as the script, frozen in an image. Best for reproducibility. |
| Ubuntu 24.04 | Docker, or see "Advanced" below | 24.04 ships Boost.Python for 3.12 only, which doesn't match Python 3.10. |
| macOS (Apple Silicon) | Docker, or a Linux server | Native builds are fragile. |
| Windows native (quick try) | Python 3.10 64-bit, then the pins below, then `pip install gfootball==2.10.2` | Prebuilt wheel, no compilation. Not tested here. |

The Docker image and the Windows-native route were **not** executed here: Docker Hub was blocked in the test environment. They use the same pins as the verified build.

## The pins that make it work

```bash
python -m pip install "pip<24.1" "setuptools==65.5.0" "wheel<0.40" psutil six
python -m pip install --no-build-isolation gym==0.21.0
python -m pip install --no-build-isolation .      # inside the cloned repo
```

- `pip<24.1` / `setuptools==65.5.0` / `wheel<0.40`: newer versions reject gym 0.21's packaging metadata.
- `six`: required at import time but not declared as a dependency.
- `--no-build-isolation`: keeps these pinned tools during the build.

Full working versions are in `requirements-lock.txt`.

## First run

```bash
source ~/grf-env/bin/activate
python smoke_test.py                     # headless 11v11, multi-agent, academy scenario
python export_tracking.py --steps 3000 --out match_001.csv   # one full match -> tracking CSV
python -m gfootball.play_game --action_set=full   # play it yourself (needs a display)
```

Headless video (e.g. on WSL/servers): `xvfb-run -a python your_script.py` with `render=True, write_video=True`.

## Key facts for the prototype

- **Time:** 1 step = 100 ms (10 fps). A full match is 3000 steps.
- **Coordinates:** x ∈ [-1, 1] goal to goal, y ∈ [-0.42, 0.42]. `export_tracking.py` converts to metres on a 105 × 68 pitch.
- **Representations:** `raw` (positions, directions, roles, ball, possession, game mode), `simple115v2` (a 115-number vector per player), `extracted` (minimap planes), `pixels`.
- **Actions:** 19 discrete actions (8 directions, passes, shot, sprint, dribble, slide, idle…).
- **Who controls whom:** `number_of_left_players_agent_controls=N` gives you N players. Everyone else is played by GRF's built-in AI, which can serve as the ready-made **scripted baseline** for Phase 3.
- **Scenarios:** `11_vs_11_stochastic`, `11_vs_11_easy_stochastic`, `11_vs_11_hard_stochastic`, `1_vs_1_easy`, `5_vs_5`, and the `academy_*` set (small tactical drills). These are useful for the 1v1 → 11v11 curriculum.

## How this maps to the phase plan

| Phase | What this kit gives you |
|---|---|
| 1 — Environment | `setup_ubuntu22.sh` / `Dockerfile` + `smoke_test.py` |
| 2 — Telemetry | `export_tracking.py`: long-format tracking CSV (frame, team, player, role, x/y in metres, velocity, ball, possession, game mode, score). Loading it into kloppy needs a small adapter; pandas works directly. |
| 3 — Scripted baseline | GRF's built-in AI plays all uncontrolled players. Compare its exported speed, spacing and formation statistics against DFL/Sportec. |

## Advanced: building on Ubuntu 24.04 without Docker

This is how it was verified here. Build Boost.Python for Python 3.10 from the Boost 1.83 sources, then point the engine's CMake at Python 3.10. In `gfootball/build_game_engine.sh`, change the `cmake .` call to:

```bash
cmake . -DPython_FIND_STRATEGY=LOCATION \
        -DPython_INCLUDE_DIR=/usr/include/python3.10 \
        -DPython_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so \
        -DBOOST_ROOT=/opt/boostpy310
```

Without these flags, CMake picks the newest Python on the system and fails with "Boost Python not found".

## Sources

- Repository (archived 19 Aug 2026): https://github.com/google-research/football
- PyPI: https://pypi.org/project/gfootball/
- Kurach et al., *Google Research Football: A Novel Reinforcement Learning Environment*, AAAI 2020 (arXiv:1907.11180)
