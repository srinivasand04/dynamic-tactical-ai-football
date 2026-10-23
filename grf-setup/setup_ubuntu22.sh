#!/usr/bin/env bash
# Google Research Football - setup for Ubuntu 22.04 (native, WSL2, or a university/cloud Linux machine).
# Ubuntu 22.04 matters: its Python 3.10 matches its Boost.Python (1.74, python310), which the engine links against.
set -e

sudo apt-get update
sudo apt-get install -y git cmake build-essential libgl1-mesa-dev libsdl2-dev libsdl2-image-dev \
  libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev libdirectfb-dev libst-dev mesa-utils xvfb x11vnc \
  python3.10 python3.10-venv python3.10-dev

python3.10 -m venv ~/grf-env
source ~/grf-env/bin/activate

# GRF pins gym<=0.21.0, and gym 0.21 only installs with these older build tools.
python -m pip install "pip<24.1" "setuptools==65.5.0" "wheel<0.40" psutil six
python -m pip install --no-build-isolation gym==0.21.0

# The repository was archived in Aug 2026 - pin the last commit so the setup never drifts.
git clone https://github.com/google-research/football.git ~/football
cd ~/football
git checkout 3d9e754720a95621bba6475c4d3b0d56fe919014
python -m pip install --no-build-isolation .     # compiles the C++ engine, ~3-5 minutes

cd -
python smoke_test.py
