# Reproducible GRF environment. Build:  docker build -t grf .
# Run headless:  docker run --rm -v "$PWD":/work grf python /work/smoke_test.py
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git cmake build-essential libgl1-mesa-dev libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev libdirectfb-dev libst-dev \
    mesa-utils xvfb x11vnc python3.10 python3.10-venv python3.10-dev && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m venv /opt/grf
ENV PATH="/opt/grf/bin:$PATH"
RUN pip install "pip<24.1" "setuptools==65.5.0" "wheel<0.40" psutil six && \
    pip install --no-build-isolation gym==0.21.0
RUN git clone https://github.com/google-research/football.git /opt/football && \
    cd /opt/football && git checkout 3d9e754720a95621bba6475c4d3b0d56fe919014 && \
    pip install --no-build-isolation .
WORKDIR /work
