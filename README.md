# BLE CS Profiling

Distance-regression models for BLE Channel Sounding (raw IQ samples), quantized to INT8
and profiled on edge hardware: nRF5340 (Cortex-M33, TFLite Micro), later STM32N6
(Neural-ART NPU) and Jetson. The focus is on deployment constraints (size, memory,
latency), not on squeezing out accuracy.

## Repository layout

```
ml/              Python package `ble_cs_profiling`: models, training, INT8 export, C/C++ codegen
  Dockerfile     Image for the ml pipeline (CPU or CUDA variant)
firmware/nrf5340/  Zephyr/NCS inference app: Renode simulation + real DK (see its README)
docs/            Design notes (e.g. why ONNX is exported with opset 13)
compose.yaml     Container entry points: `ml` (CPU), `ml-gpu` (CUDA), `fw` (firmware build), `renode`
```

## Prerequisites

There are two ways to run the pipeline. You need **either** Docker **or** a local Python setup.

### Option A: Docker (recommended)

| What | Needed for | Notes |
|---|---|---|
| Docker Engine with Compose v2 | everything | Linux or WSL2. Your user must be in the `docker` group |
| NVIDIA driver | `ml-gpu` only | On WSL2 install the **Windows** driver, nothing inside WSL |
| **NVIDIA Container Toolkit** | `ml-gpu` only | **Not included in Docker, install it yourself**, see below |

The CPU service `ml` works without any NVIDIA software.

#### Installing the NVIDIA Container Toolkit (Ubuntu / WSL2)

Containers cannot see the GPU on their own. The toolkit mounts the host's driver and GPU
device into the container at start-up, so images stay driver-free and portable. Without it,
`docker compose run ml-gpu …` fails with:

```
failed to discover GPU vendor from CDI: no known GPU vendor found
```

Install it once on the host (official NVIDIA apt repository):

```bash
# 1. Signing key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# 2. Package source
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 3. Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 4. Register the runtime with Docker (writes /etc/docker/daemon.json) and restart Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Check that it worked. This should print your GPU:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

If you still see the CDI error, generate the GPU description manually and restart Docker:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker
```

> ⚠️ **WSL2:** Never install a Linux NVIDIA driver inside WSL (`nvidia-driver-*`, or
> `cuda` meta-packages that pull one in). WSL gets the driver from Windows via
> `/usr/lib/wsl/lib`, and a Linux driver breaks that bridge. The Container Toolkit is
> not a driver and is safe to install.
>
> A local CUDA toolkit (`nvcc`) is **not** needed. PyTorch ships its own CUDA runtime,
> only the driver has to be new enough (CUDA 13.0 → driver ≥ 580).

Tested with: Ubuntu 24.04 on WSL2, Docker Engine 29.8, NVIDIA Container Toolkit 1.20.1,
Windows driver 596.08.

### Development: Dev Container (recommended for working on the code)

Requires everything from Option A (including the NVIDIA Container Toolkit, the dev
container uses the GPU) plus the VS Code extension **Dev Containers**
(`ms-vscode-remote.remote-containers`).

Open the repository in VS Code → `F1` → **Dev Containers: Reopen in Container**.

- Image: stage `dev` of `ml/Dockerfile` (CUDA torch, plus git, pytest, ruff, ipykernel)
- The repository is mounted at `/workspace`. The Python environment lives in `/app/.venv`
  inside the image and is **read-only**. The host's `ml/.venv` is never touched.
- To add a package: `uv add --no-sync <package>` (updates `pyproject.toml` + `uv.lock`),
  then **Dev Containers: Rebuild Container**.
- Shell history and caches are kept in the named volumes `ble-cs-bashhistory` and
  `ble-cs-cache`, so they survive rebuilds.

Personal extensions (e.g. AI assistants) belong in your VS Code user settings, not in
`devcontainer.json`: `"dev.containers.defaultExtensions": ["<publisher.extension>"]`.

### Option B: Local Python

- [uv](https://docs.astral.sh/uv/) ≥ 0.11 (uv installs Python 3.11 itself)
- For GPU training: an NVIDIA driver supporting CUDA 13.0

```bash
cd ml
uv sync --extra cu130      # GPU (or: --extra cpu)
source .venv/bin/activate
ble-cs --help
```

Always pass the extra (`cu130` or `cpu`). A plain `uv sync` swaps in a different torch build.

## Data

The datasets are **not** part of the repository. Place them in `ml/data/`:

```
ml/data/combined_training_reduced.parquet
ml/data/combined_eval_reduced.parquet
```

Also create the output folders before the first Docker run. Compose refuses to create
them itself, since Docker would create them owned by root:

```bash
mkdir -p ml/checkpoints ml/outputs
```

## Quickstart (Docker)

Run from the repository root:

```bash
docker compose build ml ml-gpu

docker compose run --rm ml models                                         # list models
docker compose run --rm ml-gpu train -m edge_bench_mlp                    # train on GPU
docker compose run --rm ml export /app/checkpoints/edge_bench_mlp.pt      # ONNX + TFLite INT8
docker compose run --rm ml codegen model /app/outputs/edge_bench_mlp_full_int8.tflite
```

Paths passed to `ble-cs` are container paths. `/app/data`, `/app/checkpoints` and
`/app/outputs` map to `ml/data`, `ml/checkpoints` and `ml/outputs`. Results land in those
folders and belong to your user (the container runs as a non-root user with UID 1000).

If your host user does not have UID 1000 (`id -u`), build with your own IDs:

```bash
docker compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g)
```
