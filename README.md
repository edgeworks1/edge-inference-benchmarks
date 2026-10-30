# BLE CS Profiling

Reproducible edge-AI pipeline: small regression models estimate distance from raw BLE Channel
Sounding IQ samples, get quantized to full INT8 and are profiled on a Cortex-M33 (nRF5340,
TFLite Micro + CMSIS-NN), in simulation (Renode) and on the real board.
The focus is on deployment constraints (size, memory, latency), not on squeezing out accuracy.

Everything runs in Docker. The host only needs Docker (plus J-Link tools for the real board).

**New to edge AI?** Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first (German): how such a
project is structured, why, and how to transfer it to your own data, models and hardware.

## Pipeline

```
checkpoint (.pt) ─ export ─▶ fp32_full / int8_full (.tflite)  ─ codegen ─▶ firmware ─▶ Renode / nRF5340 DK
                             fp32_full / int8_qdq  (.onnx)        (C array)                      │
                                      │                                                          │
                                      └──────── size, MAE ──▶ results.json ◀── run.json (cycles, arena, flash, RAM)
```

`ml/` and `firmware/` don't know each other: the Makefiles copy the generated C sources into the
firmware and pipe the firmware's `run.json` back into `ble-cs results record`. Another device
only has to produce the same `run.json`.

## Quickstart

Put the datasets (not part of the repo) into `ml/data/`, then from the repository root:

```bash
mkdir -p ml/checkpoints ml/outputs           # Compose won't create them (they would be owned by root)
docker compose build                         # images: ml (CPU), ml-gpu (CUDA), fw (Zephyr SDK), renode
make -C ml all M=edge_bench_mlp              # train → INT8 TFLite (+ MAE) → C sources into the firmware
make -C firmware/nrf5340 sim                 # build firmware, run it in Renode, record results
make -C ml table TARGET=renode               # all results as a Markdown table
```

No NVIDIA GPU? Add `GPU=0` to train on the CPU. `make -C ml help` lists all steps and parameters.
On the real board use `make -C firmware/nrf5340 measure` instead of `sim` (build, flash, read UART,
record), see [Hardware](#hardware).

## Results

nRF5340 DK, Cortex-M33 @ 128 MHz, one inference, full INT8. MAE on the full eval set (42,242 samples).

| Checkpoint | FP32 MAE [m] | INT8 MAE [m] | INT8 size [KiB] | Cycles (nrf5340dk) | Latency [µs] | Arena [KiB] | Flash [KiB] | RAM [KiB] |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `edge_bench_mlp` | 1.399 | 1.395 | 26.0 | 61,060 | 477 | 1.2 | 216.3 | 164.8 |
| `mobilenet` | 0.506 | 0.510 | 1905.0 | – | – | – | – | – |
| `mobilenet/bad_simd` | 0.507 | 0.509 | 1890.8 | – | – | – | – | – |
| `mobilenet/relu` | 0.470 | 0.476 | 1913.6 | – | – | – | – | – |
| `mobilenet/small` | 0.413 | 0.425 | 337.2 | 19,486,472 | 152,238 | 50.4 | 527.5 | 164.8 |
| `mobilenet_v3_micro_2d` | 0.413 | 0.425 | 337.2 | 19,486,472 | 152,238 | 50.4 | 527.5 | 164.8 |
| `my_mlp` | 1.399 | 1.395 | 26.0 | 61,060 | 477 | 1.2 | 216.3 | 164.8 |

`–` = does not fit into the 1 MB flash of the nRF5340 (runs in Renode only: `make -C ml table TARGET=renode`).
Arena = tensor memory TFLite Micro actually needs; Flash and RAM = whole firmware (Zephyr + TFLM + model).
Renode is not cycle-accurate (the board needs 24–32 % more cycles here): good for comparing models,
absolute numbers come from the board.

## Outputs

Each checkpoint gets one folder in `ml/outputs/`, mirroring its path in `ml/checkpoints/`:

```
ml/checkpoints/mobilenet/relu.pt  →  ml/outputs/mobilenet/relu/
                                       ├── fp32_full/model.tflite, model.onnx   float32
                                       ├── int8_full/model.tflite               INT8 incl. input/output (MCU)
                                       ├── int8_qdq/model.onnx                  INT8 weights/activations, float32 I/O
                                       └── results.json
```

`results.json` is filled step by step: `export` adds size, sha256 and MAE of every model;
`sim` / `measure` add one entry per device (`renode`, `nrf5340dk`). The firmware prints the
sha256 of the model it runs, and only that sha256 links a measurement to a model: it can never
end up at the wrong one, and a re-export that changes the model drops old measurements.

```json
"targets": {
  "nrf5340dk": {
    "artifact": "int8_full/model.tflite", "sha256": "795d8933…",
    "cycles": 61060, "latency_us": 477, "arena_used_bytes": 1252,
    "flash_bytes": 221472, "ram_bytes": 168792, "prediction": 1.8906,
    "measured_at": "2026-10-29T21:48:14+00:00"
  }
}
```

## Repository layout

```
ml/                 Python package ble_cs_profiling (CLI: ble-cs): models, training, export, codegen, results
  Makefile          entry points for the ML steps (all run in Docker)
firmware/nrf5340/   Zephyr / NCS 2.6.0 app: runs the model, prints per-layer cycles and a RESULT line
  scripts/          run_json.py: UART log + ELF -> run.json
  Makefile          build, simulate, flash, measure
docs/               setup guides and design notes
compose.yaml        all containers: ml, ml-gpu, fw, renode
```

Training settings are fixed constants in [ml/src/ble_cs_profiling/config.py](ml/src/ble_cs_profiling/config.py)
(seed 42): training twice gives the same checkpoint. New models are registered in
[models/registry.py](ml/src/ble_cs_profiling/models/registry.py).

## Requirements

| What | Needed for |
|---|---|
| Docker Engine + Compose v2 (Linux or WSL2), user in group `docker` | everything |
| NVIDIA driver + [NVIDIA Container Toolkit](docs/nvidia_container_toolkit.md) | GPU training (`ml-gpu`), dev container |
| J-Link, nrfutil, usbipd (WSL) — [setup guide](docs/wsl_flashing_setup.md) | real board only |

Containers run as UID 1000. If `id -u` differs: `docker compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g)`.
After changing code in `ml/` rebuild the image (`make -C ml images`), otherwise the old code runs.

### Hardware

Variant #1 (Renode) gets 8 MB RAM / 16 MB flash, so any model runs. Variant #2 is the real
nRF5340 DK (1 MB flash, 448 KB app RAM, 128 KB arena). Details: [firmware/nrf5340/README.md](firmware/nrf5340/README.md).

### Development

VS Code → **Dev Containers: Reopen in Container** gives the `dev` stage of `ml/Dockerfile`
(GPU torch, pytest, ruff, debugger; launch configs in `.vscode/launch.json`). Its Python
environment is read-only: add packages with `uv add --no-sync <package>`, then rebuild the container.

Code style is enforced with [ruff](https://docs.astral.sh/ruff/) (configured in `ml/pyproject.toml`):
`make -C ml format` formats and applies safe fixes, `make -C ml check` only reports. Run `check`
before every commit; the dev container formats on save. Once per clone, let `git blame` skip
pure formatting commits: `git config blame.ignoreRevsFile .git-blame-ignore-revs`.
