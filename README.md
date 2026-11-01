# Edge Inference Benchmarks

Reproducible edge-AI pipeline: small regression models estimate distance from raw BLE Channel
Sounding IQ samples, get quantized to full INT8 and are profiled on a Cortex-M33 (nRF5340,
TFLite Micro + CMSIS-NN) in simulation (Renode) and on the real board, and on the Neural-ART NPU
of an STM32N6. The focus is on deployment constraints (size, memory, latency), not on squeezing
out accuracy.

Everything runs in Docker. The host only needs Docker (plus J-Link tools for the nRF5340 board).
The exception is the STM32N6: ST's tools are Windows programs, called from WSL through interop.

## Pipeline

```
checkpoint (.pt) ─ export ─▶ fp32_full / int8_full (.tflite)  ─ codegen ─▶ firmware ─▶ Renode / nRF5340 DK
                             fp32_full / int8_qdq  (.onnx)        (C array)                      │
                             int8_full / int8_qdq ─ stedgeai ─▶ ST validation app ─▶ STM32N6 ────┤
                                      │                                                          │
                                      └──────── size, MAE ──▶ results.json ◀── run.json (cycles, arena, flash, RAM)
```

`ml/` and `firmware/` don't know each other: the Makefiles copy the generated C sources into the
firmware and pipe the firmware's `run.json` back into `edge-ib results record`. Another device
only has to produce the same `run.json`: the STM32N6 does, without own firmware, because ST's
compiler (`stedgeai`) builds the INT8 `.tflite` or `.onnx` into ST's validation app.

## Quickstart

The repo ships a small public data sample (`ml/data/sample/`: 100 training and 100 eval rows,
raw IQ samples + ground-truth distance), so the pipeline runs right after cloning. From the
repository root:

```bash
mkdir -p ml/checkpoints ml/outputs           # Compose won't create them (they would be owned by root)
docker compose build                         # images: ml (CPU), ml-gpu (CUDA), fw (Zephyr SDK), renode
make -C ml all M=edge_bench_mlp              # train → INT8 TFLite (+ MAE) → C sources into the firmware
make -C firmware/nrf5340 sim                 # build firmware, run it in Renode, record results
make -C ml table TARGET=renode               # all results as a Markdown table
```

No NVIDIA GPU? Add `GPU=0` to train on the CPU. `make -C ml help` lists all steps and parameters.
The sample is only meant to run the pipeline end to end: models trained on it are not accurate.
With a full dataset in `ml/data/full/` (same file names, not published) add `DATA=full`.
On the real board use `make -C firmware/nrf5340 measure` instead of `sim` (build, flash, read UART,
record), see [Hardware](#hardware). On the STM32N6, one command goes from checkpoint to
`results.json`: `make -C ml ckpt-to-stm32n6 CKPT=checkpoints/<name>.pt [FORMAT=onnx]`
(setup: [firmware/stm32n6/README.md](firmware/stm32n6/README.md)).

## Results

Produced with the full dataset (`DATA=full`, not published): MAE on 42,242 eval samples, INT8
calibration on 1,000 training samples. All tables come from `make -C ml table TARGET=<device>`.

**How to read the numbers:** this is a benchmark of edge deployment, not of model quality. The
point is a feel for latency, RAM and flash per model, optimization and device. Absolute
accuracies don't matter, and the models are not compared by MAE against each other (e.g.
`mobilenet/relu` vs. `mobilenet`). MAE only serves as a sanity check within one model: does INT8
quantization make it noticeably worse than FP32?

### nRF5340 DK (Cortex-M33 @ 128 MHz, TFLite Micro + CMSIS-NN)

One inference, full INT8.

| Checkpoint | FP32 MAE [m] | INT8 MAE [m] | INT8 size [KiB] | Cycles (nrf5340dk) | Cycles (renode) | Latency [µs] | Arena [KiB] | Flash [KiB] | RAM [KiB] |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `edge_bench_mlp` | 1.399 | 1.376 | 25.9 | 61,069 | 46,384 | 477 | 1.2 | 216.2 | 164.8 |
| `mobilenet` | 0.506 | 0.554 | 1894.7 | – | 16,328,976 | – | – | – | – |
| `mobilenet/bad_simd` | 0.507 | 0.508 | 1880.5 | – | 16,656,016 | – | – | – | – |
| `mobilenet/relu` | 0.470 | 0.477 | 1902.4 | – | 14,531,440 | – | – | – | – |
| `mobilenet/small` | 0.413 | 0.422 | 329.9 | 19,481,318 | 15,775,824 | 152,197 | 50.4 | 520.2 | 164.8 |
| `mobilenet_v3_micro_2d` | 0.413 | 0.422 | 329.9 | 19,481,318 | 15,775,824 | 152,197 | 50.4 | 520.2 | 164.8 |

`–` = does not fit into the 1 MB flash of the nRF5340 (runs in Renode only: `make -C ml table TARGET=renode`).
Arena = tensor memory TFLite Micro actually needs; Flash and RAM = whole firmware (Zephyr + TFLM + model).
Renode is not cycle-accurate (the board needs 24–32 % more cycles here): good for comparing models,
absolute numbers come from the board.

### STM32N6 (NUCLEO-N657X0-Q, Neural-ART NPU)

MCU @ 800 MHz, NPU @ 1 GHz; median of 20 inferences. Full INT8 TFLite:

| Checkpoint | FP32 MAE [m] | INT8 MAE [m] | INT8 size [KiB] | Cycles (stm32n6) | Latency [µs] | Arena [KiB] | Flash [KiB] | RAM [KiB] |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `edge_bench_mlp` | 1.399 | 1.376 | 25.9 | 79,056 | 99 | 0.4 | 38.4 | 0.4 |
| `mobilenet` | 0.506 | 0.554 | 1894.7 | 5,412,545 | 6,766 | 11.2 | 1707.5 | 11.3 |
| `mobilenet/bad_simd` | 0.507 | 0.508 | 1880.5 | 5,748,717 | 7,186 | 11.2 | 1772.1 | 11.3 |
| `mobilenet/relu` | 0.470 | 0.477 | 1902.4 | 5,292,689 | 6,616 | 11.2 | 1696.3 | 11.3 |
| `mobilenet/small` | 0.413 | 0.422 | 329.9 | 1,903,379 | 2,379 | 18.6 | 268.8 | 18.6 |
| `mobilenet_v3_micro_2d` | 0.413 | 0.422 | 329.9 | 1,903,379 | 2,379 | 18.6 | 268.8 | 18.6 |

ONNX (INT8 QDQ, float input and output). The NPU compiler maps more ONNX operators than TFLite
ones, see [docs/onnx_export_stm32n6.md](docs/onnx_export_stm32n6.md):

| Checkpoint | FP32 MAE [m] | INT8 MAE [m] | INT8 size [KiB] | Cycles (stm32n6-onnx) | Latency [µs] | Arena [KiB] | Flash [KiB] | RAM [KiB] |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `edge_bench_mlp` | 1.399 | 1.376 | 25.9 | 79,044 | 99 | 0.4 | 38.4 | 0.4 |
| `mobilenet` | 0.506 | 0.554 | 1894.7 | 5,389,305 | 6,737 | 12.1 | 1739.4 | 12.1 |
| `mobilenet/bad_simd` | 0.507 | 0.508 | 1880.5 | 5,727,805 | 7,160 | 12.1 | 1803.8 | 12.2 |
| `mobilenet/relu` | 0.470 | 0.477 | 1902.4 | 4,784,145 | 5,980 | 7.0 | 1702.8 | 7.0 |
| `mobilenet/small` | 0.413 | 0.422 | 329.9 | 1,816,429 | 2,271 | 24.9 | 291.3 | 25.0 |
| `mobilenet_v3_micro_2d` | 0.413 | 0.422 | 329.9 | 1,816,429 | 2,271 | 24.9 | 291.3 | 25.0 |

Cycles = MCU cycles per inference. Arena = activation memory; Flash and RAM cover the model only
(weights + network code, activations + network state), not the whole app as on the nRF5340.
MAE and size columns always refer to the TFLite export. No FP32 rows: the NPU needs INT8.

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
ml/                 Python package edge_ib (CLI: edge-ib): models, training, export, codegen, results
  Makefile          entry points for the ML steps (all run in Docker)
firmware/nrf5340/   Zephyr / NCS 2.6.0 app: runs the model, prints per-layer cycles and a RESULT line
  scripts/          run_json.py: UART log + ELF -> run.json
  Makefile          build, simulate, flash, measure
firmware/stm32n6/   no own firmware: ST's compiler + validation app, called from WSL via interop
  scripts/          dk.sh: Windows tool calls; run_json.py: profile over the COM port -> run.json
  Makefile          setup, compile, load, measure
docs/               setup guides and design notes
compose.yaml        all containers: ml, ml-gpu, fw, renode
```

Training settings are fixed constants in [ml/src/edge_ib/config.py](ml/src/edge_ib/config.py)
(seed 42): training twice gives the same checkpoint. New models are registered in
[models/registry.py](ml/src/edge_ib/models/registry.py).

## Requirements

| What | Needed for |
|---|---|
| Docker Engine + Compose v2 (Linux or WSL2), user in group `docker` | everything |
| NVIDIA driver + [NVIDIA Container Toolkit](docs/nvidia_container_toolkit.md) | GPU training (`ml-gpu`), dev container |
| J-Link, nrfutil, usbipd (WSL) — [setup guide](docs/wsl_flashing_setup.md) | nRF5340 board only |
| ST Edge AI Core, STM32CubeIDE, uv on Windows — [setup guide](firmware/stm32n6/README.md) | STM32N6 only |

Containers run as UID 1000. If `id -u` differs: `docker compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g)`.
After changing code in `ml/` rebuild the image (`make -C ml images`), otherwise the old code runs.

### Hardware

Variant #1 (Renode) gets 8 MB RAM / 16 MB flash, so any model runs. Variant #2 is the real
nRF5340 DK (1 MB flash, 448 KB app RAM, 128 KB arena). Details: [firmware/nrf5340/README.md](firmware/nrf5340/README.md).
The STM32N6 runs on a NUCLEO-N657X0-Q in development boot mode; the weights go to the external
flash, the app into RAM. Details: [firmware/stm32n6/README.md](firmware/stm32n6/README.md).

### Development

VS Code → **Dev Containers: Reopen in Container** gives the `dev` stage of `ml/Dockerfile`
(GPU torch, pytest, ruff, debugger). Its Python
environment is read-only: add packages with `uv add --no-sync <package>`, then rebuild the container.

Code style is enforced with [ruff](https://docs.astral.sh/ruff/) (configured in `ml/pyproject.toml`):
`make -C ml format` formats and applies safe fixes, `make -C ml check` only reports. Run `check`
before every commit; the dev container formats on save. Once per clone, let `git blame` skip
pure formatting commits: `git config blame.ignoreRevsFile .git-blame-ignore-revs`.

## License

[MIT](LICENSE)
