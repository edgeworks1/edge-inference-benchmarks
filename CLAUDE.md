# CLAUDE.md

Reference pipeline for edge-AI projects: train in PyTorch on a PC, export to hardware-independent
formats (TFLite, ONNX), compile for a specific device (here: nRF5340 with TFLite Micro), measure on
simulator and real board. The BLE data is only the example. Users are usually students who
learn this workflow. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the concepts and
[README.md](README.md) for the commands before proposing structural changes.

## Work in small, explained steps

Users should understand every change, not just receive working code. For any change that is
more than a few lines:

1. **Plan first:** explain the goal and split it into numbered sub-steps. Each step is small
   (roughly ≤ 30–50 changed lines), self-contained and testable on its own.
2. **One step at a time:** implement only the current step. Then explain what changed and
   *why*, and give the command to verify it (and run it, if possible).
3. **Wait** for the user before starting the next step. Only if the user explicitly states to do everything at once, do so, otherwise follow the substep rule always.
4. Prefer the simple, readable solution over a clever one. Match the existing code style.

## Essentials

- Everything runs in Docker from the repo root (WSL/Linux): `make -C ml help`,
  `make -C firmware/nrf5340 <target>`, or `docker compose run --rm ml <ble-cs command>`.
- After changing code in `ml/`, rebuild the image (`make -C ml images`), otherwise the old code runs.
- `ml/` (Python package) and `firmware/` don't know each other; only the Makefiles connect them
  (generated C sources → firmware, `run.json` → `ble-cs results record`).
