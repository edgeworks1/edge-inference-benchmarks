# Architecture

How this pipeline is built and why. The example task (distance from BLE Channel Sounding data)
is replaceable. The structure behind it is the part worth reusing, whether the model is a GBDT,
a CNN or a transformer, and whether the target is an MCU, an NPU or a Jetson.

This is not a template where you change two lines and it runs. Different data, models or
hardware mean real work, usually in the hard places: export, quantization, operator support.
Copy the structure, not the BLE code.

## 1. Three worlds

```
 1 TRAINING                  2 EXCHANGE FORMAT              3 TARGET HARDWARE
 PC / GPU server             hardware-independent           hardware-specific

 PyTorch, float32     export   ONNX, TFLite            compile   C array + TFLite Micro (MCU)
 Python, lots of RAM ───────▶  frozen graph           ────────▶  ST Edge AI (STM32N6 NPU)
 flexible, slow                + quantization (INT8)             TensorRT engine (Jetson)
                                                                 no Python, KB instead of GB
```

- **Training** needs flexibility: autograd, debugger, GPU. The result is a **checkpoint**
  (`.pt`): weights plus everything needed to rebuild the model.
- **Exchange format** freezes the model into a fixed graph of standard operators (Conv,
  FullyConnected, Add ...). Quantization happens here: float32 → int8, 4x smaller, much faster
  on an MCU. One export can serve several targets.
- **Target hardware**: a vendor compiler or runtime maps the graph onto one chip. It picks
  optimized kernels (CMSIS-NN on Cortex-M), plans memory and offloads layers to an NPU. Only
  here do you get real numbers for latency, RAM and flash.

Why split it like this? Same idea as a compiler: source → IR → machine code. Training knows
nothing about the chip, the chip knows nothing about PyTorch. New hardware only changes world 3.

Every transition can cost accuracy or break something: a missing operator, a quantization
loss, an unsupported layer. So each step is measured, not assumed: TFLite output vs. PyTorch,
FP32 and INT8 MAE separately, cycles in the simulator and on the board.

## 2. The path through this repo

| # | Step               | Command                                      | Result                                                  |
| - | ------------------ | -------------------------------------------- | ------------------------------------------------------- |
| 1 | Data               | –                                            | `ml/data/{sample,full}/{training,eval}.parquet`         |
| 2 | Training           | `make -C ml train M=<model>`                 | `ml/checkpoints/<name>.pt`                              |
| 3 | Export + quantize  | `make -C ml export`                          | `ml/outputs/<name>/{fp32_full,int8_full,int8_qdq}/`     |
| 4 | Codegen            | `make -C ml to-nrf samples`                  | model as a C array in `firmware/nrf5340/src/`           |
| 5 | Build + measure    | `make -C firmware/nrf5340 sim` / `measure`   | `build/<variant>/run.json`                              |
| 6 | Collect results    | (part of 3 and 5)                            | `ml/outputs/<name>/results.json`                        |

**Training.** The checkpoint stores the registry name of the model (`models/registry.py`), its
arguments, the training settings and the eval MAE. Training is expensive, export is cheap:
train once, re-export as often as needed.

**Export.** Three variants per checkpoint:

| Folder       | Format        | Use                                                             |
| ------------ | ------------- | --------------------------------------------------------------- |
| `fp32_full/` | TFLite + ONNX | float32 reference for accuracy                                  |
| `int8_full/` | TFLite        | INT8 including input and output, what an MCU needs              |
| `int8_qdq/`  | ONNX          | INT8 weights/activations, float32 I/O (NPU and GPU toolchains)  |

Quantization needs **calibration**: 1,000 real training samples run through the model to find
the value range of each layer. Bad calibration data, bad INT8 model.

**Codegen.** An MCU has no file system, so the `.tflite` becomes a byte array in C and lives in
flash. Its sha256 is embedded too (see "Traceable" below).

**Firmware.** Zephyr + TensorFlow Lite Micro + CMSIS-NN. One app, two variants:

- **Renode** simulates the chip with extra memory. Every model fits, but the cycle counts are
  only good for comparing models (the board needs 24-32 % more).
- **The real board** gives the numbers that count (hardware-in-the-loop).

The STM32N6 has no firmware of its own: ST's compiler (`stedgeai`) builds the INT8 model into
ST's validation app, and a script turns its output into the same `run.json`.

**Results.** The firmware prints one machine-readable line (`RESULT {...}`). A script on the
firmware side turns it into a device-independent `run.json`, the ML side matches it to the
right model by sha256.

## 3. Principles

### Linux / WSL first

- Embedded toolchains, Docker and scripts are at home on Linux.
- On Windows you end up with an IDE click setup nobody can reproduce. So: WSL2.
- Windows only forwards the USB debugger (`usbipd`, see [wsl_flashing_setup.md](wsl_flashing_setup.md)).
- Exception: ST's tools are Windows-only and are called from WSL through interop.

### Everything in Docker

- Every step runs in a container with pinned versions: Python via `uv.lock`, Zephyr SDK and
  Renode checked by sha256.
- No more "works on my machine". A new team member, a CI server and you in a year get the same
  environment.
- The host only needs Docker (and J-Link tools for the board).
- Separate images (`ml`, `ml-gpu`, `fw`, `renode`): they differ a lot in size and change
  independently. The ML image shouldn't rebuild because the Zephyr SDK changed.

### CLI, not notebooks

- Every step is a command (`edge-ib train`, `edge-ib export` ...). Commands can be scripted,
  reviewed and repeated, e.g. measuring 20 models overnight.
- Notebooks have hidden state and are hard to diff. Fine for exploring data
  ([ml/experiments/](../ml/experiments/README.md)), but whatever works moves into the package.

### Makefiles as entry point

- `make -C ml help` lists everything.
- They hold the long `docker compose run ...` commands, so nobody has to type or guess them.
- They are the only place that connects ML and firmware.

### ML and firmware are decoupled

`ml/` knows no firmware paths, the firmware knows no Python package. They only share two files:

```
ml/outputs/.../int8_full/codegen/model_data.{h,cc}  ── make -C ml to-nrf ──▶  firmware/nrf5340/src/ML_Models/
ml/outputs/.../results.json  ◀── edge-ib results record ──  firmware/.../build/<v>/run.json
```

- Both sides run in different containers with different toolchains and change independently.
- A new target only needs its own `firmware/<target>/` that takes a model and returns a
  `run.json` in the same format. The STM32N6 was added this way without touching the ML code.

### Reproducible: code in Git, generated files not

- In Git: code, config, lockfiles, a small data sample.
- Not in Git: checkpoints, exports, generated C code, build folders. All of it can be rebuilt.
- Training settings are constants with a fixed seed (`ml/src/edge_ib/config.py`): training
  twice gives the same checkpoint.

### Traceable: from model to measurement

- The firmware prints the sha256 of the `.tflite` it actually contains.
- No matching export → the measurement is rejected (e.g. firmware not rebuilt after codegen).
- A re-export that changes the model drops its old measurements.

### Fixed paths, fixed names

- `ml/outputs/<checkpoint>/<precision>/model.<format>`: everyone and every script knows where
  a file is.
- The folder is named after the checkpoint, not the model type, so two runs of the same model
  don't overwrite each other.

### Measure, don't guess

Simulator, exchange format and board all give different numbers. One example from this repo:
INT4 instead of INT8 weights made the MobileNet a third smaller in flash, but cost 35 % more
cycles and a 5.6x larger tensor arena, because TFLite Micro unpacks INT4 weights into an INT8
buffer at runtime. You only see that by measuring.

## 4. Reusing this for your own project

**Keep:** the three worlds, the folder layout (`ml/`, `firmware/<target>/`, `docs/`,
`compose.yaml`), Docker for every step, Makefiles as entry point, the checkpoint as hand-over,
fixed paths, `results.json` + `run.json` as contracts, measuring on real hardware.

**Expect to rewrite:**

| Area          | Here (BLE example)                 | Likely in your project                                           |
| ------------- | ---------------------------------- | ---------------------------------------------------------------- |
| Data          | Parquet, 300 IQ columns            | images, time series, radar point clouds, graphs ...              |
| Preprocessing | none (raw data)                    | normalization, windowing, features. Must be identical on device! |
| Metric        | MAE (regression)                   | accuracy, F1, mAP ...                                            |
| Models        | MLP, MobileNetV3                   | yours, added to the registry                                     |
| Export        | TFLite (litert-torch), ONNX        | depends on model and target. Usually the hardest part            |
| Firmware      | Zephyr + TFLM on nRF5340, STM32N6  | other MCU, NPU toolchain, Linux board                            |

**Typical pitfalls by model type:**

- **Classic ML (GBDT, random forest):** no PyTorch, same idea. Save the model → ONNX
  (`skl2onnx`) or straight to C (`emlearn`, `m2cgen`) → device. Often the easiest way onto an MCU.
- **CNNs (like here):** well supported. Watch for operators the runtime can't fuse or doesn't
  support (HardSwish here) and channel counts that don't match the SIMD width or NPU.
- **Transformers / ViTs:** softmax, LayerNorm and attention are hard to quantize and often
  unsupported on accelerators, so they fall back to the CPU. Check operator support early.
- **Graph NNs:** dynamic graphs don't fit a static exchange format. Usually you need a fixed
  maximum size with padding.

The firmware has to register every operator it uses (list in
`firmware/nrf5340/src/inference.cpp`). A new model often means updating that list.

**Go end to end first, then improve:**

1. Push a tiny dummy model (e.g. a small MLP) through the whole chain down to a measurement on
   hardware. The hard problems (export, memory, toolchain) show up in week one, not the last week.
2. Swap in the real model and check each step: does it export, does the output match PyTorch,
   how much does INT8 cost, does it fit?
3. Only then optimize (architecture, quantization, pruning), and measure every change.

## 5. Glossary

| Term                 | Meaning                                                            |
| -------------------- | ------------------------------------------------------------------ |
| Checkpoint           | saved trained model (weights + metadata), here `.pt`               |
| Quantization (PTQ)   | float32 → int8 after training: smaller, faster, slightly less exact |
| Calibration          | representative data that sets the value ranges for quantization    |
| Full INT8            | input and output are int8 too, no float math left on the device    |
| QDQ                  | quantize/dequantize nodes in an ONNX graph, I/O stays float        |
| TFLite Micro (TFLM)  | runtime for TFLite models on MCUs, no OS needed                    |
| CMSIS-NN             | Arm's optimized kernels for Cortex-M (SIMD)                        |
| Tensor arena         | fixed RAM block where TFLM keeps all intermediate tensors          |
| Flash / RAM          | program memory (code, model) / working memory (variables, arena)   |
| Renode               | MCU simulator: good for comparisons and CI, not cycle-accurate     |
| HIL                  | hardware-in-the-loop: testing and measuring on the real board      |
| sha256               | file checksum: same file, same value, so a unique match            |
