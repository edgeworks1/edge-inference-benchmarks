# STM32N6 (NUCLEO-N657X0-Q)

Runs an exported model (`.tflite` or `.onnx` from `ml/outputs/`) on the Neural-ART NPU of the
STM32N6, measures it on the real board and records the result. There is no own firmware here:
ST's tools compile the model and load it into ST's validation app, which is then profiled over
the ST-LINK's virtual COM port. Everything runs from the command line, no GUI.

```
model/model.tflite | .onnx      copied in by make -C ml to-stm32n6 [FORMAT=onnx]
   │  make generate              stedgeai generate: NPU compiler → network.c + weights (.raw)
   ▼
st_ai_output/                    (in the Windows work folder)
   │  make load                  n6_loader.py: builds the validation app, flashes the weights to the
   ▼                             external flash, loads the app into RAM through the debugger
NUCLEO-N657X0-Q
   │  make measure               scripts/run_json.py: inferences over the COM port (ST's AiRunner)
   ▼
build/run.json  →  edge-ib results record  →  ml/outputs/<checkpoint>/results.json
```

The last line plays the same role as on the nRF5340: [scripts/run_json.py](scripts/run_json.py)
turns the device data into the device-independent `run.json`, the only interface to the ML side.
Device name in `results.json`: `stm32n6` for a TFLite model, `stm32n6-onnx` for an ONNX model.

## Quick start

From the repo root, once the tools are installed ([section 1](#1-installation)) and the board is
prepared ([section 3](#3-board-setup)):

```bash
make -C ml ckpt-to-stm32n6 CKPT=checkpoints/edge_bench_mlp.pt DATA=full              # TFLite
make -C ml ckpt-to-stm32n6 CKPT=checkpoints/edge_bench_mlp.pt DATA=full FORMAT=onnx  # ONNX
make -C ml table TARGET=stm32n6
```

`ckpt-to-stm32n6` = export + `to-stm32n6` + `make -C firmware/stm32n6 measure`; `all-stm32n6`
trains first. One run takes 1.5 to 4 minutes, mostly building the validation app and flashing.

| Target | What it does | Board |
|---|---|---|
| `make setup` | create or update the work folder, venv and `config.json`; optional, `generate` runs it | no |
| `make generate` | `setup`, then compile `model/model.*` for the NPU; fails if it needs external RAM | no |
| `make load` | `generate`, then build the validation app with it and start it | yes |
| `make measure` | `load`, then profile 20 inferences and record the medians | yes |

## Why Windows and not Docker like the nRF5340?

The ST tools exist for Linux as well, but:

- The ST downloads sit behind a myST login with a license prompt. A Dockerfile cannot fetch them,
  and an image containing them must not be distributed publicly.
- Passing the ST-LINK through to WSL (`usbipd bind`) needs admin rights once. On Windows the
  board just works.

So the Makefile runs in WSL like everywhere else in this repo, but [scripts/dk.sh](scripts/dk.sh)
calls the Windows programs through WSL interop. They always run inside the Windows work folder
and never see a WSL path: `cmd` cannot enter `\\wsl.localhost\...`, and the `make` inside ST's
loader fails on it. That is why the model and the helper script are copied to the work folder.

## Tested versions

| Component | Version |
|---|---|
| Windows | 11, no admin rights in daily use |
| ST Edge AI Core | 4.0.1 (STM32CubeAI 12.0.1, Neural-ART compiler 1.1.3) |
| STM32CubeIDE | 1.18.1 (GNU Tools for STM32 13.3.rel1, bundled CubeProgrammer 2.19) |
| STM32CubeProgrammer (optional, standalone) | 2.23.0 |
| uv | 0.11.3 (provides Python 3.11) |
| Board | NUCLEO-N657X0-Q with STLINK-V3EC |

Older or newer versions may work but are untested. With STM32CubeIDE 2.x in particular, the
internal folder names that ST's loader searches for may change.

## 1. Installation

| Tool | Purpose | Admin rights |
|---|---|---|
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | `stedgeai` (NPU compiler), validation app, loader and runner scripts | depends on the target folder |
| [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) | Arm GCC, `make`, `ST-LINK_gdbserver`, `STM32_Programmer_CLI`, ST-LINK driver | yes, once (driver) |
| [uv](https://docs.astral.sh/uv/) (Windows) | fetches Python 3.11 and creates the venv | no |

Nothing else. Arm compiler, `make` and programmer are **not** installed separately and **not**
added to `PATH`: ST's loader takes them from the CubeIDE installation. The standalone
[STM32CubeProgrammer](https://www.st.com/en/development-tools/stm32cubeprog.html) is optional; if it
is installed in its default location, it is used. All ST downloads need a free myST account.

**ST Edge AI Core:** run the Windows installer, select at least **STM32 / Neural-ART (STM32N6)**,
and install into a folder you can write to **without admin rights**, e.g. `C:\ST\STEdgeAI\4.0`.
Not `C:\Program Files`: the loader writes into the validation app inside this installation on
every run. Check: `C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe --version` prints
`ST Edge AI Core v4.0.1-…`.

**STM32CubeIDE:** run the installer (admin rights once, for the ST-LINK driver), e.g. into
`C:\ST\STM32CubeIDE_1.18.1`. You never need to open the IDE. Check the driver: with the board on
the **ST-LINK socket** ([section 3](#3-board-setup)), Device Manager lists
**STMicroelectronics STLink Virtual COM Port (COMx)** under "Ports (COM & LPT)".

**uv:** in a normal PowerShell, `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.
Then, in WSL, `uv.exe --version` must work (WSL finds Windows programs through the Windows `PATH`).
uv brings its own Python; an existing msys2, Chocolatey or Store Python is not used.

## 2. Paths

The paths are **Make variables** in the [Makefile](Makefile) with defaults; no Windows
environment variables are needed:

| Variable | Default | Meaning |
|---|---|---|
| `STEDGEAI_DIR` | `C:/ST/STEdgeAI/4.0` | root of the ST Edge AI installation |
| `CUBEIDE_DIR` | `C:/ST/STM32CubeIDE_1.18.1/STM32CubeIDE` | folder containing `stm32cubeide.exe` (with `plugins\` below it) |
| `N6_WORK_DIR` | `C:/work/n6` | work folder on a **local Windows drive**: venv, `config.json`, compiled model |
| `CUBEPROG_CLI` | default install path of the standalone CubeProgrammer | used if it exists, else the CubeIDE one |

Override per call, always as Windows paths with `/`: `make -C firmware/stm32n6 measure N6_WORK_DIR=D:/n6`.

**Why not Windows environment variables (`setx`)?** Processes started from WSL inherit the
environment WSL got from Windows when it started. WSL only sees variables set with `setx` after
`wsl --shutdown`, and it passes them on to Windows programs only if they are also listed in
`WSLENV`. Make variables take effect immediately, are visible in the Makefile and apply to one
call only.

There is no manual setup step: every `make generate` (and so every `load` and `measure`) first
creates the work folder if needed, syncs the venv with [requirements.txt](requirements.txt) (ST's
`scripts/ai_runner/requirements.txt` plus `colorama`, with `protobuf` pinned to 3.20.x) and
writes `config.json` for ST's loader (forward slashes, no BOM). Once installed this takes about
1.5 s, and the venv always matches the repo, also after a `git pull`.

## 3. Board setup

The NUCLEO-N657X0-Q has **two** USB-C sockets. Only the one labelled **ST-LINK** leads to the
debugger with the virtual COM port. The other one (next to the Ethernet socket) goes straight to
the STM32N6; without matching firmware it shows no USB device, but it can power the board.

Tested setup with **two cables**:

1. **Data:** USB cable from the PC into the **ST-LINK socket**. Use a cable known to carry data.
2. **Power:** second USB-C cable into the **other socket** (next to Ethernet), from a USB-C port of
   the PC or a USB-C power supply. Jumper **CN9 on 3-4** (power from this socket).
3. Boot jumpers in **development mode**: **JP2 (BOOT1) on position 2** (pins 2-3), JP1 (BOOT0)
   stays on position 1. In this mode the STM32N6 does not boot from the external flash, and the
   debugger can load the validation app into RAM. In normal operation (boot from flash) both are
   on position 1. The positions are printed on the board; forums sometimes swap the names JP1
   and JP2, the board manual [UM3417](https://www.st.com/en/evaluation-tools/nucleo-n657x0-q.html#documentation)
   is authoritative.
4. Press **Reset** once or replug the board.

The board stays in this mode for all measurements. `run_json.py` finds the COM port itself
(USB ID `0483:3754` of the STLINK-V3EC).

## 4. The chain by hand

What the Make targets do, for debugging or trying options. In a `cmd` window, in the work folder
created by `make generate`; the MLP as an example. Set the paths for this window only:

```bat
set STEDGEAI_CORE_DIR=C:\ST\STEdgeAI\4.0
set N6_WORK_DIR=C:\work\n6
set STEDGEAI=%STEDGEAI_CORE_DIR%\Utilities\windows\stedgeai.exe
set PYTHONPATH=%STEDGEAI_CORE_DIR%\scripts\ai_runner
cd /d "%N6_WORK_DIR%"
.venv\Scripts\activate
```

Copy the model to the work folder first; never hand a `\\wsl.localhost\...` path to the ST tools.
From WSL: `cp ml/outputs/edge_bench_mlp/int8_full/model.tflite /mnt/c/work/n6/`.

`int8_full/model.tflite` is INT8 including input and output, `int8_qdq/model.onnx` is INT8 inside
with float input and output. Both run on the NPU. FP32 models run on the CPU only, because the
NPU needs INT8.

**Analyze (no board):**

```bat
"%STEDGEAI%" analyze -m model.tflite --target stm32n6 --st-neural-art -o out_analyze
```

**Compile for the NPU (no board), what `make generate` does:**

```bat
"%STEDGEAI%" generate -m model.tflite --target stm32n6 --st-neural-art -o st_ai_output
```

Things to look at in the output:

- **Epochs details:** how many epochs are `HW`, `SW` or `hybrid`. `SW` means part of the model
  runs on the CPU instead of the NPU.
- **Memory usage information:** the `hyperRAM` line must show **0 B**, the Nucleo has no external
  RAM (`make generate` checks this).
- `st_ai_output\` then contains `network.c` and `network_atonbuf.xSPI2.raw` (weights for the
  external flash), among others.

**Load onto the board, what `make load` does:**

```bat
python "%STEDGEAI_CORE_DIR%\scripts\N6_scripts\n6_loader.py" --config config.json --build-config N6-Nucleo --network-file "%N6_WORK_DIR%\st_ai_output\network.c" --project-path "%STEDGEAI_CORE_DIR%\Projects\STM32N6570-DK\Applications\NPU_Validation"
```

It must end with:

```
… Flashing memory xSPI2 -- 21.921 kB
… Building project (conf= N6-Nucleo)
… Loading internal memories & Running the program
… Start operation achieved successfully
```

The app now lives **only in RAM**. After a power loss or unplugging it is gone; load it again.

**Measure and read back the profile** (replace `COM5` with your port, always with baud rate 921600):

```bat
:: total and per-epoch latency
python "%STEDGEAI_CORE_DIR%\scripts\ai_runner\examples\checker.py" -d serial:COM5:921600 --perf-only -b 10
:: epochs mapped to the original layers (experimental according to ST)
python "%STEDGEAI_CORE_DIR%\scripts\ai_runner\examples\npu_profiler.py" -d serial:COM5:921600 -c "%N6_WORK_DIR%\st_ai_output\network.c" -b 10
:: board output against the model on the PC, report in st_ai_output\network_validate_report.txt
"%STEDGEAI%" validate -m model.tflite --target stm32n6 --mode target -d serial:COM5:921600
```

## 5. Reading the output

Excerpt from `checker.py` for the MLP:

```
 duration                       :   0.100 ms by sample (0.098/0.113/0.005)
 CPU cycles                     :   [27,284 45,097 7,304]

  c_id    m_id   type            dur (ms)        %     cumul  CPU cycles                   name
  0       -      epoch              0.073   73.46%    73.46%  [ 15,794  39,287   3,457 ]   EpochBlock_2
  1       -      epoch              0.019   19.05%    92.51%  [  8,125   4,982   2,075 ]   EpochBlock_3
  2       -      epoch              0.007    7.49%   100.00%  [  3,365     828   1,772 ]   EpochBlock_4
  HW: 56.6%, SW: 0.0%, SW ctrl: 43.4%
```

- **Epochs instead of layers:** the NPU processes the graph in epochs. One epoch can contain several
  original layers (e.g. Dense + ReLU). `npu_profiler.py` shows the mapping.
- **Three cycle values** per row: preparation, hardware (NPU computing), post-processing, each in
  MCU cycles at 800 MHz. `run_json.py` records the total per inference.
- **`HW` / `SW` / `SW ctrl`:** share in which the NPU computes, the CPU runs operators, or the CPU
  only drives the NPU. For very small models `SW ctrl` is high, because driving the NPU is
  expensive compared with the actual work.
- **Clocks:** the validation app runs in overdrive mode (MCU 800 MHz, NPU 1 GHz), see
  `device.attrs` in the output.
- **Flash and RAM in `results.json`** cover the model only (weights + network code, activations +
  network state), not the whole validation app as on the nRF5340.

## Pitfalls

### Board and USB

- **No COM port in Device Manager, LEDs on anyway:** wrong USB-C socket or a charge-only cable.
  See [section 3](#3-board-setup).
- **COM port present, but loading fails with only one cable:** the ST-LINK socket alone (CN9 on
  1-2, `STLK`) did not supply enough power for the STM32N6 with the validation app (CPU 800 MHz,
  NPU 1 GHz) in testing. With CN9 on 3-4 and no second cable the chip has no power at all.
  Solution: two cables as in section 3.
- **`make load` fails with "Access port is not valid":** the debugger cannot reach the chip.
  Usually power (see above) or another program holding the ST-LINK.
- **LD9 (COM) is solid red:** normal; the ST-LINK is enumerated and has no debug connection.
  It blinks red/green during a connection.
- **Connection fails or times out while measuring:** only one program may use the ST-LINK and the
  COM port at a time. Close STM32CubeIDE, the CubeProgrammer GUI, STM32Cube AI Studio and serial
  terminals first.
- **The board no longer answers after unplugging:** the validation app lived in RAM only. Run
  `make load` again (or simply `make measure`).
- **`validate` gets no answer:** without a baud rate `stedgeai` uses 115200, the validation app
  speaks 921600. Always pass `-d serial:COMx:921600`.
- **ST-LINK firmware update:** if an ST tool offers to update the ST-LINK, do not abort it and do not
  unplug the board meanwhile. The ST forum has reports of ST-LINKs bricked that way.

### Python

- **`Descriptors cannot be created directly` / "generated code is out of date":** ST's `*_pb2.py`
  files were generated with an old protoc and fail with protobuf >= 4. Hence `protobuf==3.20.3`.
- **`No module named 'colorama'`:** missing from ST's `requirements.txt`; included in ours.
- **`No module named 'stm_ai_runner'`** (by hand only): `PYTHONPATH` is not set in this window. Do
  **not** set it permanently, otherwise every other Python environment sees ST's modules.
  `run_json.py` adds the path itself.
- **msys2 or Chocolatey Python:** its `venv` creates `bin\` instead of `Scripts\`, so `activate`
  fails in `cmd`. uv avoids this with its own Python.

### Tools and paths

- **Other `make` or `arm-none-eabi-gcc` versions in `PATH`** (Chocolatey, msys2, older toolchains):
  do not disturb the loader, which takes the tools from CubeIDE via `cubeide_path`. By hand you get
  the first one in `PATH`; check with `where make arm-none-eabi-gcc`.
- **Do not use `setx PATH …`:** `setx` truncates values to 1024 characters, and the usual
  `setx PATH "%PATH%;…"` copies the whole system `Path` into the user `Path`. Edit `Path` only in
  the dialog "Edit environment variables for your account".
- **`config.json` is not read (`Bad Json syntax … at line: ï»¿{`):** the file was saved with a BOM,
  e.g. by Windows PowerShell 5.1 with `Set-Content -Encoding utf8`. `dk.sh` writes it without.
  Always use forward slashes `/` in it, also on Windows.
- **Loader starts slowly:** without further settings it searches all CubeIDE plugins for the tools,
  which can take over a minute. Listing the paths in `config.json` directly is faster (examples
  in `%STEDGEAI_CORE_DIR%\scripts\N6_scripts\config.json`).
- **Several CubeIDE versions installed:** the loader searches only below `cubeide_path` and takes
  the newest plugin version there.
- **`compiler "gcc:arm-none-eabi-gcc" is not in the PATH` during `analyze`/`generate`:** harmless.
  Only the code size estimate of the runtime library is missing; the analysis itself runs.

### ST Edge AI

- **`stedgeai` seems to hang** (e.g. at `PASS: 0%`): startup often takes ten seconds to over a
  minute, depending on the virus scanner. No reason to abort.
- **The loader reports failure only in its output:** it exits with code 0 even when loading failed.
  `dk.sh load` therefore checks for "Start operation achieved successfully"; otherwise a
  measurement would run against the previously loaded model.
- **The loader modifies the ST installation:** it copies `network.c` into
  `%STEDGEAI_CORE_DIR%\Projects\STM32N6570-DK\Applications\NPU_Validation\` and builds in
  `armgcc\build\` there. The project cannot simply be copied out, because it uses relative paths to
  the installation's `Middlewares\`. Consequences: the installation must be writable, and only
  **one** model can be loaded at a time.
- **The project name says "STM32N6570-DK"**, but it also covers the Nucleo: the board is selected
  with `--build-config N6-Nucleo` (or `N6-Nucleo-USB` for the faster USB channel, which needs a
  second cable on the other socket).
- **The Nucleo has no external RAM:** the NPU compiler's default memory map includes the
  Discovery kit's `hyperRAM`. Small and medium models (e.g. the MobileNets in this repo) fit into
  internal memory. If `generate` puts anything into `hyperRAM`, the model does not run on the
  Nucleo and `make generate` stops; it then needs a smaller model or an own memory map (`.mpool`).
- **ONNX:** The Dynamo exporter does not work (unsupported `Shape` and
  `Reshape` attributes), so the export uses `dynamo=False`.

## Further reading

The full ST documentation is installed under `%STEDGEAI_CORE_DIR%\Documentation\`, in particular:

- `stneuralart_getting_started.html`: the flow described here from ST's point of view, including
  troubleshooting
- `command_line_interface.html`: all `stedgeai` options
- `ai_runner_python_module.html`: the Python API used by `run_json.py`
- `stneuralart_operator_support.html` (online in the ST docs portal): which operators run on the NPU
