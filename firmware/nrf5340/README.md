# nRF5340 firmware

Zephyr / nRF Connect SDK v2.6.0 app that runs the INT8 model with TFLite Micro + CMSIS-NN
and reports per-layer cycle counts (DWT) over UART. The last line is machine-readable:

```
RESULT {"model_sha256": "795d8933…", "cycles": 61060, "latency_us": 477, "arena_used_bytes": 1252, "prediction": 1.8906}
```

`model_sha256` is embedded by `ble-cs codegen`. After a run, `sim` / `measure` call
[scripts/run_json.py](scripts/run_json.py) (in the `fw` container): it combines this line with
flash/RAM from `zephyr.elf` into a device-independent `build/<variant>/run.json`, which is piped
into `ble-cs results record`. That file is the only interface to the ML side.

One app, two variants:

| # | Variant | Build | Run |
|---|---------|-------|-----|
| 1 | Renode simulation, enlarged memory (8 MB RAM / 16 MB flash) | `make renode` → `build/renode` | `make sim` (+ record results) |
| 2 | Real nRF5340 DK (1 MB flash / 448 KB app RAM) | `make hil` → `build/hil` | `make measure` (flash + record results), or `make flash` + `make console` |

Everything builds in Docker (`fw` / `renode` services in the repo's `compose.yaml`), so
only Docker is needed on the host. Run from the repo root with `make -C firmware/nrf5340 <target>`.

Before the first build, generate the model sources (`src/ML_Models/`, `src/data.hpp`, gitignored):
`make -C ml all M=<model>` or, for an existing checkpoint, `make -C ml ckpt-to-nrf samples CKPT=checkpoints/<name>.pt`.

## Flashing from WSL (one-time setup)

Step-by-step guide with checks and troubleshooting (German): [docs/wsl_flashing_setup.md](../../docs/wsl_flashing_setup.md).

The DK's J-Link is passed from Windows to WSL with [usbipd-win](https://github.com/dorssel/usbipd-win):

1. Windows, admin PowerShell, once: `usbipd bind --busid <BUSID>` (see `usbipd list`, VID:PID `1366:1061`)
2. Windows, keep running: `usbipd attach --wsl --busid <BUSID> --auto-attach`
3. WSL: install SEGGER J-Link (`.deb`), Nordic's [nrf-udev](https://github.com/NordicSemiconductor/nrf-udev) rules,
   [nrfutil](https://www.nordicsemi.com/Products/Development-tools/nRF-Util) + `nrfutil install device`,
   and add yourself to `dialout` (`sudo usermod -aG dialout $USER`, then restart WSL).

Then `make flash [SNR=…] [VARIANT=…]` and `make console`. The app core's UART is interface `if02`.

## Layout

| Path | Purpose |
|------|---------|
| `west.yml` | Pins NCS v2.6.0 + TFLite Micro (used by the Docker image) |
| `prj.conf`, `Kconfig` | Shared config; `CONFIG_APP_TENSOR_ARENA_SIZE` (default 128 KiB) |
| `snippets/renode/` | Variant #1: memory overlay + 3 MB arena (`-S renode`) |
| `renode/` | Renode platform (`.repl`, must match the renode snippet) and script (`.resc`) |
| `Dockerfile` | Stages `build` (SDK + toolchain) and `renode` |

## Troubleshooting

- **Strange CMake errors after moving/deleting config files:** the build directory caches
  the old file list. Run `make clean` (or delete `build/<variant>`).
- **`region 'FLASH' overflowed` in `make hil`:** the model does not fit on the real chip.
  Use a smaller model or variant #1.
