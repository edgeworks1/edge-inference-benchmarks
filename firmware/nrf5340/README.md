# nRF5340 firmware

Zephyr / nRF Connect SDK v2.6.0 app that runs the INT8 model with TFLite Micro + CMSIS-NN
and reports per-layer cycle counts (DWT) over UART.

One app, three variants:

| # | Variant | Build | Run |
|---|---------|-------|-----|
| 1 | Renode simulation, enlarged memory (8 MB RAM / 16 MB flash) | `make renode` → `build/renode` | `make sim` |
| 2 | Real nRF5340 DK (1 MB flash / 448 KB app RAM) | `make hil` → `build/hil` | `make flash` + `make console` |
| 3 | Two DKs doing BLE distance measurement + inference | not wired up yet (`snippets/ble-ranging/`) | – |

Everything builds in Docker (`fw` / `renode` services in the repo's `compose.yaml`), so
only Docker is needed on the host. Run from the repo root with `make -C firmware/nrf5340 <target>`.

Before the first build, generate the model sources with `ble-cs codegen` (`src/ML_Models/`,
`src/data.hpp` – gitignored).

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
| `snippets/ble-ranging/` | Variant #3: BLE/DM config (`-S ble-ranging`, untested) |
| `sysbuild/`, `child_image/`, `Kconfig.sysbuild` | Variant #3: network-core image config. Must stay in the app root (NCS looks for them there); inactive without Bluetooth |
| `renode/` | Renode platform (`.repl`, must match the renode snippet) and script (`.resc`) |
| `Dockerfile` | Stages `build` (SDK + toolchain) and `renode` |

## Troubleshooting

- **Strange CMake errors after moving/deleting config files:** the build directory caches
  the old file list. Run `make clean` (or delete `build/<variant>`).
- **`region 'FLASH' overflowed` in `make hil`:** the model does not fit on the real chip.
  Use a smaller model or variant #1.
