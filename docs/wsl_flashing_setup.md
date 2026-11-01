# Flashing the nRF5340 DK from WSL: setup

Status: 2026-10-15 · tested with Windows 11 + WSL2 (Ubuntu 24.04), usbipd-win 5.3, SEGGER J-Link V9.82,
nrfutil 8.2.1 (`device` 2.22.0), nRF5340 DK (PCA10095)

Goal: build firmware, flash it and read the UART output, all from WSL, without Windows tools
and without `sudo`. This also lets a script or an agent flash unattended.

```
Windows                           WSL2
───────                           ────
DK (J-Link, USB) ── usbipd ──▶    /dev/bus/usb/…        ← nrfutil flashes through this (J-Link library)
                                  /dev/serial/by-id/…   ← UART (cat / pyserial)
```

The build itself runs in Docker (see `firmware/nrf5340/README.md`) and needs none of this.

---

## 1. Windows: pass the USB device through to WSL (usbipd-win)

WSL2 is a virtual machine and does not see USB devices on its own. **usbipd-win** passes them
to WSL over the USB/IP network protocol.

### 1.1 Install (once)

```powershell
winget install usbipd
```

### 1.2 Share the device (once, **PowerShell as administrator**)

```powershell
usbipd list                          # look for the J-Link: VID:PID 1366:1061
usbipd bind --hardware-id 1366:1061  # or: --busid <BUSID> from the list
usbipd list                          # STATE must now be "Shared"
```

`bind` only shares the device; it is still attached to Windows afterwards. The share survives reboots.

### 1.3 Attach to WSL

To try it out (the window must stay open):

```powershell
usbipd attach --wsl --hardware-id 1366:1061 --auto-attach
```

Permanently (starts hidden at every logon, a normal PowerShell is enough):

```powershell
$exe = "C:\Program Files\usbipd-win\usbipd.exe"
$a = New-ScheduledTaskAction -Execute "powershell.exe" `
     -Argument "-WindowStyle Hidden -Command & '$exe' attach --wsl --hardware-id 1366:1061 --auto-attach"
$t = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "usbipd-attach-jlink" -Action $a -Trigger $t -Settings $s
Start-ScheduledTask -TaskName "usbipd-attach-jlink"
# Remove: Unregister-ScheduledTask -TaskName usbipd-attach-jlink
```

Why this way:

- **`--auto-attach`**: The J-Link briefly disconnects and reconnects during flashing/reset; without this
  option it would be gone from WSL afterwards. It also re-attaches the device after `wsl --terminate`.
- **`--hardware-id` instead of `--busid`**: The BUSID depends on the USB port (docked differently → different
  BUSID). The hardware ID is the same for every SEGGER J-Link.
- **While the DK is attached to WSL, Windows programs do not see it** (nRF Connect for VS Code, nrfjprog,
  COM ports). Back to Windows: end the task/window or `usbipd detach --hardware-id 1366:1061`.

### 1.4 Optional: keep the laptop awake overnight

```powershell
powercfg /change standby-timeout-ac 0   # never sleep while on AC power
```

---

## 2. WSL: install the tools

### 2.1 Check that the DK has arrived

```bash
lsusb | grep -i segger          # → ID 1366:1061 SEGGER J-Link
ls -l /dev/serial/by-id/        # → usb-SEGGER_J-Link_00<SNR>-if00 and -if02
```

### 2.2 SEGGER J-Link (library nrfutil flashes through)

Version = the one nrfutil reports as tested (`nrfutil device --version`).
The POST field means: you accept the SEGGER license.

```bash
cd /tmp
curl -fL -o JLink.deb -X POST \
  -d "accept_license_agreement=accepted&submit=Download+software" \
  https://www.segger.com/downloads/jlink/JLink_Linux_V982_x86_64.deb
sudo apt install -y ./JLink.deb
```

If `curl` does not return a `.deb`: download it in the browser at segger.com/downloads/jlink (Linux, 64-bit DEB).

### 2.3 nrf-udev (access without `sudo`)

USB devices belong to root by default. These rules open Nordic/SEGGER devices to normal users.

```bash
curl -fL -o nrf-udev.deb https://github.com/NordicSemiconductor/nrf-udev/releases/download/v1.0.1/nrf-udev_1.0.1-all.deb
sudo apt install -y ./nrf-udev.deb
```

Takes effect only once the device reconnects (unplug/replug the DK or restart `attach`).

### 2.4 nrfutil + `device` command (no `sudo`, lives in your home)

```bash
mkdir -p ~/.local/bin
curl -fL -o ~/.local/bin/nrfutil \
  https://files.nordicsemi.com/artifactory/swtools/external/nrfutil/executables/x86_64-unknown-linux-gnu/nrfutil
chmod +x ~/.local/bin/nrfutil
nrfutil install device
```

`nrfutil` is Nordic's current tool (successor of the deprecated `nrfjprog`) and can also flash the
network core of the nRF5340.

### 2.5 Permission to read serial ports (group `dialout`)

```bash
sudo usermod -aG dialout $USER
```

Takes effect only after a new login. In WSL: run `wsl --terminate Ubuntu-24.04` in PowerShell, then open a new terminal.
Check: `id -nG` contains `dialout`.

---

## 3. Verify

```bash
nrfutil device list
```

Expected (important: `jlink` in `Traits`, no warning "No SEGGER J-Link install found"):

```
<SNR>
Product         J-Link
Board version   PCA10095
Ports           /dev/ttyACM0, vcom: 0
                /dev/ttyACM1, vcom: 1
Traits          devkit, jlink, seggerUsb, serialPorts, usb
```

First test with Zephyr's hello_world (from the repo root):

```bash
docker compose run --rm fw west build -b nrf5340dk_nrf5340_cpuapp \
  -s /ncs/zephyr/samples/hello_world -d build/hello -p auto --no-sysbuild
make -C firmware/nrf5340 console &                  # listen first …
make -C firmware/nrf5340 flash VARIANT=hello        # … then flash (incl. reset)
# → *** Booting nRF Connect SDK … ***
#   Hello World! nrf5340dk_nrf5340_cpuapp
kill %1
```

---

## 4. Usage

```bash
make -C firmware/nrf5340 hil                            # build variant #2 (Docker)
make -C firmware/nrf5340 flash [SNR=…] [VARIANT=hil]    # flash + reset (~1.6 s)
make -C firmware/nrf5340 console [SNR=…]                # read the app core UART
make -C firmware/nrf5340 measure [SNR=…]                # build, flash, capture UART, record results
```

Directly with nrfutil:

```bash
# <SNR> = serial number from `nrfutil device list` (also printed on the J-Link sticker)
nrfutil device program --serial-number <SNR> --firmware build/hil/zephyr/zephyr.hex \
  --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE
nrfutil device reset --serial-number <SNR>
nrfutil --json device program …     # machine-readable progress for scripts
```

---

## 5. Pitfalls

| Symptom | Cause / fix |
|---|---|
| UART stays empty | Wrong port: the **app core** console is `…-if02` (vcom 1). `if00` (vcom 0) belongs to the network core. |
| Beginning of the output is missing | The UART does not buffer. Open the port first, **then** flash/reset. |
| `nrfutil device program` prints nothing | Normal without a terminal (e.g. in scripts): progress only on a TTY. Check the exit code or use `--json`. |
| `Permission denied` on `/dev/ttyACM*` | Not in `dialout`, or WSL not restarted after `usermod` (2.5). |
| Flashing only works with `sudo` | nrf-udev missing, or the device has not reconnected since the installation (2.3). |
| "No SEGGER J-Link install found" | J-Link missing (2.2). `nrfutil device list` still finds the DK, but flashing fails. |
| DK missing in WSL after replugging/docking | `attach` ran with `--busid` → restart it with `--hardware-id` (1.3). |
| Windows tools do not find the DK | It is currently attached to WSL → `usbipd detach --hardware-id 1366:1061`. |

**Several DKs at once (not tested):** All J-Links share the hardware ID `1366:1061`;
`attach --hardware-id` probably attaches only one. Then create one task per DK with `--busid`
(each DK must then always use the same USB port). Flashing/UART already tell the DKs apart by
serial number (`SNR=`, `/dev/serial/by-id/usb-SEGGER_J-Link_00<SNR>-if02`).
