# nRF5340 DK aus WSL flashen: Einrichtung

Stand: 2026-10-15 · getestet mit Windows 11 + WSL2 (Ubuntu 24.04), usbipd-win 5.3, SEGGER J-Link V9.82,
nrfutil 8.2.1 (`device` 2.22.0), nRF5340 DK (PCA10095)

Ziel: Firmware bauen, flashen und die UART-Ausgabe lesen – alles aus WSL, ohne Windows-Werkzeuge
und ohne `sudo`. Damit kann auch ein Skript bzw. ein Agent unbeaufsichtigt flashen.

```
Windows                           WSL2
───────                           ────
DK (J-Link, USB) ── usbipd ──▶    /dev/bus/usb/…        ← nrfutil flasht darüber (J-Link-Bibliothek)
                                  /dev/serial/by-id/…   ← UART (cat / pyserial)
```

Der Build selbst läuft in Docker (siehe `firmware/nrf5340/README.md`) und braucht nichts davon.

---

## 1. Windows: USB-Gerät an WSL weiterreichen (usbipd-win)

WSL2 ist eine virtuelle Maschine und sieht USB-Geräte nicht von selbst. **usbipd-win** reicht sie
über das Netzwerkprotokoll USB/IP an WSL weiter.

### 1.1 Installieren (einmalig)

```powershell
winget install usbipd
```

### 1.2 Gerät freigeben (einmalig, **PowerShell als Administrator**)

```powershell
usbipd list                          # J-Link suchen: VID:PID 1366:1061
usbipd bind --hardware-id 1366:1061  # oder: --busid <BUSID> aus der Liste
usbipd list                          # STATE muss jetzt "Shared" sein
```

`bind` gibt das Gerät nur frei – es hängt danach noch an Windows. Die Freigabe bleibt über Neustarts erhalten.

### 1.3 An WSL anhängen

Zum Ausprobieren (Fenster muss offen bleiben):

```powershell
usbipd attach --wsl --hardware-id 1366:1061 --auto-attach
```

Dauerhaft (startet versteckt bei jeder Anmeldung, normale PowerShell genügt):

```powershell
$exe = "C:\Program Files\usbipd-win\usbipd.exe"
$a = New-ScheduledTaskAction -Execute "powershell.exe" `
     -Argument "-WindowStyle Hidden -Command & '$exe' attach --wsl --hardware-id 1366:1061 --auto-attach"
$t = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "usbipd-attach-jlink" -Action $a -Trigger $t -Settings $s
Start-ScheduledTask -TaskName "usbipd-attach-jlink"
# Entfernen: Unregister-ScheduledTask -TaskName usbipd-attach-jlink
```

Warum so:

- **`--auto-attach`**: Der J-Link meldet sich beim Flashen/Reset kurz ab und wieder an; ohne diese Option
  wäre er danach aus WSL verschwunden. Hängt das Gerät auch nach `wsl --terminate` wieder an.
- **`--hardware-id` statt `--busid`**: Die BUSID hängt am USB-Port (anders angedockt → andere BUSID).
  Die Hardware-ID ist für jeden SEGGER J-Link gleich.
- **Solange das DK an WSL hängt, sehen Windows-Programme es nicht** (nRF Connect for VS Code, nrfjprog,
  COM-Ports). Zurück zu Windows: Aufgabe/Fenster beenden oder `usbipd detach --hardware-id 1366:1061`.

### 1.4 Optional: Laptop über Nacht wach halten

```powershell
powercfg /change standby-timeout-ac 0   # im Netzbetrieb nie in den Ruhezustand
```

---

## 2. WSL: Werkzeuge installieren

### 2.1 Prüfen, ob das DK angekommen ist

```bash
lsusb | grep -i segger          # → ID 1366:1061 SEGGER J-Link
ls -l /dev/serial/by-id/        # → usb-SEGGER_J-Link_00<SNR>-if00 und -if02
```

### 2.2 SEGGER J-Link (Bibliothek, über die nrfutil flasht)

Version = die, die nrfutil als getestet meldet (`nrfutil device --version`).
Das POST-Feld bedeutet: Man akzeptiert die SEGGER-Lizenz.

```bash
cd /tmp
curl -fL -o JLink.deb -X POST \
  -d "accept_license_agreement=accepted&submit=Download+software" \
  https://www.segger.com/downloads/jlink/JLink_Linux_V982_x86_64.deb
sudo apt install -y ./JLink.deb
```

Liefert `curl` keine `.deb`: im Browser unter segger.com/downloads/jlink herunterladen (Linux, 64-bit DEB).

### 2.3 nrf-udev (Zugriff ohne `sudo`)

USB-Geräte gehören standardmäßig root. Die Regeln geben Nordic-/SEGGER-Geräte für normale Nutzer frei.

```bash
curl -fL -o nrf-udev.deb https://github.com/NordicSemiconductor/nrf-udev/releases/download/v1.0.1/nrf-udev_1.0.1-all.deb
sudo apt install -y ./nrf-udev.deb
```

Wirkt erst, wenn sich das Gerät neu anmeldet (DK aus-/einstecken oder `attach` neu starten).

### 2.4 nrfutil + `device`-Befehl (ohne `sudo`, liegt im Home)

```bash
mkdir -p ~/.local/bin
curl -fL -o ~/.local/bin/nrfutil \
  https://files.nordicsemi.com/artifactory/swtools/external/nrfutil/executables/x86_64-unknown-linux-gnu/nrfutil
chmod +x ~/.local/bin/nrfutil
nrfutil install device
```

`nrfutil` ist Nordics aktuelles Werkzeug (Nachfolger des abgekündigten `nrfjprog`) und kann auch den
Netzkern des nRF5340 flashen.

### 2.5 Serielle Ports lesen dürfen (Gruppe `dialout`)

```bash
sudo usermod -aG dialout $USER
```

Gilt erst nach neuer Anmeldung. In WSL: in PowerShell `wsl --terminate Ubuntu-24.04`, dann neues Terminal.
Prüfen: `id -nG` enthält `dialout`.

---

## 3. Prüfen

```bash
nrfutil device list
```

Erwartet (wichtig: `jlink` in `Traits`, keine Warnung „No SEGGER J-Link install found“):

```
1050025601
Product         J-Link
Board version   PCA10095
Ports           /dev/ttyACM0, vcom: 0
                /dev/ttyACM1, vcom: 1
Traits          devkit, jlink, seggerUsb, serialPorts, usb
```

Erster Test mit Zephyrs hello_world (aus dem Repo-Root):

```bash
docker compose run --rm fw west build -b nrf5340dk_nrf5340_cpuapp \
  -s /ncs/zephyr/samples/hello_world -d build/hello -p auto --no-sysbuild
make -C firmware/nrf5340 console &                  # erst lauschen …
make -C firmware/nrf5340 flash VARIANT=hello        # … dann flashen (inkl. Reset)
# → *** Booting nRF Connect SDK … ***
#   Hello World! nrf5340dk_nrf5340_cpuapp
kill %1
```

---

## 4. Benutzung

```bash
make -C firmware/nrf5340 hil                            # Variante #2 bauen (Docker)
make -C firmware/nrf5340 flash [SNR=…] [VARIANT=hil]    # flashen + Reset (~1,6 s)
make -C firmware/nrf5340 console [SNR=…]                # UART des App-Kerns mitlesen
```

Direkt mit nrfutil:

```bash
nrfutil device program --serial-number 1050025601 --firmware build/hil/zephyr/zephyr.hex \
  --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE
nrfutil device reset --serial-number 1050025601
nrfutil --json device program …     # maschinenlesbarer Fortschritt für Skripte
```

---

## 5. Stolperfallen

| Symptom | Ursache / Lösung |
|---|---|
| UART bleibt leer | Falscher Port: Die Konsole des **App-Kerns** ist `…-if02` (vcom 1). `if00` (vcom 0) gehört zum Netzkern. |
| Anfang der Ausgabe fehlt | UART puffert nicht. Erst den Port öffnen, **dann** flashen/resetten. |
| `nrfutil device program` gibt nichts aus | Normal ohne Terminal (z. B. in Skripten): Fortschritt nur bei TTY. Erfolg über Exit-Code oder `--json`. |
| `Permission denied` auf `/dev/ttyACM*` | Nicht in `dialout` oder WSL nach `usermod` nicht neu gestartet (2.5). |
| Flashen nur mit `sudo` möglich | nrf-udev fehlt oder Gerät hat sich seit der Installation nicht neu angemeldet (2.3). |
| „No SEGGER J-Link install found“ | J-Link fehlt (2.2). `nrfutil device list` findet das DK trotzdem – flashen geht aber nicht. |
| DK fehlt in WSL nach Umstecken/Andocken | `attach` lief mit `--busid` → mit `--hardware-id` neu starten (1.3). |
| Windows-Tools finden das DK nicht | Es hängt gerade an WSL → `usbipd detach --hardware-id 1366:1061`. |

**Offen für Variante #3 (zwei DKs):** Beide J-Links haben dieselbe Hardware-ID `1366:1061`.
Ob `attach --hardware-id` dann beide anhängt, ist noch nicht getestet. Getrennt werden sie über
die Seriennummer (`SNR=`) bzw. `/dev/serial/by-id/usb-SEGGER_J-Link_00<SNR>-if02`.
