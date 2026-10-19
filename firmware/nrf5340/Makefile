# Einstiegspunkte für die nRF5340-Firmware.
# Läuft auf dem Host (WSL); Build und Simulation passieren in Docker
# (Dienste fw / renode aus compose.yaml im Repo-Root).
#
#   make renode   Variante #1 (Renode) bauen     -> build/renode
#   make sim      Variante #1 bauen + simulieren
#   make hil      Variante #2 (echtes DK) bauen  -> build/hil
#   make flash    build/$(VARIANT) aufs DK flashen + reset (nativ in WSL, nrfutil)
#   make console  UART-Konsole des App-Kerns mitlesen (Ctrl+C beendet)
#   make clean    alle Build-Ordner löschen
#
# Parameter: SNR=<J-Link-Seriennummer> (Default 601er DK), VARIANT=<Build-Ordner> (Default hil)
#
# Aufruf aus dem Repo-Root: make -C firmware/nrf5340 <target>

BOARD := nrf5340dk_nrf5340_cpuapp

# Ordner dieses Makefiles -> funktioniert unabhängig vom aktuellen Verzeichnis
HERE    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
COMPOSE := docker compose -f $(HERE)../../compose.yaml

# Welches DK (Seriennummer auf dem J-Link-Aufkleber / `nrfutil device list`)
SNR     ?= 1050025601
VARIANT ?= hil
# Konsole des App-Kerns = Interface 02 (vcom 1); if00 gehört zum Netzkern.
# by-id statt ttyACMx: bleibt gleich, egal in welcher Reihenfolge DKs erkannt werden.
UART    := /dev/serial/by-id/usb-SEGGER_J-Link_00$(SNR)-if02

# -p auto: pristine nur wenn nötig (z. B. anderes Board)
WEST_BUILD := $(COMPOSE) run --rm fw west build -b $(BOARD) -p auto --no-sysbuild

.PHONY: renode sim hil flash console clean

renode:
	$(WEST_BUILD) -d build/renode -S renode

sim: renode
	$(COMPOSE) run --rm renode

hil:
	$(WEST_BUILD) -d build/hil

# Löscht nur die Flash-Bereiche, die das Image belegt (Netzkern-Image bleibt unberührt)
flash:
	nrfutil device program --serial-number $(SNR) \
		--firmware $(HERE)build/$(VARIANT)/zephyr/zephyr.hex \
		--options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE
	nrfutil device reset --serial-number $(SNR)

console:
	stty -F $(UART) 115200 raw -echo
	cat $(UART)

clean:
	rm -rf $(HERE)build
