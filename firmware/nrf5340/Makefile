# Entry points of the nRF5340 firmware.
# Runs on the host (WSL/Linux); building and simulating happen in Docker
# (services fw / renode from compose.yaml in the repo root), the DK is accessed natively.
#
#   make renode   build variant #1 (Renode)          -> build/renode
#   make sim      build #1 + simulate, record results -> ml/outputs/<checkpoint>/results.json
#   make hil      build variant #2 (real DK)         -> build/hil
#   make flash    flash build/$(VARIANT) to the DK and reset it
#   make measure  build #2, flash, capture the UART, record results -> results.json
#   make console  read the app core UART (Ctrl+C to stop)
#   make clean    delete all build folders
#
# Parameters: SNR=<J-Link serial number> (default: first connected DK), VARIANT=<build folder> (default hil)
#
# Usage from the repo root: make -C firmware/nrf5340 <target>

BOARD := nrf5340dk_nrf5340_cpuapp

# Folder of this Makefile -> works from any current directory
HERE    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
COMPOSE := docker compose -f $(HERE)../../compose.yaml

VARIANT ?= hil
# Board detection, flashing and UART capture live in scripts/dk.sh (runs natively in WSL).
# Optional SNR=<J-Link serial number> selects one of several connected DKs.
DK      := $(HERE)scripts/dk.sh
export SNR

# -p auto: pristine build only when needed (e.g. a different board)
WEST_BUILD := $(COMPOSE) run --rm fw west build -b $(BOARD) -p auto --no-sysbuild

# Record a firmware run in results.json: $(call RECORD,<device>,<build folder>)
# 1. fw container: uart.log + zephyr.elf -> build/<folder>/run.json (device-independent format)
# 2. ml container reads run.json from stdin and links it to its model via the sha256
RECORD = $(COMPOSE) run --rm -T fw python3 scripts/run_json.py $(1) build/$(2) && \
	$(COMPOSE) run --rm -T ml results record < $(HERE)build/$(2)/run.json

.PHONY: renode sim hil flash measure console clean

renode:
	$(WEST_BUILD) -d build/renode -S renode

sim: renode
	$(COMPOSE) run --rm -T renode | tee $(HERE)build/renode/uart.log
	$(call RECORD,renode,renode)

hil:
	$(WEST_BUILD) -d build/hil

flash:
	$(DK) flash $(HERE)build/$(VARIANT)/zephyr/zephyr.hex

measure: hil
	$(DK) measure $(HERE)build/hil/zephyr/zephyr.hex $(HERE)build/hil/uart.log
	$(call RECORD,nrf5340dk,hil)

console:
	$(DK) console

clean:
	rm -rf $(HERE)build
