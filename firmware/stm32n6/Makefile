# Entry points of the STM32N6 (NUCLEO-N657X0-Q) measurement.
# Runs on the host (WSL). The ST tools are Windows programs, called through WSL interop by
# scripts/dk.sh inside a Windows work folder; nothing here runs in Docker (README.md explains why).
#
#   make setup      create the Python venv and config.json in the work folder (once)
#   make generate   compile model/model.{tflite,onnx} for the NPU (stedgeai generate), no board needed
#   make load       generate + build the ST validation app with it and start it on the board
#   make measure    load + profile on the board, record results -> ml/outputs/<checkpoint>/results.json
#
# Parameters (Windows paths with forward slashes):
#   STEDGEAI_DIR  ST Edge AI Core installation
#   CUBEIDE_DIR   folder containing stm32cubeide.exe (make, Arm GCC and ST-LINK tools come from it)
#   N6_WORK_DIR   work folder on a local Windows drive: venv, config.json, compiled model
#   CUBEPROG_CLI  standalone STM32_Programmer_CLI.exe; used if it exists, else the CubeIDE one
#
# Usage from the repo root: make -C firmware/stm32n6 <target>

STEDGEAI_DIR ?= C:/ST/STEdgeAI/4.0
CUBEIDE_DIR  ?= C:/ST/STM32CubeIDE_1.18.1/STM32CubeIDE
N6_WORK_DIR  ?= C:/work/n6
CUBEPROG_CLI ?= C:/Program Files/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI.exe
export STEDGEAI_DIR CUBEIDE_DIR N6_WORK_DIR CUBEPROG_CLI

# Folder of this Makefile -> works from any current directory
HERE     := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
COMPOSE  := docker compose -f $(HERE)../../compose.yaml
DK       := $(HERE)scripts/dk.sh
RUN_JSON := $(HERE)build/run.json

# make -C ml to-stm32n6 [FORMAT=onnx] leaves exactly one model file here. Its format decides the
# device name in results.json, so a TFLite and an ONNX measurement never overwrite each other.
MODEL  := $(or $(wildcard $(HERE)model/model.*),$(HERE)model/model.tflite)
TARGET := stm32n6$(if $(filter %.onnx,$(MODEL)),-onnx)
export TARGET

.PHONY: setup generate load measure

setup:
	$(DK) setup

generate:
	$(DK) generate $(MODEL)

load: generate
	$(DK) load

# run.json goes to the ml container on stdin, which links it to its model via the sha256
measure: load
	$(DK) measure $(MODEL) $(RUN_JSON)
	$(COMPOSE) run --rm -T ml results record < $(RUN_JSON)
