# Entry points of the ML pipeline. Runs on the host (WSL/Linux); every step runs in Docker
# (services ml / ml-gpu from compose.yaml in the repo root). Paths are container paths below /app.
#
# Usage from the repo root: make -C ml <target> M=<model>      (`make -C ml help` lists everything)

# ---- Parameters (override on the command line: make -C ml all M=mobilenet_v3_micro_2d) ----
# (Comments deliberately NOT after the values: make would keep the spaces in front of them)
# M:         model name from `make -C ml models`
# GPU:       0 = train on the CPU (service ml, default), 1 = on an NVIDIA GPU (service ml-gpu)
# DATA:      dataset folder in ml/data/: sample (public, default) or full (not in the repo)
# EVAL:      0 = export without MAE evaluation (faster)
# CKPT:      input of export/tflite/onnx
# CKPT_NAME: folder in outputs/ = checkpoint path inside checkpoints/ without .pt
#            (checkpoints/mobilenet/bad_simd.pt -> outputs/mobilenet/bad_simd/{fp32_full,int8_full,int8_qdq}/)
# TFLITE:    input of to-nrf
# TARGET:    device whose measurements `table` shows (nrf5340dk, renode, stm32n6, stm32n6-onnx)
# FORMAT:    model format for the STM32N6: tflite (int8_full) or onnx (int8_qdq)
M      ?= mobilenet_v3_micro_2d
GPU    ?= 0
DATA   ?= sample
EVAL   ?= 1
CKPT   ?= checkpoints/$(M).pt
CKPT_NAME ?= $(patsubst checkpoints/%.pt,%,$(CKPT))
TFLITE ?= outputs/$(CKPT_NAME)/int8_full/model.tflite
TARGET ?= nrf5340dk
FORMAT ?= tflite

# ---- Internals ----
HERE    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
COMPOSE := docker compose -f $(HERE)../compose.yaml
RUN     := $(COMPOSE) run --rm -e EDGE_IB_DATA_DIR=/app/data/$(DATA)
ML      := $(RUN) ml
DEV     := $(COMPOSE) run --rm -T ml-dev
TRAIN   := $(RUN) $(if $(filter 1,$(GPU)),ml-gpu,ml)
NOEVAL  := $(if $(filter 0,$(EVAL)),--no-eval)
# Host paths: the ml package does not know the firmware, only this Makefile copies into it
NRF_SRC := $(HERE)../firmware/nrf5340/src
STM32N6_DIR   := $(HERE)../firmware/stm32n6
STM32N6_MODEL := $(STM32N6_DIR)/model
# Exported file per FORMAT: full INT8 TFLite, or ONNX with INT8 inside and float I/O
STM32N6_SRC   := $(if $(filter onnx,$(FORMAT)),outputs/$(CKPT_NAME)/int8_qdq/model.onnx,$(TFLITE))
STEP    := $(MAKE) --no-print-directory -f $(lastword $(MAKEFILE_LIST))
# Mounted by the ml service. Created here as your user: Compose would create them as root
# (create_host_path: false in compose.yaml forbids that), and the container could not write.
DIRS    := $(HERE)checkpoints $(HERE)outputs

.PHONY: help models images train export tflite onnx to-nrf samples to-stm32n6 table format lint check train-export ckpt-to-nrf all ckpt-to-stm32n6 all-stm32n6

help:
	@echo "Single steps:"
	@echo "  train        train model M                    -> ml/$(CKPT)"
	@echo "  export       TFLite + ONNX (FP32 + INT8)      -> ml/outputs/$(CKPT_NAME)/"
	@echo "  tflite       TFLite only (MCU)                -> ml/$(TFLITE)"
	@echo "  onnx         ONNX only (STM32N6, Jetson)      -> ml/outputs/$(CKPT_NAME)/*/model.onnx"
	@echo "  to-nrf       INT8 TFLite into the firmware    -> firmware/nrf5340/src/ML_Models/"
	@echo "  samples      test inputs into the firmware    -> firmware/nrf5340/src/data.hpp"
	@echo "  to-stm32n6   INT8 model for the STM32N6       -> firmware/stm32n6/model/ (FORMAT=$(FORMAT))"
	@echo "Flows:"
	@echo "  train-export train + export"
	@echo "  ckpt-to-nrf  tflite + to-nrf (existing checkpoint)"
	@echo "  all          train + tflite + to-nrf + samples (up to a buildable firmware)"
	@echo "  ckpt-to-stm32n6  export + to-stm32n6 + measure on the board (existing checkpoint)"
	@echo "  all-stm32n6      train + ckpt-to-stm32n6 (model to results.json in one go)"
	@echo "Results:"
	@echo "  table        all results.json as a Markdown table (TARGET=$(TARGET))"
	@echo "Code quality:"
	@echo "  format       format code and apply safe lint fixes (changes files)"
	@echo "  lint         report lint findings"
	@echo "  check        formatting + lint, changes nothing (run before committing)"
	@echo "Other: models, images"
	@echo "Parameters: M=$(M) GPU=$(GPU) DATA=$(DATA) EVAL=$(EVAL) FORMAT=$(FORMAT) [CKPT=...] [TFLITE=...]"

# Every target that starts the ml service needs the mounted folders. Order-only (|): only their
# existence counts, not their timestamp. mkdir -p never touches an existing folder.
models train export tflite onnx to-nrf samples table: | $(DIRS)

$(DIRS):
	mkdir -p $@

models:
	$(ML) models

# After changes to ml/src, the Dockerfile, pyproject.toml or uv.lock, otherwise old code runs.
# The CUDA image (several GB) is only built with GPU=1.
images:
	$(COMPOSE) build ml ml-dev $(if $(filter 1,$(GPU)),ml-gpu)

# ---- Single steps ----
train:
	$(TRAIN) train -m $(M) -o $(CKPT)

export:
	$(ML) export $(CKPT) $(NOEVAL)

tflite:
	$(ML) export $(CKPT) --no-onnx $(NOEVAL)

onnx:
	$(ML) export $(CKPT) --no-tflite $(NOEVAL)

# Codegen writes next to the .tflite (.../int8_full/codegen/), then the files are copied into the firmware
to-nrf:
	$(ML) codegen model $(TFLITE)
	mkdir -p $(NRF_SRC)/ML_Models
	cp $(HERE)$(dir $(TFLITE))codegen/model_data.h $(HERE)$(dir $(TFLITE))codegen/model_data.cc $(NRF_SRC)/ML_Models/

samples:
	$(ML) codegen samples
	cp $(HERE)outputs/samples/data.hpp $(NRF_SRC)/

# The STM32N6 toolchain compiles the model file itself (stedgeai), so no codegen is needed.
# Exactly one model file stays in the folder: the firmware side reads the format from it.
to-stm32n6:
	mkdir -p $(STM32N6_MODEL)
	rm -f $(STM32N6_MODEL)/model.*
	cp $(HERE)$(STM32N6_SRC) $(STM32N6_MODEL)/model$(suffix $(STM32N6_SRC))

table:
	@$(COMPOSE) --progress quiet run --rm -T ml results table --target $(TARGET)

# ---- Code quality (ruff from uv.lock, configured in pyproject.toml) ----
format:
	$(DEV) ruff format .
	$(DEV) ruff check --fix .

lint:
	$(DEV) ruff check .

check:
	$(DEV) ruff format --check .
	$(DEV) ruff check .

# ---- Flows (run one after another, even with make -j) ----
train-export:
	$(STEP) train
	$(STEP) export

ckpt-to-nrf:
	$(STEP) tflite
	$(STEP) to-nrf

# Change export file (here tflite) or firmware target (here to-nrf) for your specific hardware
all:
	$(STEP) train
	$(STEP) tflite
	$(STEP) to-nrf
	$(STEP) samples

# Export only the format the board needs: the targets are named like FORMAT (tflite / onnx)
ckpt-to-stm32n6:
	$(STEP) $(FORMAT)
	$(STEP) to-stm32n6
	$(MAKE) --no-print-directory -C $(STM32N6_DIR) measure

all-stm32n6:
	$(STEP) train
	$(STEP) ckpt-to-stm32n6
