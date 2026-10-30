# Entry points of the ML pipeline. Runs on the host (WSL/Linux); every step runs in Docker
# (services ml / ml-gpu from compose.yaml in the repo root). Paths are container paths below /app.
#
# Usage from the repo root: make -C ml <target> M=<model>      (`make -C ml help` lists everything)

# ---- Parameters (override on the command line: make -C ml all M=mobilenet_v3_micro_2d) ----
# (Comments deliberately NOT after the values: make would keep the spaces in front of them)
# M:         model name from `make -C ml models`
# GPU:       1 = train on the GPU (service ml-gpu), 0 = on the CPU (service ml)
# EVAL:      0 = export without MAE evaluation (faster)
# CKPT:      input of export/tflite/onnx
# CKPT_NAME: folder in outputs/ = checkpoint path inside checkpoints/ without .pt
#            (checkpoints/mobilenet/bad_simd.pt -> outputs/mobilenet/bad_simd/{fp32_full,int8_full,int8_qdq}/)
# TFLITE:    input of to-nrf
# TARGET:    device whose measurements `table` shows (nrf5340dk or renode)
M      ?= mobilenet_v3_micro_2d
GPU    ?= 1
EVAL   ?= 1
CKPT   ?= checkpoints/$(M).pt
CKPT_NAME ?= $(patsubst checkpoints/%.pt,%,$(CKPT))
TFLITE ?= outputs/$(CKPT_NAME)/int8_full/model.tflite
TARGET ?= nrf5340dk

# ---- Internals ----
HERE    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
COMPOSE := docker compose -f $(HERE)../compose.yaml
ML      := $(COMPOSE) run --rm ml
DEV     := $(COMPOSE) run --rm -T ml-dev
TRAIN   := $(COMPOSE) run --rm $(if $(filter 1,$(GPU)),ml-gpu,ml)
NOEVAL  := $(if $(filter 0,$(EVAL)),--no-eval)
# Host path: the ml package does not know the firmware, only this Makefile copies into it
NRF_SRC := $(HERE)../firmware/nrf5340/src
STEP    := $(MAKE) --no-print-directory -f $(lastword $(MAKEFILE_LIST))

.PHONY: help models images train export tflite onnx to-nrf samples table format lint check train-export ckpt-to-nrf all

help:
	@echo "Single steps:"
	@echo "  train        train model M                    -> ml/$(CKPT)"
	@echo "  export       TFLite + ONNX (FP32 + INT8)      -> ml/outputs/$(CKPT_NAME)/"
	@echo "  tflite       TFLite only (MCU)                -> ml/$(TFLITE)"
	@echo "  onnx         ONNX only (STM32N6, Jetson)      -> ml/outputs/$(CKPT_NAME)/*/model.onnx"
	@echo "  to-nrf       INT8 TFLite into the firmware    -> firmware/nrf5340/src/ML_Models/"
	@echo "  samples      test inputs into the firmware    -> firmware/nrf5340/src/data.hpp"
	@echo "Flows:"
	@echo "  train-export train + export"
	@echo "  ckpt-to-nrf  tflite + to-nrf (existing checkpoint)"
	@echo "  all          train + tflite + to-nrf + samples (up to a buildable firmware)"
	@echo "Results:"
	@echo "  table        all results.json as a Markdown table (TARGET=$(TARGET))"
	@echo "Code quality:"
	@echo "  format       format code and apply safe lint fixes (changes files)"
	@echo "  lint         report lint findings"
	@echo "  check        formatting + lint, changes nothing (run before committing)"
	@echo "Other: models, images"
	@echo "Parameters: M=$(M) GPU=$(GPU) EVAL=$(EVAL) [CKPT=...] [TFLITE=...]"

models:
	$(ML) models

# After changes to ml/src, the Dockerfile, pyproject.toml or uv.lock, otherwise old code runs
images:
	$(COMPOSE) build ml ml-gpu ml-dev

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

all:
	$(STEP) train
	$(STEP) tflite
	$(STEP) to-nrf
	$(STEP) samples
