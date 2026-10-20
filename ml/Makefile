# Einstiegspunkte für die ML-Pipeline. Läuft auf dem Host (WSL), alles passiert in Docker
# (Dienste ml / ml-gpu aus compose.yaml im Repo-Root). Pfade sind Container-Pfade relativ zu /app.
#
# Aufruf aus dem Repo-Root: make -C ml <target> M=<modell>      (`make -C ml help` zeigt alles)

# ---- Parameter (auf der Kommandozeile überschreibbar: make -C ml all M=mobilenet_v3_micro_2d) ----
# (Kommentare bewusst NICHT hinter den Werten: Make würde die Leerzeichen davor mitnehmen)
# M:      Modellname aus `make -C ml models`
# GPU:    1 = Training auf GPU (Dienst ml-gpu), 0 = CPU (Dienst ml)
# EVAL:   0 = Export ohne MAE-Auswertung (schneller)
# CKPT:   Eingabe für export/tflite/onnx
# TFLITE: Eingabe für to-nrf (Dateiname kommt von export: <modell>_full_int8.tflite)
M      ?= mobilenet_v3_micro_2d
GPU    ?= 1
EVAL   ?= 1
CKPT   ?= checkpoints/$(M).pt
TFLITE ?= outputs/$(M)_full_int8.tflite

# ---- Interna ----
HERE    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
COMPOSE := docker compose -f $(HERE)../compose.yaml
ML      := $(COMPOSE) run --rm ml
TRAIN   := $(COMPOSE) run --rm $(if $(filter 1,$(GPU)),ml-gpu,ml)
NOEVAL  := $(if $(filter 0,$(EVAL)),--no-eval)
# firmware = firmware/nrf5340/src (Mount im ml-Dienst)
NRF_SRC := firmware
STEP    := $(MAKE) --no-print-directory -f $(lastword $(MAKEFILE_LIST))

.PHONY: help models images train export tflite onnx to-nrf samples train-export ckpt-to-nrf all

help:
	@echo "Einzelaufgaben:"
	@echo "  train        Modell M trainieren              -> ml/$(CKPT)"
	@echo "  export       TFLite + ONNX (FP32 + INT8)      -> ml/outputs/"
	@echo "  tflite       nur TFLite (MCU)                 -> ml/$(TFLITE)"
	@echo "  onnx         nur ONNX (STM32N6, Jetson)       -> ml/outputs/$(M)_int8.onnx"
	@echo "  to-nrf       INT8-TFLite in die nRF-Firmware  -> firmware/nrf5340/src/ML_Models/"
	@echo "  samples      Test-Eingaben in die Firmware    -> firmware/nrf5340/src/data.hpp"
	@echo "Abläufe:"
	@echo "  train-export train + export"
	@echo "  ckpt-to-nrf  tflite + to-nrf (vorhandener Checkpoint)"
	@echo "  all          train + tflite + to-nrf (End-to-End bis zur Firmware)"
	@echo "Sonstiges: models, images"
	@echo "Parameter: M=$(M) GPU=$(GPU) EVAL=$(EVAL) [CKPT=…] [TFLITE=…]"

models:
	$(ML) models

# Nach Änderungen an ml/src, Dockerfile, pyproject.toml oder uv.lock – sonst läuft alter Code
images:
	$(COMPOSE) build ml ml-gpu

# ---- Einzelaufgaben ----
train:
	$(TRAIN) train -m $(M) -o $(CKPT)

export:
	$(ML) export $(CKPT) $(NOEVAL)

tflite:
	$(ML) export $(CKPT) --no-onnx $(NOEVAL)

onnx:
	$(ML) export $(CKPT) --no-tflite $(NOEVAL)

to-nrf:
	$(ML) codegen model $(TFLITE) -o $(NRF_SRC)/ML_Models

samples:
	$(ML) codegen samples -o $(NRF_SRC)

# ---- Abläufe (nacheinander, auch bei make -j) ----
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
