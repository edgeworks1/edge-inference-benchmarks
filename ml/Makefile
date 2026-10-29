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
# CKPT_NAME: Ordner in outputs/ = Pfad des Checkpoints in checkpoints/ ohne .pt
#           (checkpoints/mobilenet/bad_simd.pt -> outputs/mobilenet/bad_simd/{fp32_full,int8_full,int8_qdq}/)
# TFLITE: Eingabe für to-nrf
# TARGET: Gerät, dessen Messwerte `table` zeigt (nrf5340dk oder renode)
M      ?= mobilenet_v3_micro_2d
GPU    ?= 1
EVAL   ?= 1
CKPT   ?= checkpoints/$(M).pt
CKPT_NAME ?= $(patsubst checkpoints/%.pt,%,$(CKPT))
TFLITE ?= outputs/$(CKPT_NAME)/int8_full/model.tflite
TARGET ?= nrf5340dk

# ---- Interna ----
HERE    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
COMPOSE := docker compose -f $(HERE)../compose.yaml
ML      := $(COMPOSE) run --rm ml
TRAIN   := $(COMPOSE) run --rm $(if $(filter 1,$(GPU)),ml-gpu,ml)
NOEVAL  := $(if $(filter 0,$(EVAL)),--no-eval)
# Host-Pfad: ml selbst kennt die Firmware nicht, nur dieses Makefile kopiert hinein
NRF_SRC := $(HERE)../firmware/nrf5340/src
STEP    := $(MAKE) --no-print-directory -f $(lastword $(MAKEFILE_LIST))

.PHONY: help models images train export tflite onnx to-nrf samples table train-export ckpt-to-nrf all

help:
	@echo "Einzelaufgaben:"
	@echo "  train        Modell M trainieren              -> ml/$(CKPT)"
	@echo "  export       TFLite + ONNX (FP32 + INT8)      -> ml/outputs/$(CKPT_NAME)/"
	@echo "  tflite       nur TFLite (MCU)                 -> ml/$(TFLITE)"
	@echo "  onnx         nur ONNX (STM32N6, Jetson)       -> ml/outputs/$(CKPT_NAME)/*/model.onnx"
	@echo "  to-nrf       INT8-TFLite in die nRF-Firmware  -> firmware/nrf5340/src/ML_Models/"
	@echo "  samples      Test-Eingaben in die Firmware    -> firmware/nrf5340/src/data.hpp"
	@echo "Abläufe:"
	@echo "  train-export train + export"
	@echo "  ckpt-to-nrf  tflite + to-nrf (vorhandener Checkpoint)"
	@echo "  all          train + tflite + to-nrf + samples (bis zur baubaren Firmware)"
	@echo "Ergebnisse:"
	@echo "  table        alle results.json als Markdown-Tabelle (TARGET=$(TARGET))"
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

# Codegen schreibt neben die .tflite (…/int8_full/codegen/), danach Kopie in die Firmware
to-nrf:
	$(ML) codegen model $(TFLITE)
	mkdir -p $(NRF_SRC)/ML_Models
	cp $(HERE)$(dir $(TFLITE))codegen/model_data.h $(HERE)$(dir $(TFLITE))codegen/model_data.cc $(NRF_SRC)/ML_Models/

samples:
	$(ML) codegen samples
	cp $(HERE)outputs/samples/data.hpp $(NRF_SRC)/

table:
	@$(COMPOSE) --progress quiet run --rm -T ml results table --target $(TARGET)

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
	$(STEP) samples
