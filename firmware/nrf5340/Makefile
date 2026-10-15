.PHONY: build build-pristine flash-starter flash-responder flash-both all

BOARD := nrf5340dk_nrf5340_cpuapp
WEST := west

BUILD_DIR := build

# DevKit IDs
SNR_STARTER := 1050025601
SNR_RESPONDER := 1050080297

MERGED_APP := $(BUILD_DIR)/merged.hex
MERGED_NET := $(BUILD_DIR)/merged_CPUNET.hex

# ===========================
# BUILD TARGETS
# ===========================

# Fast incremental sysbuild
build:
	$(WEST) build -d $(BUILD_DIR)

# Clean sysbuild
build-pristine:
	$(WEST) build \
		-b $(BOARD) \
		-d $(BUILD_DIR) \
		--pristine \
		--no-sysbuild

# ===========================
# FLASH TARGETS (FAST)
# ===========================

flash-starter:
	$(WEST) flash \
		-d $(BUILD_DIR) \
		--dev-id $(SNR_STARTER)

# Flash RESPONDER (APP + NET via sysbuild)
flash-responder:
	$(WEST) flash \
		-d $(BUILD_DIR) \
		--dev-id $(SNR_RESPONDER)

# Beide flashen
flash-both:
	$(MAKE) flash-starter
	$(MAKE) flash-responder

# ===========================
# FULL PIPELINE
# ===========================

all:
	$(MAKE) build
	$(MAKE) flash-both
