#
# Copyright (c) 2021 Nordic Semiconductor
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
cmake_minimum_required(VERSION 3.20)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(ble_cs_profiling C CXX)

# ---------------------------------------------------------
# 1. APPLICATION SOURCES
# ---------------------------------------------------------
target_sources(app PRIVATE
  src/main.cpp
  src/ML_Models/model_data.cc
  src/inference.cpp
  src/profiling.cpp
)

zephyr_library_include_directories(${CMAKE_CURRENT_SOURCE_DIR})
