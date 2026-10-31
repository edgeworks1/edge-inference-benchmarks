#pragma once

#include <stdint.h>
#include <stddef.h>
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define PROFILER_MAX_EVENTS 512

struct LayerProfile {
    const char* tag;
    uint32_t start_cycles;
    uint32_t total_cycles;
    uint32_t start_lsu;
    uint32_t total_lsu;
    uint32_t start_cpi;
    uint32_t total_cpi;
};

class DwtMicroProfiler : public tflite::MicroProfilerInterface {
public:
    DwtMicroProfiler();

    // Hardware init of the Cortex-M33 DWT registers
    void InitHardware();

    // TFLM Profiler Interface
    uint32_t BeginEvent(const char* tag) override;
    void EndEvent(uint32_t event_handle) override;

    // Analysis and report
    void Reset();
    void PrintSummary(const tflite::Model* model, uint32_t cpu_freq_mhz = 128);

private:
    LayerProfile events_[PROFILER_MAX_EVENTS];
    uint32_t num_events_;

    // Computes the theoretical MACs from the flatbuffer tensor shapes
    uint64_t CalculateOperatorMacs(const tflite::Operator* op, 
                                   const tflite::SubGraph* subgraph, 
                                   tflite::BuiltinOperator op_type);
};