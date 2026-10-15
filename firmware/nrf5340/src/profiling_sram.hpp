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
};

class DwtMicroProfiler : public tflite::MicroProfilerInterface {
public:
    DwtMicroProfiler();

    // TFLM Profiler Interface
    uint32_t BeginEvent(const char* tag) override;
    void EndEvent(uint32_t event_handle) override;

    // Analyse & Ausgabe
    void Reset();
    void PrintSummary(const tflite::Model* model, uint32_t cpu_freq_mhz = 128);

private:
    LayerProfile events_[PROFILER_MAX_EVENTS];
    uint32_t num_events_;
};