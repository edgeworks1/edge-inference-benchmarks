#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float prediction;
    uint32_t execution_cycles;
    uint32_t execution_time_us;
} InferenceResult;

bool inference_init(const unsigned char* model_data);
bool inference_run(const float* raw_features, InferenceResult* result);
void inference_dump_profiling(void);

#ifdef __cplusplus
}
#endif