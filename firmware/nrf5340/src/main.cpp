#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include "inference.hpp"
#include "data.hpp"
#include "ML_Models/model_data.h"

int main(void)
{
    printk("Starting Edge AI Benchmark on nRF5340...\n");

    if (!inference_init(g_model_data)) {
        printk("Initialization aborted.\n");
        return -1;
    }

    InferenceResult res;

    // First run (also records the per-layer profile)
    if (!inference_run(TEST_INPUT_SAMPLES[0], &res)) {
        printk("Inference failed.\n");
        return -1;
    }
    printk("[Warm-up Run] Pred: %6.3f | Latency: %u us | Cycles: %u\n",
           (double)res.prediction,
           res.execution_time_us,
           res.execution_cycles);

    // Print the detailed per-layer report
    inference_dump_profiling();

    // Machine-readable (JSON) for scripts/run_json.py, always the last line.
    // sha256 of the .tflite (embedded by codegen) -> the measurement is linked to the right model.
    printk("RESULT {\"model_sha256\": \"%s\", \"cycles\": %u, \"latency_us\": %u, "
           "\"arena_used_bytes\": %u, \"prediction\": %.4f}\n",
           g_model_data_sha256,
           res.execution_cycles,
           res.execution_time_us,
           static_cast<unsigned int>(inference_arena_used()),
           (double)res.prediction);

    return 0;
}