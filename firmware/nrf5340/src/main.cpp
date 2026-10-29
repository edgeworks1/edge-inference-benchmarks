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

    // Erster Run (inkl. Profiler-Aufzeichnung)
    if (!inference_run(TEST_INPUT_SAMPLES[0], &res)) {
        printk("Inference failed.\n");
        return -1;
    }
    printk("[Warm-up Run] Pred: %6.3f | Latency: %u us | Cycles: %u\n",
           (double)res.prediction,
           res.execution_time_us,
           res.execution_cycles);

    // Detail-Report für jeden Layer ausgeben
    inference_dump_profiling();

    // Maschinenlesbar (JSON) für scripts/run_json.py, immer als letzte Zeile.
    // sha256 der .tflite (von codegen eingebettet) -> Messwert landet sicher beim richtigen Modell.
    printk("RESULT {\"model_sha256\": \"%s\", \"cycles\": %u, \"latency_us\": %u, "
           "\"arena_used_bytes\": %u, \"prediction\": %.4f}\n",
           g_model_data_sha256,
           res.execution_cycles,
           res.execution_time_us,
           static_cast<unsigned int>(inference_arena_used()),
           (double)res.prediction);

    return 0;
}