#include "inference.hpp"
#include "profiling.hpp"

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <string.h>
#include <math.h>

#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define TENSOR_ARENA_SIZE (3 * 1024 * 1024)

alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static tflite::RecordingMicroInterpreter* interpreter = nullptr;
static const tflite::Model* current_model = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;
static bool is_initialized = false;
static int total_input_elements = 0;

static DwtMicroProfiler profiler;

bool inference_init(const unsigned char* model_data)
{
    if (model_data == nullptr) {
        printk("Error: Model pointer is NULL\n");
        return false;
    }

    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    DWT->CYCCNT = 0;

    tflite::InitializeTarget();

    current_model = tflite::GetModel(model_data);
    if (current_model->version() != TFLITE_SCHEMA_VERSION) {
        printk("Error: Model schema mismatch (Expected %d, got %d)\n",
               TFLITE_SCHEMA_VERSION, current_model->version());
        return false;
    }

    static tflite::MicroMutableOpResolver<30> resolver;
    resolver.AddFullyConnected();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddHardSwish();
    resolver.AddSum();
    resolver.AddMean();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddSub();
    resolver.AddLogistic();
    resolver.AddReshape();
    resolver.AddPad();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddSoftmax();

    static tflite::RecordingMicroInterpreter static_interpreter(
        current_model,
        resolver,
        tensor_arena,
        TENSOR_ARENA_SIZE,
        nullptr,
        &profiler
    );

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printk("Error: AllocateTensors failed\n");
        return false;
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);

    total_input_elements = 1;
    for (int i = 0; i < input_tensor->dims->size; ++i) {
        total_input_elements *= input_tensor->dims->data[i];
    }

    is_initialized = true;

    printk("TFLM initialized. Inputs: %d | Total Arena: %u / %d bytes\n",
           total_input_elements,
           static_cast<unsigned int>(interpreter->arena_used_bytes()), 
           TENSOR_ARENA_SIZE);

    printk("\n============================ MEMORY PROFILING REPORT ============================\n");
    interpreter->GetMicroAllocator().PrintAllocations();
    printk("==================================================================================\n\n");

    return true;
}

bool inference_run(const float* raw_features, InferenceResult* result)
{
    if (!is_initialized || raw_features == nullptr || result == nullptr) {
        return false;
    }

    if (input_tensor->type == kTfLiteFloat32) {
        memcpy(input_tensor->data.f, raw_features, sizeof(float) * total_input_elements);
    } 
    else if (input_tensor->type == kTfLiteInt8) {
        const float scale = input_tensor->params.scale;
        const int32_t zero_point = input_tensor->params.zero_point;

        for (int i = 0; i < total_input_elements; ++i) {
            int32_t q = static_cast<int32_t>(roundf(raw_features[i] / scale)) + zero_point;
            if (q < -128) q = -128;
            if (q > 127)  q = 127;
            input_tensor->data.int8[i] = static_cast<int8_t>(q);
        }
    } else {
        return false;
    }

    profiler.Reset();

    uint32_t start_cycles = DWT->CYCCNT;
    TfLiteStatus status = interpreter->Invoke();
    uint32_t stop_cycles = DWT->CYCCNT;
    uint32_t delta_cycles = stop_cycles - start_cycles;

    if (status != kTfLiteOk) {
        return false;
    }

    float predicted_value = 0.0f;
    if (output_tensor->type == kTfLiteFloat32) {
        predicted_value = output_tensor->data.f[0];
    } else if (output_tensor->type == kTfLiteInt8) {
        const float scale = output_tensor->params.scale;
        const int32_t zero_point = output_tensor->params.zero_point;
        predicted_value = (static_cast<float>(output_tensor->data.int8[0]) - zero_point) * scale;
    }

    result->prediction = predicted_value;
    result->execution_cycles = delta_cycles;
    result->execution_time_us = delta_cycles / 128;

    return true;
}

void inference_dump_profiling(void)
{
    if (current_model && interpreter) {
        profiler.PrintSummary(current_model, 128);

        printk("\n============================ MEMORY PROFILING REPORT ============================\n");
        interpreter->GetMicroAllocator().PrintAllocations();
        printk("==================================================================================\n\n");
    }
}