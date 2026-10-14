#include "profiling.hpp"
#include <zephyr/sys/printk.h>
#include <zephyr/kernel.h>
#include <string.h>

DwtMicroProfiler::DwtMicroProfiler() : num_events_(0) {
    memset(events_, 0, sizeof(events_));
}

void DwtMicroProfiler::InitHardware() {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;

    uint32_t ctrl_mask = DWT_CTRL_CYCCNTENA_Msk;
#if defined(DWT_CTRL_LSUEVTENA_Msk)
    ctrl_mask |= DWT_CTRL_LSUEVTENA_Msk;
#elif defined(DWT_CTRL_LSUCNTENA_Msk)
    ctrl_mask |= DWT_CTRL_LSUCNTENA_Msk;
#endif

#if defined(DWT_CTRL_CPIEVTENA_Msk)
    ctrl_mask |= DWT_CTRL_CPIEVTENA_Msk;
#elif defined(DWT_CTRL_CPICNTENA_Msk)
    ctrl_mask |= DWT_CTRL_CPICNTENA_Msk;
#endif

    DWT->CTRL |= ctrl_mask;
    DWT->CYCCNT = 0;
    DWT->LSUCNT = 0;
    DWT->CPICNT = 0;

    printk("[Profiler] Cortex-M33 DWT Memory & Stall Profiler: ACTIVE\n");
}

void DwtMicroProfiler::Reset() {
    num_events_ = 0;
    DWT->LSUCNT = 0;
    DWT->CPICNT = 0;
}

uint32_t DwtMicroProfiler::BeginEvent(const char* tag) {
    if (num_events_ >= PROFILER_MAX_EVENTS) {
        return 0xFFFFFFFF;
    }
    uint32_t handle = num_events_++;
    events_[handle].tag = tag;
    events_[handle].start_lsu    = DWT->LSUCNT;
    events_[handle].start_cpi    = DWT->CPICNT;
    events_[handle].start_cycles = DWT->CYCCNT;
    return handle;
}

void DwtMicroProfiler::EndEvent(uint32_t event_handle) {
    uint32_t stop_cycles = DWT->CYCCNT;
    uint32_t stop_lsu    = DWT->LSUCNT;
    uint32_t stop_cpi    = DWT->CPICNT;

    if (event_handle != 0xFFFFFFFF && event_handle < num_events_) {
        events_[event_handle].total_cycles = stop_cycles - events_[event_handle].start_cycles;
        events_[event_handle].total_lsu    = (stop_lsu - events_[event_handle].start_lsu) & 0xFF;
        events_[event_handle].total_cpi    = (stop_cpi - events_[event_handle].start_cpi) & 0xFF;
    }
}

uint64_t DwtMicroProfiler::CalculateOperatorMacs(const tflite::Operator* op, 
                                               const tflite::SubGraph* subgraph, 
                                               tflite::BuiltinOperator op_type) {
    if (!op || !subgraph) return 0;

    auto inputs  = op->inputs();
    auto outputs = op->outputs();
    auto tensors = subgraph->tensors();

    if (!inputs || !outputs || !tensors) return 0;
    if (inputs->size() < 2 || outputs->size() < 1) return 0;

    int32_t weight_idx = inputs->Get(1);
    int32_t output_idx = outputs->Get(0);

    if (weight_idx < 0 || weight_idx >= static_cast<int32_t>(tensors->size())) return 0;
    if (output_idx < 0 || output_idx >= static_cast<int32_t>(tensors->size())) return 0;

    const tflite::Tensor* weight_tensor = tensors->Get(weight_idx);
    const tflite::Tensor* output_tensor = tensors->Get(output_idx);

    if (!weight_tensor || !output_tensor || !weight_tensor->shape() || !output_tensor->shape()) {
        return 0;
    }

    auto w_shape = weight_tensor->shape();
    auto o_shape = output_tensor->shape();

    // 1. Fully Connected (Weight: [out_features, in_features])
    if (op_type == tflite::BuiltinOperator_FULLY_CONNECTED && w_shape->size() == 2) {
        return static_cast<uint64_t>(w_shape->Get(0)) * static_cast<uint64_t>(w_shape->Get(1));
    }

    // 2. Convolutions (4D Shapes)
    if (w_shape->size() == 4 && o_shape->size() == 4) {
        uint64_t h_out = static_cast<uint64_t>(o_shape->Get(1));
        uint64_t w_out = static_cast<uint64_t>(o_shape->Get(2));
        uint64_t c_out = static_cast<uint64_t>(o_shape->Get(3));

        uint64_t k_h = static_cast<uint64_t>(w_shape->Get(1));
        uint64_t k_w = static_cast<uint64_t>(w_shape->Get(2));

        if (op_type == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
            // Depthwise: Jeder Ausgangskanal multipliziert genau einen Kernel-Patch (kein C_in Faktor)
            return h_out * w_out * c_out * (k_h * k_w);
        } 
        else if (op_type == tflite::BuiltinOperator_CONV_2D) {
            // Standard Conv2D: Filter Shape ist [C_out, K_H, K_W, C_in]
            uint64_t c_in = static_cast<uint64_t>(w_shape->Get(3));
            return h_out * w_out * c_out * (k_h * k_w * c_in);
        }
    }

    return 0;
}

void DwtMicroProfiler::PrintSummary(const tflite::Model* model, uint32_t cpu_freq_mhz) {
    if (num_events_ == 0) {
        printk("[Profiler] No layer events recorded.\n");
        return;
    }

    uint64_t total_model_cycles = 0;
    for (uint32_t i = 0; i < num_events_; ++i) {
        total_model_cycles += events_[i].total_cycles;
    }

    auto subgraphs = model->subgraphs();
    const tflite::SubGraph* subgraph = subgraphs ? subgraphs->Get(0) : nullptr;
    auto operators = subgraph ? subgraph->operators() : nullptr;
    auto opcodes = model->operator_codes();

    printk("\n=================================== ADVANCED HARDWARE PROFILING REPORT ===================================\n");
    printk("%-3s | %-20s | %-9s | %-8s | %-7s | %-9s | %-8s | %-15s\n", 
           "Idx", "Operator Type", "Cycles", "Time(us)", "% Total", "Cyc/MAC", "LSU-Wait", "Classification");
    printk("----------------------------------------------------------------------------------------------------------\n");

    for (uint32_t i = 0; i < num_events_; ++i) {
        const char* op_name = "UnknownOp";
        uint64_t macs = 0;
        tflite::BuiltinOperator builtin_code = tflite::BuiltinOperator_CUSTOM;

        if (operators && opcodes && i < operators->size()) {
            const tflite::Operator* op = operators->Get(i);
            uint32_t opcode_index = op->opcode_index();
            if (opcode_index < opcodes->size()) {
                const tflite::OperatorCode* op_code = opcodes->Get(opcode_index);
                builtin_code = op_code->builtin_code();
                if (builtin_code == tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES) {
                    builtin_code = static_cast<tflite::BuiltinOperator>(op_code->deprecated_builtin_code());
                }
                op_name = tflite::EnumNameBuiltinOperator(builtin_code);
            }
            macs = CalculateOperatorMacs(op, subgraph, builtin_code);
        }

        uint32_t time_us = events_[i].total_cycles / cpu_freq_mhz;
        uint32_t percent_x10 = (total_model_cycles > 0) 
            ? static_cast<uint32_t>((static_cast<uint64_t>(events_[i].total_cycles) * 1000ULL) / total_model_cycles) 
            : 0;

        float cyc_per_mac = 0.0f;
        const char* bound_desc = "OTHER";

        if (macs > 0) {
            cyc_per_mac = static_cast<float>(events_[i].total_cycles) / static_cast<float>(macs);

            if (cyc_per_mac < 1.25f) {
                bound_desc = "COMPUTE";
            } else if (cyc_per_mac < 2.5f) {
                bound_desc = "BALANCED";
            } else {
                bound_desc = "MEMORY_FLASH";
            }
        } else {
            if (builtin_code == tflite::BuiltinOperator_ADD || 
                builtin_code == tflite::BuiltinOperator_MUL || 
                builtin_code == tflite::BuiltinOperator_SUM ||
                builtin_code == tflite::BuiltinOperator_HARD_SWISH) {
                bound_desc = "ELEMENTWISE_RAM";
            } else if (builtin_code == tflite::BuiltinOperator_PAD || 
                       builtin_code == tflite::BuiltinOperator_TRANSPOSE ||
                       builtin_code == tflite::BuiltinOperator_RESHAPE) {
                bound_desc = "IO_OVERHEAD";
            }
        }

        if (macs > 0) {
            printk("%-3u | %-20s | %-9u | %-8u | %3u.%1u%% | %7.2f   | %-8u | %-15s\n",
                   i, op_name, events_[i].total_cycles, time_us,
                   percent_x10 / 10, percent_x10 % 10,
                   static_cast<double>(cyc_per_mac),
                   events_[i].total_lsu,
                   bound_desc);
        } else {
            printk("%-3u | %-20s | %-9u | %-8u | %3u.%1u%% |    --     | %-8u | %-15s\n",
                   i, op_name, events_[i].total_cycles, time_us,
                   percent_x10 / 10, percent_x10 % 10,
                   events_[i].total_lsu,
                   bound_desc);
        }
    }

    uint32_t total_time_us = static_cast<uint32_t>(total_model_cycles / cpu_freq_mhz);
    printk("----------------------------------------------------------------------------------------------------------\n");
    printk("Total Layer Time: %u us (%u ms) | Total Cycles: %llu\n", 
           total_time_us, 
           total_time_us / 1000, 
           total_model_cycles);
    printk("Classification Legend:\n");
    printk("  COMPUTE         : Cyc/MAC < 1.25 -> DSP SMLAD Pipeline optimal ausgelastet\n");
    printk("  BALANCED        : 1.25 <= Cyc/MAC < 2.5 -> Ausgewogene Rechen- und Lade-Pipelines\n");
    printk("  MEMORY_FLASH    : Cyc/MAC >= 2.5 -> Flash Bus-Wait-States / D-Code Latenz dominieren\n");
    printk("  ELEMENTWISE_RAM : Keine MACs -> Elementweise Byte-Loads/Stores belasten das SRAM\n");
    printk("  IO_OVERHEAD     : Speicher-Umkopierungen (Padding / Reshape / Transpose)\n");
    printk("==========================================================================================================\n\n");
}