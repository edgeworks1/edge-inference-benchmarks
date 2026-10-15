#include "profiling.hpp"
#include <zephyr/sys/printk.h>
#include <zephyr/kernel.h>
#include <string.h>

DwtMicroProfiler::DwtMicroProfiler() : num_events_(0) {
    memset(events_, 0, sizeof(events_));
}

void DwtMicroProfiler::Reset() {
    num_events_ = 0;
}

uint32_t DwtMicroProfiler::BeginEvent(const char* tag) {
    if (num_events_ >= PROFILER_MAX_EVENTS) {
        return 0xFFFFFFFF; // Ungültiger Handle, damit nicht 0 überschrieben wird
    }
    uint32_t handle = num_events_++;
    events_[handle].tag = tag;
    events_[handle].start_cycles = DWT->CYCCNT;
    return handle;
}

void DwtMicroProfiler::EndEvent(uint32_t event_handle) {
    uint32_t stop_cycles = DWT->CYCCNT;
    if (event_handle != 0xFFFFFFFF && event_handle < num_events_) {
        events_[event_handle].total_cycles = stop_cycles - events_[event_handle].start_cycles;
    }
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

    printk("\n============================= LAYER PROFILING REPORT (SRAM EXECUTION) =============================\n");
    printk("%-4s | %-24s | %-12s | %-10s | %-7s\n", "Idx", "Operator Type", "Cycles", "Time (us)", "% Total");
    printk("----------------------------------------------------------------------------------------------------\n");

    for (uint32_t i = 0; i < num_events_; ++i) {
        const char* op_name = "UnknownOp";

        if (operators && opcodes && i < operators->size()) {
            const tflite::Operator* op = operators->Get(i);
            uint32_t opcode_index = op->opcode_index();
            if (opcode_index < opcodes->size()) {
                const tflite::OperatorCode* op_code = opcodes->Get(opcode_index);
                tflite::BuiltinOperator builtin_code = op_code->builtin_code();
                if (builtin_code == tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES) {
                    builtin_code = static_cast<tflite::BuiltinOperator>(op_code->deprecated_builtin_code());
                }
                op_name = tflite::EnumNameBuiltinOperator(builtin_code);
            }
        }

        uint32_t time_us = events_[i].total_cycles / cpu_freq_mhz;

        // Overflow-sichere Prozentberechnung via uint64_t
        uint32_t percent_x10 = (total_model_cycles > 0) 
            ? static_cast<uint32_t>((static_cast<uint64_t>(events_[i].total_cycles) * 1000ULL) / total_model_cycles) 
            : 0;

        printk("%-4u | %-24s | %-12u | %-10u | %3u.%1u%%\n",
               i,
               op_name,
               events_[i].total_cycles,
               time_us,
               percent_x10 / 10,
               percent_x10 % 10);
    }

    uint32_t total_time_us = static_cast<uint32_t>(total_model_cycles / cpu_freq_mhz);
    printk("----------------------------------------------------------------------------------------------------\n");
    printk("Total Layer Time: %u us (%u ms) | Total Cycles: %llu\n", 
           total_time_us, 
           total_time_us / 1000, 
           total_model_cycles);
    printk("====================================================================================================\n\n");
}