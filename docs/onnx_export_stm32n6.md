# ONNX export for the STM32N6 NPU: why opset 13

Status: 2026-10-05 · Code: `ml/src/edge_ib/export.py` (`export_onnx`), `ml/src/edge_ib/quantization.py` (`quantize_onnx_int8_qdq`)

## What the ONNX export is for

| Format | Target | Toolchain |
|---|---|---|
| `.tflite` (full INT8) | nRF5340, Cortex-M33 without NPU | TFLite Micro + CMSIS-NN |
| `.onnx` (INT8 QDQ) | STM32N6, Neural-ART NPU | ST Edge AI Core (STM32Cube.AI) |

So the ONNX path exists because of the STM32N6 NPU. The nRF does not need it (`edge-ib export … --no-onnx`).

## Decision

Export with **opset 13** through the **legacy TorchScript exporter** (`torch.onnx.export(..., dynamo=False)`).

**Reason:** In the first implementation, the ST toolchain did not accept the ONNX operator `HardSwish` for the NPU. `.tflite` did not have this problem. Unfortunately, the ST Edge AI version used back then was not recorded.

**How opset 13 solves it:** The ONNX operator `HardSwish` only exists from opset 14 on. At opset 13, the TorchScript exporter decomposes `nn.Hardswish` into operators that already existed before. Checked with torch 2.12.1:

```
opset 13: Conv → HardSigmoid → Mul      (HardSwish(x) = x · HardSigmoid(x))
opset 14: Conv → HardSwish
```

**Why `dynamo=False`:** PyTorch's newer Dynamo exporter targets current opsets. Exporting to opset 13 with this decomposition is most direct with the legacy TorchScript exporter. PyTorch considers the legacy exporter deprecated, so this path may break with a torch update.

## What the current ST docs say

Source: ST Edge AI Core **4.0** embedded docs (retrieved 2026-10-05):
- [Neural-ART: Supported operators and limitations](https://stm32ai-cs.st.com/assets/embedded-docs/stneuralart_operator_support.html)
- [ONNX toolbox support](https://stm32ai-cs.st.com/assets/embedded-docs/supported_ops_onnx.html)
- [TFLite toolbox support](https://stm32ai-cs.st.com/assets/embedded-docs/supported_ops_tflite.html)
- [Neural-ART programming model](https://stm32ai-cs.st.com/assets/embedded-docs/stneuralart_programming_model.html)

### HardSwish and the decomposition

| Operator | ONNX → NPU | TFLite → NPU | Note |
|---|---|---|---|
| HardSwish | **HW** | **HW** (`HARD_SWISH`) | Misspelled as "HardSwitch" in the ONNX table |
| HardSigmoid | **HW** | no dedicated TFLite op | Import restriction (ONNX page): *"supported only for 1D tensor and only on the channel dimension"* |
| Mul | **HW** | **HW** | Software fallback when broadcasting exceeds the HW limits |

**Important:** According to docs 4.0, **native HardSwish (opset ≥ 14) is now HW-supported, without restrictions.** The opset 13 decomposition, on the other hand, contains `HardSigmoid`, for which the import page lists a restriction. With current ST tools, opset 14+ might therefore be the better choice. This is **not verified in practice**, see "Open points".

### Importable opsets

> "A part of a subset of operators from Opset 7, 8, 9, 10 up to 20 of ONNX 1.15 is supported."

Opset 13 and 14 are both within the supported range.

### ONNX partly maps more operators to the NPU than TFLite

This is confirmed. The ONNX mapping table is longer and contains HW-mapped operators that are missing from the TFLite table, e.g.:

`Gelu` · `Selu` · `Erf` · `HardSigmoid` · `BatchNormalization` (standalone) · trigonometric ops (`Asin`, `Acos`, `Atan`, `Tan`, `Sinh` …)

For the MobileNet models here this hardly matters: Conv, DepthwiseConv, ReLU, HardSwish, Add and pooling are HW in both formats. It becomes interesting for transformer models (`Gelu`, `Erf`).

### Quantization format

Both formats use the same scheme: **INT8, per-channel, "ss/sa"** (weights signed symmetric, activations signed asymmetric).

- **TFLite:** INT8-quantized model (PTQ or QAT)
- **ONNX:** must be in **QDQ format** (QuantizeLinear/DequantizeLinear around the float ops)

`quantize_onnx_int8_qdq` produces exactly that: `QuantFormat.QDQ`, `QInt8` for weights and activations, `per_channel=True`. The onnxruntime symmetry defaults (weights symmetric, activations asymmetric) match ss/sa.

Ops that are not quantized do not run on the NPU but as a software fallback on the Cortex-M55.

## Open points (to settle in the STM32N6 step)

1. Run both variants through `stedgeai analyze` (target Neural-ART) with the current ST Edge AI version: opset 13 (HardSigmoid + Mul) and opset 14 (HardSwish). Compare which ops are mapped to HW and which to SW.
2. If opset 14+ maps cleanly to HW: raise the `onnx_opset` default (`ExportOptions` or `edge-ib export --onnx-opset`) and replace the TorchScript exporter (`dynamo=False`) with the Dynamo exporter.
3. Record the ST Edge AI version used here.
