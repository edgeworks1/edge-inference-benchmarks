# ONNX-Export für die STM32N6-NPU: warum opset 13

Stand: 2026-10-05 · Code: `ml/src/edge_ib/export.py` (`export_onnx`, `quantize_onnx_int8`)

## Wofür es den ONNX-Export gibt

| Format | Ziel | Toolchain |
|---|---|---|
| `.tflite` (full INT8) | nRF5340, Cortex-M33 ohne NPU | TFLite Micro + CMSIS-NN |
| `.onnx` (INT8 QDQ) | STM32N6, Neural-ART NPU | ST Edge AI Core (STM32Cube.AI) |

Der ONNX-Pfad existiert also wegen der STM32N6-NPU. Für das nRF wird er nicht gebraucht (`edge-ib export … --no-onnx`).

## Entscheidung

Export mit **opset 13** über den **alten TorchScript-Exporter** (`torch.onnx.export(..., dynamo=False)`).

**Grund:** Bei der ersten Umsetzung hat die ST-Toolchain den ONNX-Operator `HardSwish` für die NPU nicht akzeptiert. Mit `.tflite` gab es dieses Problem nicht. Die damals benutzte Version von ST Edge AI ist leider nicht notiert.

**Wie opset 13 das löst:** Den ONNX-Operator `HardSwish` gibt es erst ab opset 14. Bei opset 13 zerlegt der TorchScript-Exporter `nn.Hardswish` in Operatoren, die es schon vorher gab. Geprüft mit torch 2.12.1:

```
opset 13: Conv → HardSigmoid → Mul      (HardSwish(x) = x · HardSigmoid(x))
opset 14: Conv → HardSwish
```

**Warum `dynamo=False`:** Der neuere Dynamo-Exporter von PyTorch zielt auf aktuelle Opsets. Ein Export auf opset 13 mit dieser Zerlegung ist mit dem alten TorchScript-Exporter am direktesten möglich. Der alte Exporter gilt in PyTorch als Auslaufmodell. Bei einem torch-Update kann dieser Pfad also brechen.

## Was die aktuelle ST-Doku sagt

Quelle: ST Edge AI Core **4.0** Embedded Docs (abgerufen 2026-10-05):
- [Neural-ART: Supported operators and limitations](https://stm32ai-cs.st.com/assets/embedded-docs/stneuralart_operator_support.html)
- [ONNX toolbox support](https://stm32ai-cs.st.com/assets/embedded-docs/supported_ops_onnx.html)
- [TFLite toolbox support](https://stm32ai-cs.st.com/assets/embedded-docs/supported_ops_tflite.html)
- [Neural-ART programming model](https://stm32ai-cs.st.com/assets/embedded-docs/stneuralart_programming_model.html)

### HardSwish und die Zerlegung

| Operator | ONNX → NPU | TFLite → NPU | Hinweis |
|---|---|---|---|
| HardSwish | **HW** | **HW** (`HARD_SWISH`) | In der ONNX-Tabelle verschrieben als „HardSwitch“ |
| HardSigmoid | **HW** | kein eigener TFLite-Op | Import-Einschränkung (ONNX-Seite): *„supported only for 1D tensor and only on the channel dimension“* |
| Mul | **HW** | **HW** | Software-Fallback bei Broadcasting außerhalb der HW-Grenzen |

**Wichtig:** Laut Doku 4.0 ist **natives HardSwish (opset ≥ 14) inzwischen HW-unterstützt, und zwar ohne Einschränkung.** Ausgerechnet die opset-13-Zerlegung enthält dagegen `HardSigmoid`, und dafür nennt die Import-Seite eine Einschränkung. Mit aktuellen ST-Tools könnte opset 14+ also die bessere Wahl sein. Das ist **nicht praktisch geprüft**. Siehe „Offene Punkte“.

### Importierbare Opsets

> „A part of a subset of operators from Opset 7, 8, 9, 10 up to 20 of ONNX 1.15 is supported.“

opset 13 und 14 liegen beide im unterstützten Bereich.

### ONNX hat auf der NPU teilweise mehr Operatoren als TFLite

Das hat sich bestätigt. Die ONNX-Mapping-Tabelle ist länger und enthält HW-gemappte Operatoren, die in der TFLite-Tabelle fehlen, z. B.:

`Gelu` · `Selu` · `Erf` · `HardSigmoid` · `BatchNormalization` (einzeln) · trigonometrische Ops (`Asin`, `Acos`, `Atan`, `Tan`, `Sinh` …)

Für die MobileNet-Modelle hier spielt das kaum eine Rolle: Conv, DepthwiseConv, ReLU, HardSwish, Add und Pooling sind in beiden Formaten HW. Interessant wird es für Transformer-Modelle (`Gelu`, `Erf`).

### Quantisierungsformat

Beide Formate nutzen dasselbe Schema: **INT8, per-channel, „ss/sa“** (Gewichte signed symmetrisch, Aktivierungen signed asymmetrisch).

- **TFLite:** INT8-quantisiertes Modell (PTQ oder QAT)
- **ONNX:** muss im **QDQ-Format** vorliegen (QuantizeLinear/DequantizeLinear um die Float-Ops)

`quantize_onnx_int8` erzeugt genau das: `QuantFormat.QDQ`, `QInt8` für Gewichte und Aktivierungen, `per_channel=True`. Die Symmetrie-Defaults von onnxruntime (Gewichte symmetrisch, Aktivierungen asymmetrisch) entsprechen ss/sa.

Ops, die nicht quantisiert sind, laufen nicht auf der NPU, sondern als Software-Fallback auf dem Cortex-M55.

## Offene Punkte (beim STM32N6-Schritt klären)

1. Mit der aktuellen ST-Edge-AI-Version beide Varianten durch `stedgeai analyze` (Target Neural-ART) schicken: opset 13 (HardSigmoid + Mul) und opset 14 (HardSwish). Vergleichen, welche Ops HW und welche SW gemappt werden.
2. Wenn opset 14+ sauber auf HW mappt: Default von `onnx_opset` hochsetzen (`ExportOptions` bzw. `edge-ib export --onnx-opset`) und den TorchScript-Exporter (`dynamo=False`) durch den Dynamo-Exporter ersetzen.
3. Die verwendete ST-Edge-AI-Version hier notieren.
