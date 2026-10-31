# Architektur: Wie ein Edge-AI-Projekt aufgebaut ist

Dieses Dokument erklärt, **wie** diese Pipeline aufgebaut ist und **warum**. Das Beispiel ist
Distanzschätzung aus BLE-Channel-Sounding-Daten auf einem nRF5340, das ist aber austauschbar.
Wichtig ist das Muster dahinter, das du in deinem eigenen Projekt wiederfinden solltest:
egal ob GBDT, CNN, Vision Transformer oder Graph-NN, egal ob Mikrocontroller, NPU oder Jetson.

> **Das hier ist kein Template, bei dem du zwei Zeilen änderst und es läuft.** Andere Daten,
> andere Modelle und andere Hardware bedeuten echten Umbau, oft gerade an den schwierigen
> Stellen (Export, Quantisierung, Operator-Support). Übernimm die *Struktur* und die
> *Prinzipien*, nicht den BLE-Code.

---

## 1. Das mentale Modell: drei Welten

Ein Edge-AI-Projekt durchläuft drei Welten mit völlig unterschiedlichen Regeln:

```
 ① TRAINING                  ② AUSTAUSCHFORMAT              ③ ZIELHARDWARE
 PC / GPU-Server             hardware-unabhängig            hardware-spezifisch

 PyTorch, float32     export   ONNX, TFLite            compile   C-Array + TFLite Micro (MCU)
 Python, viel RAM    ───────▶  eingefrorener Graph    ────────▶  TensorRT-Engine (Jetson)
 flexibel, langsam             + Quantisierung (INT8)            ST Edge AI (STM32N6-NPU)
                                                                 kein Python, KB statt GB
```

**① Training** braucht Flexibilität: Autograd, Debugger, beliebige Python-Logik, GPU. Hier
entwickelst du das Modell. Ergebnis ist ein **Checkpoint** (`.pt`): Gewichte plus alles, was man
braucht, um das Modell wieder aufzubauen.

**② Austauschformat** friert das Modell ein: ein fester Rechengraph aus Standard-Operatoren
(Conv, FullyConnected, Add …), ohne Python. Hier wird auch **quantisiert** (float32 → int8:
4× kleiner, auf MCUs deutlich schneller). ONNX und TFLite sind von keinem Gerät abhängig. Aus
einem Export können mehrere Zielplattformen bedient werden.

**③ Zielhardware**: Ein herstellerspezifischer Compiler oder eine Runtime bringt den Graphen
auf genau einen Chip: Er wählt optimierte Kernels (z. B. CMSIS-NN mit SIMD auf Cortex-M),
plant den Speicher und legt Operatoren auf Beschleuniger (NPU). Erst hier sieht man die echten
Zahlen: Latenz, RAM, Flash.

**Warum diese Trennung?** Es ist dasselbe Prinzip wie beim Compiler (Quellcode → Zwischencode →
Maschinencode). Das Training muss nichts über den Chip wissen, und der Chip muss kein PyTorch
können. Das Austauschformat ist der Vertrag dazwischen. Wechselt die Hardware, ändert sich nur
Welt ③.

**Jeder Übergang kann Genauigkeit oder Funktion kosten.** Ein Operator existiert im Zielformat
nicht, die Quantisierung verschlechtert das Ergebnis, die Hardware unterstützt eine Schicht nicht.
Deshalb wird nach jedem Schritt **gemessen**, nicht angenommen (hier: TFLite-Ausgabe gegen
PyTorch, MAE für FP32 und INT8 getrennt, Zyklen auf Simulator *und* echtem Board).

---

## 2. Der Weg in diesem Repo

| # | Schritt                | Befehl                                         | Ergebnis                                                                                                                        | Welt     |
| - | ---------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------- |
| 1 | Daten                  | –                                             | `ml/data/sample/{training,eval}.parquet` (kleines Sample damit die Pipeline läuft); voller Datensatz wird nicht mitgeliefert | ①       |
| 2 | Training               | `make -C ml train M=<modell>`                | `ml/checkpoints/<name>.pt`                                                                                                    | ①       |
| 3 | Export + Quantisierung | `make -C ml export`                          | `ml/outputs/<name>/{fp32_full,int8_full,int8_qdq}/model.*`                                                                    | ②       |
| 4 | Codegen                | `make -C ml to-nrf samples`                  | Modell als C-Array in `firmware/nrf5340/src/`                                                                                 | ② → ③ |
| 5 | Firmware bauen         | `make -C firmware/nrf5340 renode` / `hil`  | `build/<variante>/zephyr/zephyr.elf`                                                                                          | ③       |
| 6 | Ausführen + messen    | `make -C firmware/nrf5340 sim` / `measure` | `build/<variante>/run.json`                                                                                                   | ③       |
| 7 | Ergebnisse sammeln     | (passiert in 3 und 6)                          | `ml/outputs/<name>/results.json`                                                                                              | –       |

Was in den Schritten passiert und worauf man achten muss:

**2 – Training.** Der Checkpoint enthält nicht nur Gewichte, sondern auch den Modellnamen aus
der Registry (`models/registry.py`), dessen Argumente, die Trainingseinstellungen und die
Eval-MAE. Damit lässt sich das Modell ohne den Trainingscode-Stand von damals wieder aufbauen.
*Warum der Checkpoint die Übergabe ist:* Training ist teuer (GPU, Minuten bis Tage), Export
billig. Man trainiert einmal und exportiert beliebig oft neu, z. B. für eine andere Hardware.

**3 – Export.** Drei Varianten pro Checkpoint, jede in einem eigenen Ordner:

| Ordner         | Format        | Bedeutung                                                                            |
| -------------- | ------------- | ------------------------------------------------------------------------------------ |
| `fp32_full/` | TFLite + ONNX | float32, Referenz für die Genauigkeit                                               |
| `int8_full/` | TFLite        | INT8 inkl. Ein- und Ausgang: was ein Mikrocontroller braucht                         |
| `int8_qdq/`  | ONNX          | INT8-Gewichte/-Aktivierungen, aber float32-Ein-/Ausgang (für NPU/Jetson-Toolchains) |

Quantisierung braucht **Kalibrierung**: 1000 echte Trainings-Samples laufen durch das Modell,
um die Wertebereiche jeder Schicht zu bestimmen. Schlechte Kalibrierdaten bedeuten ein schlechtes
INT8-Modell.

**4 – Codegen.** Ein Mikrocontroller hat kein Dateisystem. Das Modell wird deshalb als
Byte-Array in C-Code umgewandelt und landet im Flash. Zusätzlich wird der **sha256** der
`.tflite` eingebettet (dazu unten mehr).

**5/6 – Firmware.** Zephyr (RTOS) + TensorFlow Lite Micro + CMSIS-NN-Kernels. Zwei Varianten
derselben App: **Renode** simuliert den Chip mit vergrößertem Speicher (jedes Modell läuft,
Zyklen sind aber nur zum *Vergleichen* geeignet, das Board braucht hier 24–32 % mehr), das **echte Board**
liefert die gültigen Zahlen (Hardware-in-the-Loop, HIL).

**7 – Ergebnisse.** Die Firmware gibt am Ende eine maschinenlesbare Zeile aus
(`RESULT {...}`). Ein Skript auf Firmware-Seite macht daraus eine geräteunabhängige
`run.json`, die ML-Seite ordnet sie per sha256 dem richtigen Modell zu.

---

## 3. Die Prinzipien und warum

### Linux / WSL first

Embedded-Toolchains, Docker, Skripte und CI laufen nativ unter Linux. Unter Windows entsteht
schnell ein Setup aus Klicks in IDEs, das niemand nachbauen kann. Unter Windows arbeitest du
deshalb in **WSL2**. Windows ist nur noch für das Durchreichen des USB-Debuggers zuständig
(`usbipd`, siehe [wsl_flashing_setup.md](wsl_flashing_setup.md)).

### Alles in Docker

Jeder Schritt läuft in einem Container mit festgelegten Versionen: Python-Pakete über
`uv.lock`, Zephyr-SDK und Renode mit geprüftem sha256. *Warum:* „Bei mir läuft's“ verschwindet.
Ein neuer Student, ein CI-Server und du in einem Jahr bekommen exakt dieselbe Umgebung. Auf dem
Host wird nichts installiert außer Docker (und für das Board die J-Link-Tools).

Es gibt **getrennte Images** (`ml`, `ml-gpu`, `fw`, `renode`), weil sie unterschiedlich groß
sind und sich unabhängig ändern. Die ML-Umgebung soll nicht neu bauen, nur weil sich das
Zephyr-SDK ändert.

### CLI first, keine Notebooks in der Pipeline

Jeder Schritt ist ein Kommandozeilen-Befehl (`ble-cs train`, `ble-cs export` …). *Warum:*
Befehle sind automatisierbar (CI, über Nacht 20 Modelle durchmessen), reviewbar und
wiederholbar. Notebooks haben versteckten Zustand (Zellen in beliebiger Reihenfolge ausgeführt)
und lassen sich schlecht versionieren. **Zum Erkunden** von Daten sind Notebooks gut, aber was
funktioniert, wandert in das Python-Paket.

### Makefiles als Einstiegspunkt

`make -C ml help` zeigt, was es gibt. Die Makefiles sind ausführbare Dokumentation: Sie
enthalten die langen `docker compose run …`-Befehle mit allen Parametern, damit niemand sie
abtippen oder erraten muss. Sie verbinden außerdem ML und Firmware (siehe nächster Punkt).

### ML und Firmware sind entkoppelt

Das Python-Paket `ml/` kennt keine Firmware-Pfade, die Firmware kennt kein Python-Paket. Sie
reden nur über **zwei Dateiformate**:

```
ml/outputs/…/codegen/model_data.{h,cc}  ── make -C ml to-nrf (cp) ──▶  firmware/nrf5340/src/
ml/outputs/…/results.json  ◀── ble-cs results record (stdin) ──  firmware/…/build/<v>/run.json
```

*Warum:* Beide Seiten ändern sich unabhängig und laufen in unterschiedlichen Containern mit
unterschiedlichen Toolchains. Eine neue Zielplattform (STM32N6, Jetson) braucht nur ein eigenes
`firmware/<ziel>/`, das C-Code (oder ONNX) entgegennimmt und eine `run.json` im gleichen Format
liefert. Am ML-Code ändert sich nichts. Die einzige Stelle, die beide Seiten kennt, sind die
Makefiles, und genau dafür sind sie da.

### Reproduzierbar: Code in Git, Erzeugtes nicht

In Git liegen Code, Konfiguration, Lockfiles und ein kleiner Datenausschnitt. **Nicht** in Git:
Checkpoints, Exporte, generierter C-Code, Build-Ordner. Alles davon lässt sich aus Code +
Daten + Checkpoint neu erzeugen. Trainings-Einstellungen sind feste Konstanten mit festem Seed
(`ml/src/ble_cs_profiling/config.py`): Zweimal trainieren ergibt dasselbe Ergebnis.

### Rückverfolgbar: vom Modell bis zur Messung

Jede Messung muss sich eindeutig einem Modell zuordnen lassen. Deshalb gibt die Firmware den
sha256 der `.tflite` aus, die sie wirklich enthält. Passt er zu keinem exportierten Modell,
wird die Messung abgelehnt (z. B. wenn die Firmware nach dem Codegen nicht neu gebaut wurde).
Wird ein Modell neu exportiert und ändert sich dabei, fliegen alte Messungen automatisch raus.

### Feste Ablage, feste Namen

`ml/outputs/<checkpoint>/<präzision>/model.<format>`: Jeder weiß ohne Nachfragen, wo welche
Datei liegt, und Skripte können sich darauf verlassen. Der Ordnername kommt vom Checkpoint,
nicht vom Modelltyp, damit Varianten desselben Modells (z. B. zwei Trainingsläufe) sich nicht
überschreiben.

### Messen statt raten

Simulator, Austauschformat und Board liefern unterschiedliche Zahlen. Ein Beispiel aus diesem
Repo: INT4- statt INT8-Gewichte haben beim MobileNet den Flash um ein Drittel verkleinert, aber
die Zyklen um 35 % und die Tensor-Arena auf das 5,6-Fache erhöht, weil TFLite Micro die
INT4-Gewichte zur Laufzeit in einen INT8-Puffer entpackt. Das sieht man nur, wenn man misst.

---

## 4. Auf das eigene Projekt übertragen

### Was du übernehmen solltest

Die drei Welten, die Ordnerstruktur (`ml/`, `firmware/<ziel>/`, `docs/`, `compose.yaml`),
Docker für jeden Schritt, Makefiles als Einstieg, Checkpoint als Übergabe, feste Ablage,
`results.json` + `run.json` als Verträge, Messen auf echter Hardware.

### Was du mit hoher Wahrscheinlichkeit neu schreiben musst

| Bereich         | Hier (BLE-Beispiel)                    | Bei dir vermutlich                                                                                      |
| --------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Daten           | Parquet, 300 IQ-Spalten, `config.py` | Bilder, Zeitreihen, Radar-Punktwolken, Graphen …                                                       |
| Vorverarbeitung | keine (Rohdaten)                       | Normalisierung, Fenster, Feature-Extraktion. **Muss auf dem Gerät identisch nachgebaut werden!** |
| Metrik          | MAE (Regression)                       | Accuracy, F1, mAP …                                                                                    |
| Modelle         | MLP, MobileNetV3                       | dein Modell, in der Registry eingetragen                                                                |
| Export          | TFLite (litert-torch), ONNX            | hängt von Modell und Ziel ab: der schwierigste Teil                                                    |
| Firmware        | Zephyr + TFLM auf nRF5340              | andere MCU, NPU-Toolchain, Linux-Board (Jetson)                                                         |

### Typische Stolpersteine je Modellart

- **Klassisches ML (GBDT, Random Forest):** kein PyTorch. Das mentale Modell bleibt gleich:
  Modell speichern → Austauschformat (z. B. ONNX über `skl2onnx`) oder direkt C-Code (z. B.
  `emlearn`, `m2cgen`) → Gerät. Oft der einfachste Weg auf einen Mikrocontroller.
- **CNNs (wie hier):** gut unterstützt. Achte auf Operatoren, die die Ziel-Runtime nicht
  fusionieren kann oder gar nicht unterstützt (hier z. B. HardSwish, siehe
  [onnx_export_stm32n6.md](onnx_export_stm32n6.md)), und auf Kanalzahlen, die zur SIMD-Breite
  bzw. NPU passen.
- **Transformer / ViTs:** Softmax, LayerNorm und Attention sind schwer zu quantisieren und auf
  vielen Beschleunigern nicht unterstützt (laufen dann langsam auf der CPU). Früh prüfen, ob das
  Ziel die Operatoren kann.
- **Graph-NNs:** dynamische Graphen (variable Knotenanzahl) lassen sich kaum in ein statisches
  Austauschformat bringen. Häufig ist eine feste Maximalgröße mit Padding nötig.

Die Firmware muss jeden verwendeten Operator registrieren (hier die Liste in
`firmware/nrf5340/src/inference.cpp`). Ein neues Modell bedeutet oft auch, dass diese Liste
angepasst werden muss.

### Empfohlenes Vorgehen: erst durchstechen, dann verbessern

1. **Zuerst die ganze Kette mit einem dummen Modell** (z. B. ein winziges MLP) bis zur
   Messung auf der Hardware. So findest du die harten Probleme (Export, Speicher, Toolchain)
   am Anfang statt in der letzten Woche.
2. Dann das echte Modell einsetzen und jeden Schritt einzeln prüfen: Läuft der Export? Stimmt
   die Ausgabe mit PyTorch überein? Wie viel Genauigkeit kostet INT8? Passt es in den Speicher?
3. Erst danach optimieren (Architektur, Quantisierung, Pruning), und jede Änderung messen.

---

## 5. Begriffe

| Begriff             | Bedeutung                                                                      |
| ------------------- | ------------------------------------------------------------------------------ |
| Checkpoint          | Gespeichertes trainiertes Modell (Gewichte + Metadaten), hier `.pt`          |
| Quantisierung (PTQ) | float32 → int8 nach dem Training. Kleiner und schneller, etwas ungenauer      |
| Kalibrierung        | Repräsentative Daten bestimmen die Wertebereiche für die Quantisierung       |
| full INT8           | Auch Ein- und Ausgang sind int8 (keine float-Rechnung mehr auf dem Gerät)     |
| QDQ                 | Quantize/Dequantize-Knoten im ONNX-Graphen, Ein-/Ausgang bleibt float          |
| TFLite Micro (TFLM) | Runtime für TFLite-Modelle auf Mikrocontrollern, ohne Betriebssystem-Features |
| CMSIS-NN            | Optimierte Kernels von Arm für Cortex-M (SIMD-Befehle)                        |
| Tensor-Arena        | Fester RAM-Block, in dem TFLM alle Zwischenergebnisse ablegt                   |
| Flash / RAM         | Programmspeicher (Code, Modell) / Arbeitsspeicher (Variablen, Arena)           |
| Renode              | Simulator für Mikrocontroller: gut für Vergleiche und CI, nicht zyklengenau  |
| HIL                 | Hardware-in-the-Loop: Test/Messung auf dem echten Board                        |
| sha256              | Prüfsumme einer Datei. Gleiche Datei = gleicher Wert, eindeutige Zuordnung    |
