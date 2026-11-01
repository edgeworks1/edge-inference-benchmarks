# STM32N6 (NUCLEO-N657X0-Q): Einrichtung unter Windows

Dieser Ordner bringt ein exportiertes Modell (`.tflite` oder `.onnx` aus `ml/outputs/`) auf die
Neural-ART-NPU des STM32N6, misst es auf dem echten Board und liest das Profiling zurück. Alles
läuft über die Kommandozeile, ohne GUI. Diese Anleitung richtet die Werkzeuge einmalig ein und
geht die Kette Schritt für Schritt von Hand durch. Ein Makefile, das diese Schritte automatisiert,
baut darauf auf.

```
model.tflite / model.onnx
   │  stedgeai generate        NPU-Compiler: network.c + Gewichte (.raw)
   ▼
st_ai_output/
   │  n6_loader.py             baut die ST-Testanwendung, flasht die Gewichte ins externe Flash,
   ▼                           lädt die Firmware per Debugger in den RAM und startet sie
NUCLEO-N657X0-Q
   │  AiRunner / validate      Inferenzen über den virtuellen COM-Port des ST-LINK
   ▼
Latenz pro Epoche, Zyklen, Genauigkeit gegen das Host-Modell
```

**Warum Windows und nicht WSL/Docker wie beim nRF5340?** Technisch ginge beides, die ST-Werkzeuge
gibt es auch für Linux. Zwei praktische Gründe sprechen dagegen:

- Die ST-Downloads liegen hinter einem myST-Login mit Lizenzabfrage. Ein Dockerfile kann sie nicht
  selbst herunterladen, und ein fertiges Image darf wegen der Lizenz nicht öffentlich verteilt werden.
- Den ST-LINK an WSL durchzureichen (`usbipd bind`) braucht einmal Adminrechte. Unter Windows hängt
  das Board dagegen ohne weiteres Zutun am Rechner.

## Getestete Versionen

| Komponente | Version |
|---|---|
| Windows | 11, ohne Adminrechte im Alltag |
| ST Edge AI Core | 4.0.1 (STM32CubeAI 12.0.1, Neural-ART-Compiler 1.1.3) |
| STM32CubeIDE | 1.18.1 (GNU Tools for STM32 13.3.rel1, mitgelieferter CubeProgrammer 2.19) |
| STM32CubeProgrammer (optional, eigenständig) | 2.23.0 |
| Python | 3.11 (über uv) |
| Board | NUCLEO-N657X0-Q mit STLINK-V3EC |

Ältere oder neuere Versionen können funktionieren, sind aber nicht geprüft. Gerade bei
STM32CubeIDE 2.x können sich die internen Ordnernamen ändern, nach denen das ST-Ladeskript sucht.

## Überblick: was du installierst

| Werkzeug | Wofür | Adminrechte |
|---|---|---|
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | `stedgeai` (NPU-Compiler), ST-Testanwendung, Lade- und Messskripte | je nach Zielordner |
| [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) | Arm-GCC, `make`, `ST-LINK_gdbserver`, `STM32_Programmer_CLI`, ST-LINK-Treiber | ja, einmalig (Treiber) |
| [uv](https://docs.astral.sh/uv/) | holt Python 3.11 und legt die virtuelle Umgebung an | nein |

Mehr nicht. Arm-Compiler, `make` und Programmer werden **nicht** separat installiert und **nicht**
in den `PATH` eingetragen: Das ST-Ladeskript holt sie sich selbst aus der CubeIDE-Installation.
Die eigenständige [STM32CubeProgrammer](https://www.st.com/en/development-tools/stm32cubeprog.html)-GUI
ist optional (siehe [Stolperfallen](#stolperfallen)).

Alle ST-Downloads brauchen einen kostenlosen myST-Account.

## 1. Installation

### 1.1 ST Edge AI Core

1. Auf der [Produktseite](https://www.st.com/en/development-tools/stedgeai-core.html) den
   Windows-Installer herunterladen und starten.
2. Als Komponente mindestens **STM32 / Neural-ART (STM32N6)** auswählen.
3. Zielordner: einen Ordner, in den du **ohne Adminrechte schreiben** kannst, z. B. `C:\ST\STEdgeAI\4.0`.
   Nicht `C:\Program Files`, denn das Ladeskript schreibt bei jedem Lauf in die ST-Testanwendung
   innerhalb dieser Installation (siehe [Stolperfallen](#stolperfallen)).

Prüfen:

```bat
C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe --version
```

Erwartet: `ST Edge AI Core v4.0.1-…` und darunter u. a. `STM32CubeAI 12.0.1`.

### 1.2 STM32CubeIDE

1. Den Windows-Installer von der [Produktseite](https://www.st.com/en/development-tools/stm32cubeide.html)
   ausführen (braucht einmal Adminrechte, weil er den ST-LINK-Treiber installiert).
2. Zielordner z. B. `C:\ST\STM32CubeIDE_1.18.1`. Die IDE selbst musst du nie öffnen.

Prüfen, ob der Treiber da ist: Board über die **ST-LINK-Buchse** anstecken (siehe
[Abschnitt 4](#4-board-vorbereiten)). Im Geräte-Manager erscheint unter „Anschlüsse (COM & LPT)“
der Eintrag **STMicroelectronics STLink Virtual COM Port (COMx)**.

### 1.3 uv

In einer normalen PowerShell (keine Adminrechte nötig):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Danach ein **neues** Terminal öffnen und prüfen: `uv --version`.

uv lädt das passende Python selbst herunter. Ein bereits installiertes Python (msys2, Chocolatey,
Microsoft Store) wird dafür nicht gebraucht und sollte auch nicht benutzt werden, siehe
[Stolperfallen](#stolperfallen).

## 2. Pfade

Die Pfade sind **Make-Variablen** in [Makefile](Makefile) mit Standardwerten. Windows-Umgebungsvariablen
werden nicht gebraucht:

| Variable | Standard | Bedeutung |
|---|---|---|
| `STEDGEAI_DIR` | `C:/ST/STEdgeAI/4.0` | Wurzel der ST-Edge-AI-Installation |
| `CUBEIDE_DIR` | `C:/ST/STM32CubeIDE_1.18.1/STM32CubeIDE` | Ordner, in dem `stm32cubeide.exe` liegt (darunter `plugins\`) |
| `N6_WORK_DIR` | `C:/work/n6` | Arbeitsordner auf einem **lokalen Windows-Laufwerk**: venv, `config.json`, kompiliertes Modell |
| `CUBEPROG_CLI` | Standard-Installationspfad der CubeProgrammer | eigenständiger Programmer; wird genommen, wenn er existiert |

Abweichende Pfade pro Aufruf überschreiben: `make -C firmware/stm32n6 setup N6_WORK_DIR=D:/n6`.
Immer Windows-Pfade mit `/`.

**Warum keine Windows-Umgebungsvariablen (`setx`)?** Prozesse, die aus WSL gestartet werden,
übernehmen die Umgebung, die WSL beim Start von Windows bekommen hat. Mit `setx` gesetzte Variablen
sieht WSL erst nach `wsl --shutdown`, und an Windows-Programme reicht WSL sie nur weiter, wenn sie
zusätzlich in `WSLENV` stehen. Make-Variablen wirken sofort, stehen sichtbar im Makefile und gelten
nur für den einen Aufruf.

Für die Kette von Hand in `cmd` (Abschnitte 3 und 5) die Werte nur im aktuellen Fenster setzen:

```bat
set STEDGEAI_CORE_DIR=C:\ST\STEdgeAI\4.0
set STM32CUBEIDE_DIR=C:\ST\STM32CubeIDE_1.18.1\STM32CubeIDE
set N6_WORK_DIR=C:\work\n6
```

## 3. Arbeitsordner und Python-Umgebung

In einem neuen `cmd`-Fenster:

```bat
mkdir "%N6_WORK_DIR%"
cd /d "%N6_WORK_DIR%"
uv venv .venv --python 3.11
.venv\Scripts\activate
uv pip install numpy "protobuf==3.20.3" "pyserial==3.4" "tqdm==4.50.2" colorama
```

Das sind die Pakete aus `%STEDGEAI_CORE_DIR%\scripts\ai_runner\requirements.txt`, mit zwei
Korrekturen: `colorama` fehlt dort, wird aber gebraucht, und `protobuf` muss auf 3.20.x festgelegt
werden (siehe [Stolperfallen](#stolperfallen)).

Die `config.json` für das ST-Ladeskript anlegen. Sie braucht nur den Pfad zur CubeIDE, alle
Werkzeuge findet das Skript darunter selbst. Der Befehl erzeugt die Datei aus der Umgebungsvariablen,
damit Schrägstriche und Kodierung stimmen:

```bat
python -c "import json, os; json.dump({'compiler_type': 'gcc', 'cubeide_path': os.environ['STM32CUBEIDE_DIR'].replace(chr(92), '/')}, open('config.json', 'w'), indent=2)"
type config.json
```

Prüfen, ob das Ladeskript alle Werkzeuge findet und die venv vollständig ist:

```bat
python "%STEDGEAI_CORE_DIR%\scripts\N6_scripts\n6_loader.py" --help
set PYTHONPATH=%STEDGEAI_CORE_DIR%\scripts\ai_runner
python -c "from stm_ai_runner import AiRunner; import google.protobuf as p; print('AiRunner OK, protobuf', p.__version__)"
```

## 4. Board vorbereiten

Das NUCLEO-N657X0-Q hat **zwei** USB-C-Buchsen. Nur die mit **ST-LINK** beschriftete führt zum
Debugger mit virtuellem COM-Port. Die andere (neben der Ethernet-Buchse) geht direkt an den
STM32N6 und liefert ohne passende Firmware kein USB-Gerät, kann das Board aber mit Strom versorgen.

Getesteter Aufbau mit **zwei Kabeln**:

1. **Daten:** USB-Kabel vom PC in die **ST-LINK-Buchse**. Ein Kabel nehmen, das nachweislich
   Daten überträgt.
2. **Strom:** zweites USB-C-Kabel in die **andere Buchse** (neben Ethernet), an einen USB-C-Port
   des PCs oder ein USB-C-Netzteil. Jumper **CN9 auf 3-4** (Versorgung über diese Buchse).
3. Boot-Jumper auf **Development-Mode**: **JP2 (BOOT1) auf Position 2** (Pins 2-3), JP1 (BOOT0)
   bleibt auf Position 1. Darin startet der STM32N6 keine Firmware aus dem externen Flash, und der
   Debugger kann die Testanwendung in den RAM laden. Im Normalbetrieb (Boot aus dem Flash) stehen
   beide auf Position 1. Die Positionen sind auf der Platine aufgedruckt; in Foren werden JP1 und
   JP2 manchmal vertauscht benannt, maßgeblich ist das Board-Handbuch
   [UM3417](https://www.st.com/en/evaluation-tools/nucleo-n657x0-q.html#documentation).
4. Einmal **Reset** drücken oder das Board neu anstecken.

Für die Messungen bleibt das Board dauerhaft in diesem Modus.

Den COM-Port findest du im Geräte-Manager oder mit:

```bat
python -m serial.tools.list_ports -v
```

Gesucht ist der Eintrag mit `VID:PID=0483:3754` (STLINK-V3EC).

## 5. Die Kette einmal von Hand

Alle Befehle im `cmd`-Fenster mit aktivierter venv, im Arbeitsordner. Als Beispiel das kleine MLP.
Modelle aus dem WSL-Repo zuerst auf das Windows-Laufwerk **kopieren**. `REPO` ist der Pfad des
Repos, wie Windows ihn sieht (in WSL: `wslpath -w .` im Repo-Ordner):

```bat
cd /d "%N6_WORK_DIR%"
.venv\Scripts\activate
set PYTHONPATH=%STEDGEAI_CORE_DIR%\scripts\ai_runner
set STEDGEAI=%STEDGEAI_CORE_DIR%\Utilities\windows\stedgeai.exe
set REPO=\\wsl.localhost\Ubuntu-24.04\home\<wsl-benutzer>\Projekte\edge-inference-benchmarks
copy "%REPO%\ml\outputs\edge_bench_mlp\int8_full\model.tflite" mlp_int8.tflite
```

Welche Datei: `int8_full/model.tflite` (INT8 inklusive Ein- und Ausgang) oder
`int8_qdq/model.onnx` (INT8 im Inneren, float am Ein- und Ausgang). Beide laufen auf der NPU.
FP32-Modelle laufen nur auf der CPU, weil die NPU INT8 braucht.

### 5.1 Analyse (ohne Board)

```bat
"%STEDGEAI%" analyze -m mlp_int8.tflite --target stm32n6 --st-neural-art -o out_analyze
```

### 5.2 NPU-Kompilierung (ohne Board)

```bat
"%STEDGEAI%" generate -m mlp_int8.tflite --target stm32n6 --st-neural-art -o st_ai_output
```

Worauf achten:

- **Epochs details:** wie viele Epochen `HW`, `SW` oder `hybrid` sind. `SW` bedeutet, dass ein Teil
  des Modells auf der CPU läuft statt auf der NPU.
- **Memory usage information:** Die Zeile `hyperRAM` muss bei **0 B** stehen. Das Nucleo hat kein
  externes RAM, siehe [Stolperfallen](#stolperfallen).
- In `st_ai_output\` liegen danach u. a. `network.c` und `network_atonbuf.xSPI2.raw` (die Gewichte
  für das externe Flash).

### 5.3 Auf das Board laden

```bat
python "%STEDGEAI_CORE_DIR%\scripts\N6_scripts\n6_loader.py" --config config.json --build-config N6-Nucleo --network-file "%N6_WORK_DIR%\st_ai_output\network.c" --project-path "%STEDGEAI_CORE_DIR%\Projects\STM32N6570-DK\Applications\NPU_Validation"
```

Das dauert beim ersten Mal einige Minuten (Build der Testanwendung). Am Ende muss stehen:

```
… Flashing memory xSPI2 -- 21.921 kB
… Building project (conf= N6-Nucleo)
… Loading internal memories & Running the program
… Start operation achieved successfully
```

Die Testanwendung liegt jetzt **nur im RAM**. Nach einem Stromausfall oder Ausstecken ist sie weg,
dann diesen Schritt wiederholen.

### 5.4 Messen und Profiling zurücklesen

`COM5` durch deinen Port ersetzen. Die Baudrate **921600** immer mit angeben.

Latenz gesamt und pro Epoche:

```bat
python "%STEDGEAI_CORE_DIR%\scripts\ai_runner\examples\checker.py" -d serial:COM5:921600 --perf-only -b 10
```

Epochen den Original-Layern zugeordnet (laut ST experimentell):

```bat
python "%STEDGEAI_CORE_DIR%\scripts\ai_runner\examples\npu_profiler.py" -d serial:COM5:921600 -c "%N6_WORK_DIR%\st_ai_output\network.c" -b 10
```

Genauigkeit des Boards gegen das Modell auf dem PC, mit Bericht in
`st_ai_output\network_validate_report.txt`:

```bat
"%STEDGEAI%" validate -m mlp_int8.tflite --target stm32n6 --mode target -d serial:COM5:921600
```

## 6. Die Ausgabe lesen

Ausschnitt aus `checker.py` für das MLP:

```
 duration                       :   0.100 ms by sample (0.098/0.113/0.005)
 CPU cycles                     :   [27,284 45,097 7,304]

  c_id    m_id   type            dur (ms)        %     cumul  CPU cycles                   name
  0       -      epoch              0.073   73.46%    73.46%  [ 15,794  39,287   3,457 ]   EpochBlock_2
  1       -      epoch              0.019   19.05%    92.51%  [  8,125   4,982   2,075 ]   EpochBlock_3
  2       -      epoch              0.007    7.49%   100.00%  [  3,365     828   1,772 ]   EpochBlock_4
  HW: 56.6%, SW: 0.0%, SW ctrl: 43.4%
```

- **Epoche statt Layer:** Die NPU arbeitet den Graphen in Epochen ab. Eine Epoche kann mehrere
  Original-Layer enthalten (z. B. Dense + ReLU). Die Zuordnung zeigt `npu_profiler.py`.
- **Drei Zyklenwerte** pro Zeile: Vorbereitung, Hardware (NPU rechnet), Nachbereitung, jeweils in
  MCU-Zyklen bei 800 MHz.
- **`HW` / `SW` / `SW ctrl`:** Anteil, in dem die NPU rechnet, die CPU Operatoren ausführt bzw. die
  CPU die NPU nur ansteuert. Bei sehr kleinen Modellen ist `SW ctrl` hoch, weil die Ansteuerung
  im Verhältnis zur Rechenarbeit teuer ist.
- **Taktung:** Die Testanwendung läuft im Overdrive-Modus (MCU 800 MHz, NPU 1 GHz), siehe
  `device.attrs` in der Ausgabe.

## Stolperfallen

### Board und USB

- **Kein COM-Port im Geräte-Manager, LEDs leuchten trotzdem:** falsche USB-C-Buchse oder ein Kabel,
  das nur lädt. Siehe [Abschnitt 4](#4-board-vorbereiten).
- **COM-Port da, aber Laden/Flashen scheitert mit nur einem Kabel:** Über die ST-LINK-Buchse allein
  (CN9 auf 1-2, `STLK`) reichte die Versorgung im Test nicht für den STM32N6 mit der
  Testanwendung (CPU 800 MHz, NPU 1 GHz). Mit CN9 auf 3-4 ohne zweites Kabel ist der Chip ganz
  ohne Strom. Lösung: zwei Kabel wie in Abschnitt 4.
- **LD9 (COM) leuchtet dauerhaft rot:** normal, der ST-LINK ist am PC angemeldet und gerade ohne
  Debug-Verbindung. Während einer Verbindung blinkt sie rot/grün.
- **Verbindung schlägt fehl oder Timeout beim Messen:** Es darf nur ein Programm gleichzeitig auf
  ST-LINK und COM-Port zugreifen. STM32CubeIDE, die CubeProgrammer-GUI, STM32Cube AI Studio und
  serielle Terminals vorher schließen.
- **Nach dem Ausstecken antwortet das Board nicht mehr:** Die Testanwendung lag nur im RAM.
  Schritt 5.3 wiederholen.
- **`validate` bekommt keine Antwort:** Ohne Angabe nimmt `stedgeai` 115200 Baud, die Testanwendung
  spricht 921600. Immer `-d serial:COMx:921600` angeben.
- **Die COM-Nummer ändert sich:** an einem anderen USB-Port oder Dock vergibt Windows oft eine neue
  Nummer. Mit `python -m serial.tools.list_ports -v` nachsehen.
- **ST-LINK-Firmware-Update:** Bietet ein ST-Werkzeug ein Update des ST-LINK an, das Update nicht
  abbrechen und das Board währenddessen nicht ausstecken. Im ST-Forum gibt es Berichte über
  danach unbrauchbare ST-LINKs.

### Python

- **Nicht das msys2- oder Chocolatey-Python benutzen.** Dessen `venv` legt `bin\` statt
  `Scripts\` an, `activate` funktioniert dann in `cmd` nicht („Das System kann den angegebenen Pfad
  nicht finden“). Mit `where python` siehst du, welches Python zuerst gefunden wird. uv umgeht das,
  weil es sein eigenes Python mitbringt.
- **`Descriptors cannot be created directly` / „generated code is out of date“:** Die
  `*_pb2.py`-Dateien von ST sind mit einem alten protoc erzeugt und laufen nicht mit protobuf ≥ 4.
  Lösung: `protobuf==3.20.3`.
- **`No module named 'colorama'`:** fehlt in der ST-`requirements.txt`, einfach nachinstallieren.
- **`No module named 'stm_ai_runner'`:** `PYTHONPATH` ist im aktuellen Fenster nicht gesetzt.
  `PYTHONPATH` bewusst **nicht** dauerhaft als Umgebungsvariable setzen, sonst sehen alle anderen
  Python-Umgebungen auf dem Rechner die ST-Module.

### Werkzeuge und Pfade

- **Andere `make`- oder `arm-none-eabi-gcc`-Versionen im `PATH`** (Chocolatey, msys2, ältere
  Toolchains): stören das Ladeskript nicht, weil es über `cubeide_path` die Werkzeuge aus der
  CubeIDE nimmt. Beim Aufruf von Hand bekommst du aber das erste aus dem `PATH`. Prüfen mit
  `where make arm-none-eabi-gcc`.
- **`setx PATH …` nicht benutzen:** `setx` kürzt Werte auf 1024 Zeichen. Beim üblichen
  `setx PATH "%PATH%;…"` landet außerdem der komplette System-`Path` als Kopie im Benutzer-`Path`.
  Den `Path` nur über den Dialog „Umgebungsvariablen für dieses Konto bearbeiten“ ändern.
- **Benutzer- statt System-`Path` reicht:** Windows hängt den Benutzer-`Path` an den System-`Path`
  an. Bei gleichnamigen Programmen gewinnt deshalb das aus dem System-`Path`.
- **`config.json` wird nicht gelesen (`Bad Json syntax … at line: ï»¿{`):** Die Datei wurde mit
  BOM gespeichert, z. B. von Windows PowerShell 5.1 mit `Set-Content -Encoding utf8`. Mit dem
  `python -c …`-Befehl aus Abschnitt 3 erzeugen oder im Editor als „UTF-8“ ohne BOM speichern. In
  der Datei immer Schrägstriche `/` verwenden, auch unter Windows.
- **Ladeskript startet langsam:** Ohne weitere Angaben durchsucht es alle CubeIDE-Plugins nach den
  Werkzeugen, das kann über eine Minute dauern. Schneller wird es, wenn du die Pfade in der
  `config.json` direkt angibst (Beispiele in `%STEDGEAI_CORE_DIR%\scripts\N6_scripts\config.json`).
- **Mehrere CubeIDE-Versionen installiert:** Das Ladeskript sucht nur unterhalb von `cubeide_path`
  und nimmt dort bei mehreren Plugin-Versionen die neueste.
- **Den eigenständigen CubeProgrammer verwenden:** Funktioniert das Flashen mit dem
  CubeIDE-Programmer nicht, die [STM32CubeProgrammer](https://www.st.com/en/development-tools/stm32cubeprog.html)
  installieren und in der `config.json` zusätzlich
  `"cubeProgrammerCLI_binary_path": "C:/Program Files/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI.exe"`
  eintragen. Der Ablauf wurde vollständig mit Version 2.23 getestet.
- **`compiler "gcc:arm-none-eabi-gcc" is not in the PATH` bei `analyze`/`generate`:** harmlos.
  Es fehlt nur die Abschätzung der Code-Größe der Laufzeitbibliothek, die eigentliche Analyse läuft.
- **Keine WSL-Pfade (`\\wsl.localhost\…`) an die ST-Werkzeuge übergeben.** `cmd` kann nicht in
  UNC-Pfade wechseln, `make` scheitert daran, und der Zugriff ist langsam. Deshalb liegt alles in
  `%N6_WORK_DIR%` auf einem lokalen Laufwerk, Modelle werden dorthin kopiert.

### ST Edge AI

- **`stedgeai` scheint zu hängen** (z. B. bei `PASS: 0%`): Der Start dauert oft zehn Sekunden bis
  über eine Minute, je nach Virenscanner. Kein Grund zum Abbrechen.
- **Das Ladeskript verändert die ST-Installation:** Es kopiert `network.c` in
  `%STEDGEAI_CORE_DIR%\Projects\STM32N6570-DK\Applications\NPU_Validation\` und baut dort nach
  `armgcc\build\`. Das Projekt lässt sich nicht einfach herauskopieren, weil es über relative Pfade
  auf `Middlewares\` der Installation zugreift. Folgen: Die Installation muss beschreibbar sein, und
  es kann immer nur **ein** Modell gleichzeitig geladen werden.
- **Der Projektname enthält „STM32N6570-DK“**, gilt aber auch für das Nucleo: Das Board wählt
  `--build-config N6-Nucleo` (bzw. `N6-Nucleo-USB` für den schnelleren USB-Kanal, dafür ist ein
  zweites Kabel an der anderen Buchse nötig).
- **Das Nucleo hat kein externes RAM:** Die Standard-Speicherkonfiguration des NPU-Compilers enthält
  aber das `hyperRAM` des Discovery-Kits. Kleine und mittlere Modelle (z. B. die MobileNets in
  diesem Repo) passen in den internen Speicher. Zeigt der `generate`-Bericht bei `hyperRAM` mehr als
  0 B, läuft das Modell auf dem Nucleo nicht. Dann braucht es ein kleineres Modell oder eine eigene
  Speicherkonfiguration (`.mpool`).
- **ONNX-Opset:** zur Wahl zwischen opset 13 und 14 siehe
  [docs/onnx_export_stm32n6.md](../../docs/onnx_export_stm32n6.md).

## Weiterführend

Die vollständige ST-Doku liegt lokal unter `%STEDGEAI_CORE_DIR%\Documentation\`, vor allem:

- `stneuralart_getting_started.html`: der hier beschriebene Ablauf aus Sicht von ST, inklusive
  Fehlersuche
- `command_line_interface.html`: alle Optionen von `stedgeai`
- `ai_runner_python_module.html`: die Python-API zum Messen, Grundlage für die Automatisierung
- `stneuralart_operator_support.html` (online im ST-Doku-Portal): welche Operatoren auf der NPU laufen
