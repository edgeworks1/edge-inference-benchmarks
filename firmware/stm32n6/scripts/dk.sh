#!/usr/bin/env bash
# Host-side helpers for the STM32N6 (NUCLEO-N657X0-Q). The ST tools are Windows programs: they run
# through WSL interop inside the Windows work folder $N6_WORK_DIR. Called by the Makefile.
#
#   dk.sh setup              create or update the work folder, venv and config.json
#   dk.sh generate <model>   setup, then copy a .tflite/.onnx into the work folder and compile it
#   dk.sh load               build the ST validation app with the compiled model and start it
#   dk.sh measure <model> <run.json>   profile the loaded model and write run.json
#
# Configuration (Windows paths with forward slashes, exported by the Makefile):
# STEDGEAI_DIR, CUBEIDE_DIR, N6_WORK_DIR and optionally CUBEPROG_CLI.
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(wslpath -u "$N6_WORK_DIR")
STEDGEAI=$(wslpath -u "$STEDGEAI_DIR")/Utilities/windows/stedgeai.exe
readonly HERE WORK STEDGEAI
readonly PYTHON=.venv/Scripts/python.exe
# ST's test application; the loader copies network.c into it and builds it in place
readonly VALIDATION_APP=Projects/STM32N6570-DK/Applications/NPU_Validation

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

fail() {
    echo "dk.sh: $*" >&2
    exit 1
}

# Windows programs need a current directory on a Windows drive and never get a WSL path:
# cmd cannot enter \\wsl.localhost\..., and make inside the ST scripts fails on it.
in_work_dir() {
    (cd "$WORK" && "$@")
}

# Our helper scripts run with the venv's Windows Python, so they are copied next to it.
copy_scripts() {
    mkdir -p "$WORK/scripts"
    cp "$HERE"/scripts/*.py "$WORK/scripts/"
}

# Read by n6_loader.py: forward slashes only and no BOM. The standalone CubeProgrammer is used
# when it is installed, otherwise the loader takes the one bundled with STM32CubeIDE.
write_loader_config() {
    local programmer=""
    if [[ -n ${CUBEPROG_CLI:-} && -f $(wslpath -u "$CUBEPROG_CLI") ]]; then
        programmer=$(printf ',\n  "cubeProgrammerCLI_binary_path": "%s"' "$CUBEPROG_CLI")
    fi
    printf '{\n  "compiler_type": "gcc",\n  "cubeide_path": "%s"%s\n}\n' \
        "$CUBEIDE_DIR" "$programmer" > "$WORK/config.json"
    cat "$WORK/config.json"
}

# Idempotent and fast (~1.5 s once installed), so generate runs it every time: the venv always
# matches requirements.txt in the repo, also after a git pull.
setup() {
    [[ -f $STEDGEAI ]] || fail "stedgeai not found: $STEDGEAI (set STEDGEAI_DIR)"
    [[ -d $(wslpath -u "$CUBEIDE_DIR")/plugins ]] || fail "no STM32CubeIDE in $CUBEIDE_DIR (set CUBEIDE_DIR)"
    command -v uv.exe > /dev/null || fail "uv.exe not found (install uv on Windows)"

    mkdir -p "$WORK"
    cp "$HERE/requirements.txt" "$WORK/"
    in_work_dir uv.exe venv --allow-existing --python 3.11 .venv
    in_work_dir uv.exe pip install --python "$PYTHON" -r requirements.txt
    write_loader_config
    copy_scripts
}

# A fresh output folder per model, so a failed compile can never leave the previous model behind.
generate() {
    local model=$1 name
    [[ -f $model ]] || fail "model not found: $model (run make -C ml to-stm32n6 first)"
    name=$(basename "$model")

    setup
    cp "$model" "$WORK/$name"
    rm -rf "$WORK/st_ai_output"
    # No stdin: without a terminal (CI, IDE task, piped make) stedgeai otherwise hangs forever
    in_work_dir "$STEDGEAI" generate -m "$name" --target stm32n6 --st-neural-art -o st_ai_output \
        < /dev/null
    in_work_dir "$PYTHON" scripts/run_json.py check st_ai_output/network_c_info.json
}

# Flashes the weights into the external flash, then loads the app into RAM through the debugger.
# The loader does not report failure in its exit code, only in its last line.
load() {
    [[ -f $WORK/st_ai_output/network.c ]] || fail "no compiled model in $N6_WORK_DIR (run make generate first)"
    in_work_dir "$PYTHON" "$STEDGEAI_DIR/scripts/N6_scripts/n6_loader.py" --config config.json \
        --build-config N6-Nucleo --network-file "$N6_WORK_DIR/st_ai_output/network.c" \
        --project-path "$STEDGEAI_DIR/$VALIDATION_APP" 2>&1 | tee "$WORK/loader.log"
    grep -q "Start operation achieved successfully" "$WORK/loader.log" ||
        fail "loading failed, see $N6_WORK_DIR/loader.log (board powered, ST tools closed?)"
}

# <model> names the file that was compiled; its work folder copy is what gets hashed.
# TARGET (exported by the Makefile) is the device name in results.json.
measure() {
    local name run_json=$2
    name=$(basename "$1")
    copy_scripts
    in_work_dir "$PYTHON" scripts/run_json.py measure "$name" st_ai_output/network_c_info.json \
        run.json --target "${TARGET:-stm32n6}" --ai-runner "$STEDGEAI_DIR/scripts/ai_runner"
    mkdir -p "$(dirname "$run_json")"
    cp "$WORK/run.json" "$run_json"
}

case "${1:-}" in
    setup) (($# == 1)) || usage; setup ;;
    generate) (($# == 2)) || usage; generate "$2" ;;
    load) (($# == 1)) || usage; load ;;
    measure) (($# == 3)) || usage; measure "$2" "$3" ;;
    *) usage ;;
esac
