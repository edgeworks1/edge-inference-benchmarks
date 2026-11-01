"""Profile the model running on the NUCLEO-N657X0-Q and write ``run.json`` for the ML side.

Runs on Windows with the work folder's venv; called by ``dk.sh``. Like the nRF's run_json.py,
this is the only place that turns device data into the device-independent ``run.json``:

- ``check <c_info>``: after compiling, fail if the model does not fit the Nucleo
- ``measure <model> <c_info> <run.json>``: profile the loaded model over the ST-LINK COM port
"""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

BAUD_RATE = 921600
STLINK_V3EC_USB_ID = (0x0483, 0x3754)  # its virtual COM port talks to the ST validation app
# The runner reports per inference: [pre, NPU, post, total, ...] MCU cycles; the total is used
TOTAL_CYCLES_INDEX = 3
# External RAM of the STM32N6570-DK; the compiler's default memory map includes it, the Nucleo
# has none. A model that places anything there cannot run on the Nucleo.
EXTERNAL_RAM_POOL = "hyperRAM"


# ============================================================
# Compiled model (network_c_info.json from stedgeai generate)
# ============================================================
def footprint(c_info: dict[str, Any]) -> dict[str, int]:
    """Return weights, activations and the generated network code in bytes."""
    memory = c_info["memory_footprint"]
    return {
        "weights_bytes": memory["weights"],
        "activations_bytes": memory["activations"],
        "network_code_bytes": memory["kernel_flash"],
        "network_ram_bytes": memory["kernel_ram"],
    }


def external_ram_bytes(c_info: dict[str, Any]) -> int:
    """Return how many bytes the compiler placed in external RAM."""
    return sum(
        pool["used_size_bytes"]
        for pool in c_info["memory_pools"]
        if pool["name"] == EXTERNAL_RAM_POOL
    )


def check(c_info: dict[str, Any]) -> None:
    """Print the footprint and exit with an error if the model needs external RAM."""
    for name, value in footprint(c_info).items():
        print(f"{name:20s} {value:>10,}")
    if external := external_ram_bytes(c_info):
        sys.exit(f"Model needs {external:,} B in {EXTERNAL_RAM_POOL}, the Nucleo has none.")


# ============================================================
# Board (ST's AiRunner over the ST-LINK virtual COM port)
# ============================================================
def find_stlink_port() -> str:
    """Return the COM port of the first connected STLINK-V3EC."""
    from serial.tools import list_ports

    for port in list_ports.comports():
        if (port.vid, port.pid) == STLINK_V3EC_USB_ID:
            return port.device
    sys.exit("No STLINK-V3EC COM port found (is the board connected to the ST-LINK socket?)")


def profile(port: str, inferences: int) -> dict[str, Any]:
    """Run inferences with random inputs on the loaded model; return ST's profiler data."""
    from stm_ai_runner import AiRunner

    runner = AiRunner()
    runner.connect(f"serial:{port}:{BAUD_RATE}")
    if not runner.is_connected:
        sys.exit(f"Cannot connect on {port}: {runner.get_error()} (run dk.sh load first)")
    try:
        session = runner.session(runner.names[0])
        inputs = runner.generate_rnd_inputs(session.name, batch_size=inferences)
        _, profiler = session.invoke(inputs, mode=AiRunner.Mode.PERF_ONLY, disable_pb=True)
    finally:
        runner.disconnect()
    return profiler


def measurement(profiler: dict[str, Any]) -> dict[str, Any]:
    """Reduce the per-inference values to medians, robust against a slow first inference."""
    cycles = [values[TOTAL_CYCLES_INDEX] for values in profiler["debug"]["counters"]["values"]]
    return {
        "cycles": int(statistics.median(cycles)),
        "latency_us": round(statistics.median(profiler["c_durations"]) * 1000),
        "inferences": len(cycles),
        "device": profiler["info"]["device"]["desc"],
    }


# ============================================================
# run.json
# ============================================================
def run_json(target: str, model: Path, c_info: dict[str, Any], profiler: dict[str, Any]) -> dict:
    """Combine model hash, board measurement and compiled footprint into the run.json format.

    Flash and RAM cover the model only (weights + network code, activations + network state),
    not the whole validation app as on the nRF.
    """
    memory = footprint(c_info)
    return {
        "target": target,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        **measurement(profiler),
        "arena_used_bytes": memory["activations_bytes"],
        "flash_bytes": memory["weights_bytes"] + memory["network_code_bytes"],
        "ram_bytes": memory["activations_bytes"] + memory["network_ram_bytes"],
    }


def main() -> None:
    """Dispatch to ``check`` or ``measure``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check_cmd = commands.add_parser("check", help="check the compiled model against the Nucleo")
    check_cmd.add_argument("c_info", type=Path)
    measure_cmd = commands.add_parser("measure", help="profile the loaded model, write run.json")
    measure_cmd.add_argument("model", type=Path, help="the .tflite/.onnx that was compiled")
    measure_cmd.add_argument("c_info", type=Path)
    measure_cmd.add_argument("output", type=Path)
    measure_cmd.add_argument("--ai-runner", type=Path, required=True, help="STEdgeAI ai_runner")
    measure_cmd.add_argument("--target", default="stm32n6", help="device name in results.json")
    measure_cmd.add_argument("--inferences", type=int, default=20)
    measure_cmd.add_argument("--port", help="COM port (default: first STLINK-V3EC)")
    args = parser.parse_args()

    c_info = json.loads(args.c_info.read_text())
    if args.command == "check":
        check(c_info)
        return

    # ST's runner module ships with STEdgeAI and is not installed into the venv
    sys.path.insert(0, str(args.ai_runner))
    port = args.port or find_stlink_port()
    print(f"Profiling {args.inferences} inferences on {port}")
    run = run_json(args.target, args.model, c_info, profile(port, args.inferences))
    args.output.write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
