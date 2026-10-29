"""Results per checkpoint as JSON: ``<output dir>/<checkpoint>/results.json``.

One file collects everything measured for one checkpoint, written step by step:

- ``ble-cs export``            -> ``artifacts``: size, sha256 and MAE of every exported model
- ``ble-cs results record``     -> ``targets``: cycles, latency, arena, flash, RAM on a device
                                   (from a device-independent ``run.json``, read from stdin)

Every target entry stores the sha256 of the model it measured. If a re-export changes
that model, the old measurement no longer belongs to it and is dropped.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RESULTS_FILE = "results.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def save(path: Path, results: dict[str, Any]) -> None:
    path.write_text(json.dumps(results, indent=2) + "\n")


def record_export(
    run_dir: Path,
    checkpoint_name: str,
    checkpoint: dict[str, Any],
    artifacts: list[Path],
    maes: dict[Path, float],
) -> Path:
    """Add exported models to ``results.json``; drop target runs of replaced models."""
    path = run_dir / RESULTS_FILE
    results = load(path)
    results.update({
        "schema_version": SCHEMA_VERSION,
        "checkpoint": checkpoint_name,
        "model_name": checkpoint["model_name"],
        "train_mae_m": round(checkpoint["eval_mae"], 4) if "eval_mae" in checkpoint else None,
    })
    entries = results.setdefault("artifacts", {})
    for artifact in artifacts:
        entry = {"size_bytes": artifact.stat().st_size, "sha256": sha256(artifact)}
        if artifact in maes:
            entry["mae_m"] = round(maes[artifact], 4)
        entries[artifact.relative_to(run_dir).as_posix()] = entry

    current = {entry["sha256"] for entry in entries.values()}
    results["targets"] = {
        name: run for name, run in results.get("targets", {}).items() if run["sha256"] in current
    }
    save(path, results)
    print(f"Ergebnisse gespeichert: {path}")
    return path


def record_run(run: dict[str, Any], output_root: Path) -> list[Path]:
    """Add a device run (``run.json``, see firmware/*/scripts) to every model with its sha256.

    Only the sha256 links a run to a model: the ML side needs no knowledge about the device.
    Byte-identical models (e.g. a copied checkpoint) all get the measurement.
    """
    missing = {"target", "model_sha256", "cycles", "latency_us"} - run.keys()
    if missing:
        raise ValueError(f"run.json unvollständig, es fehlt: {', '.join(sorted(missing))}")

    entry = {key: value for key, value in run.items() if key not in ("target", "model_sha256")}
    entry["measured_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    updated = []
    for path in sorted(output_root.glob(f"**/{RESULTS_FILE}")):
        results = load(path)
        for artifact, info in results.get("artifacts", {}).items():
            if info["sha256"] == run["model_sha256"]:
                results.setdefault("targets", {})[run["target"]] = {
                    "artifact": artifact, "sha256": info["sha256"], **entry,
                }
                save(path, results)
                updated.append(path)
    if not updated:
        raise ValueError(
            f"Kein exportiertes Modell mit sha256 {run['model_sha256'][:16]}… in {output_root}: "
            "Firmware mit altem Modell gebaut oder Export gelöscht?"
        )
    for path in updated:
        print(f"{run['target']}: {run['cycles']:,} Zyklen -> {path}")
    return updated


def markdown_table(output_root: Path, target: str) -> str:
    """One row per checkpoint: model size/MAE from the export, metrics from ``target``."""
    def kib(value: int | None) -> str:
        return f"{value / 1024:.1f}" if value is not None else "–"

    def mae(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "–"

    rows = [
        f"| Checkpoint | FP32 MAE [m] | INT8 MAE [m] | INT8 size [KiB] "
        f"| Cycles ({target}) | Latency [µs] | Arena [KiB] | Flash [KiB] | RAM [KiB] |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    all_results = [load(path) for path in output_root.glob(f"**/{RESULTS_FILE}")]
    for results in sorted(all_results, key=lambda r: r["checkpoint"]):
        artifacts = results.get("artifacts", {})
        fp32 = artifacts.get("fp32_full/model.tflite", {})
        int8 = artifacts.get("int8_full/model.tflite", {})
        run = results.get("targets", {}).get(target, {})
        cycles = f"{run['cycles']:,}" if run else "–"
        latency = f"{run['latency_us']:,}" if run else "–"
        rows.append(
            f"| `{results['checkpoint']}` | {mae(fp32.get('mae_m'))} | {mae(int8.get('mae_m'))} "
            f"| {kib(int8.get('size_bytes'))} | {cycles} | {latency} "
            f"| {kib(run.get('arena_used_bytes'))} | {kib(run.get('flash_bytes'))} "
            f"| {kib(run.get('ram_bytes'))} |"
        )
    return "\n".join(rows)
