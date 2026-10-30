"""Results per checkpoint as JSON: ``<output root>/<checkpoint>/results.json``.

One file collects everything measured for one checkpoint, written step by step:

- ``ble-cs export``          -> ``artifacts``: size, sha256 and MAE of every exported model file
- ``ble-cs results record``  -> ``targets``: cycles, latency, arena, flash and RAM on a device,
                                taken from a device-independent ``run.json`` (read from stdin)

The sha256 is the link between both: a device run belongs to the model file with the same
hash. When a re-export changes a model, its old device measurements are dropped.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RESULTS_FILE = "results.json"
REQUIRED_RUN_KEYS = {"target", "model_sha256", "cycles", "latency_us"}
MISSING = "–"  # placeholder in the Markdown table


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    """Read a results file, or return an empty dict if it does not exist yet."""
    return json.loads(path.read_text()) if path.is_file() else {}


def save(path: Path, results: dict[str, Any]) -> None:
    """Write a results file as indented JSON."""
    path.write_text(json.dumps(results, indent=2) + "\n")


# ============================================================
# Writing
# ============================================================
def record_export(
    run_dir: Path,
    checkpoint_name: str,
    checkpoint: dict[str, Any],
    artifacts: list[Path],
    maes: dict[Path, float],
) -> Path:
    """Add the exported model files to ``results.json`` and drop measurements of replaced models."""
    path = run_dir / RESULTS_FILE
    results = load(path)
    results.update(
        {
            "schema_version": SCHEMA_VERSION,
            "checkpoint": checkpoint_name,
            "model_name": checkpoint["model_name"],
            "train_mae_m": round(checkpoint["eval_mae"], 4) if "eval_mae" in checkpoint else None,
        }
    )

    entries = results.setdefault("artifacts", {})
    for artifact in artifacts:
        key = artifact.relative_to(run_dir).as_posix()
        entries[key] = _artifact_entry(artifact, maes.get(artifact), previous=entries.get(key, {}))

    _drop_stale_targets(results)
    save(path, results)
    print(f"Saved results: {path}")
    return path


def record_run(run: dict[str, Any], output_root: Path) -> list[Path]:
    """Add a device run (``run.json``) to every exported model with the same sha256.

    Byte-identical models (e.g. a copied checkpoint) all receive the measurement.
    """
    missing = REQUIRED_RUN_KEYS - run.keys()
    if missing:
        raise ValueError(f"Incomplete run.json, missing: {', '.join(sorted(missing))}")

    measurement = {
        key: value for key, value in run.items() if key not in ("target", "model_sha256")
    }
    measurement["measured_at"] = datetime.now(UTC).isoformat(timespec="seconds")

    updated = []
    for path in sorted(output_root.glob(f"**/{RESULTS_FILE}")):
        results = load(path)
        for artifact, info in results.get("artifacts", {}).items():
            if info["sha256"] == run["model_sha256"]:
                results.setdefault("targets", {})[run["target"]] = {
                    "artifact": artifact,
                    "sha256": info["sha256"],
                    **measurement,
                }
                save(path, results)
                updated.append(path)

    if not updated:
        raise ValueError(
            f"No exported model with sha256 {run['model_sha256'][:16]}... in {output_root}. "
            "Was the firmware built with an old model, or was the export deleted?"
        )
    for path in updated:
        print(f"{run['target']}: {run['cycles']:,} cycles -> {path}")
    return updated


def _artifact_entry(artifact: Path, mae: float | None, previous: dict[str, Any]) -> dict[str, Any]:
    """Describe one model file; keep its old MAE if it was not re-evaluated and is unchanged."""
    entry: dict[str, Any] = {"size_bytes": artifact.stat().st_size, "sha256": sha256(artifact)}
    if mae is not None:
        entry["mae_m"] = round(mae, 4)
    elif previous.get("sha256") == entry["sha256"] and "mae_m" in previous:
        entry["mae_m"] = previous["mae_m"]  # e.g. a re-export with --no-eval
    return entry


def _drop_stale_targets(results: dict[str, Any]) -> None:
    """Remove device measurements whose model file no longer exists in this export."""
    current_hashes = {entry["sha256"] for entry in results["artifacts"].values()}
    results["targets"] = {
        target: run
        for target, run in results.get("targets", {}).items()
        if run["sha256"] in current_hashes
    }


# ============================================================
# Reporting
# ============================================================
def markdown_table(output_root: Path, target: str) -> str:
    """Summarize all results as a Markdown table: one row per checkpoint, metrics of ``target``."""
    header = [
        f"| Checkpoint | FP32 MAE [m] | INT8 MAE [m] | INT8 size [KiB] "
        f"| Cycles ({target}) | Latency [µs] | Arena [KiB] | Flash [KiB] | RAM [KiB] |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    all_results = [load(path) for path in output_root.glob(f"**/{RESULTS_FILE}")]
    rows = [
        _table_row(results, target)
        for results in sorted(all_results, key=lambda r: r["checkpoint"])
    ]
    return "\n".join(header + rows)


def _table_row(results: dict[str, Any], target: str) -> str:
    """Format one checkpoint's results as a Markdown table row."""
    artifacts = results.get("artifacts", {})
    fp32 = artifacts.get("fp32_full/model.tflite", {})
    int8 = artifacts.get("int8_full/model.tflite", {})
    run = results.get("targets", {}).get(target, {})
    cells = [
        f"`{results['checkpoint']}`",
        _format_mae(fp32.get("mae_m")),
        _format_mae(int8.get("mae_m")),
        _format_kib(int8.get("size_bytes")),
        _format_count(run.get("cycles")),
        _format_count(run.get("latency_us")),
        _format_kib(run.get("arena_used_bytes")),
        _format_kib(run.get("flash_bytes")),
        _format_kib(run.get("ram_bytes")),
    ]
    return "| " + " | ".join(cells) + " |"


def _format_kib(value: int | None) -> str:
    """Format a byte count as KiB with one decimal."""
    return f"{value / 1024:.1f}" if value is not None else MISSING


def _format_mae(value: float | None) -> str:
    """Format an MAE in metres with three decimals."""
    return f"{value:.3f}" if value is not None else MISSING


def _format_count(value: int | None) -> str:
    """Format an integer with thousands separators."""
    return f"{value:,}" if value is not None else MISSING
