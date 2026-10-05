"""Command line interface: ``ble-cs <command>`` (overview: ``ble-cs --help``).

The commands only parse arguments and delegate to the library modules. Heavy
imports (torch, litert) happen inside the commands so that ``--help`` stays fast.
"""

from pathlib import Path
from typing import Annotated, Optional

import typer

from ble_cs_profiling import config as cfg

app = typer.Typer(
    help="BLE CS profiling: train models, export them for edge targets, generate firmware sources.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
codegen_app = typer.Typer(help="Generate C/C++ sources for the firmware.", no_args_is_help=True)
app.add_typer(codegen_app, name="codegen")

ExistingFile = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, readable=True, resolve_path=True)
]


@app.command("models")
def list_models() -> None:
    """List all registered model names with their default arguments."""
    from ble_cs_profiling.models import MODELS

    for name, preset in MODELS.items():
        print(f"{name:34s} {preset.func.__name__:28s} {preset.keywords}")


@app.command()
def train(
    model: Annotated[str, typer.Option("--model", "-m", help="Registry name, see `ble-cs models`.")] = "mobilenet_v3_small_2d",
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Checkpoint path; default: <checkpoint dir>/<model>.pt")] = None,
) -> None:
    """Train a registered model with the fixed settings from config.py and save a checkpoint."""
    from ble_cs_profiling.train import train as run_training

    run_training(model, output or cfg.CHECKPOINT_DIR / f"{model}.pt")


@app.command()
def export(
    checkpoint: ExistingFile,
    out_dir: Annotated[Optional[Path], typer.Option("--out-dir", "-o", help="Default: <output dir>.")] = None,
    onnx: Annotated[bool, typer.Option(help="FP32 + INT8 ONNX.")] = True,
    tflite: Annotated[bool, typer.Option(help="FP32 + full INT8 TFLite.")] = True,
    onnx_opset: int = 13,
    evaluate: Annotated[bool, typer.Option("--eval/--no-eval", help="MAE of the FP32 and full INT8 TFLite on the eval set.")] = True,
) -> None:
    """Export a checkpoint to ONNX / TFLite (FP32 + full INT8)."""
    from ble_cs_profiling.export import ExportOptions, export_checkpoint

    options = ExportOptions(
        onnx=onnx,
        tflite=tflite,
        onnx_opset=onnx_opset,
        evaluate=evaluate,
    )
    artifacts = export_checkpoint(checkpoint, out_dir or cfg.OUTPUT_DIR, options)
    print("\nArtefakte:")
    for name, path in artifacts.items():
        print(f"  {name:20s} {path}")


@codegen_app.command("model")
def codegen_model(
    tflite: ExistingFile,
    out_dir: Annotated[Optional[Path], typer.Option("--out-dir", "-o", help="E.g. the firmware's src/ML_Models; default: <output dir>/codegen.")] = None,
) -> None:
    """Embed a .tflite model as model_data.h / model_data.cc."""
    from ble_cs_profiling.codegen import write_model_sources

    write_model_sources(tflite, out_dir or cfg.OUTPUT_DIR / "codegen")


@codegen_app.command("samples")
def codegen_samples(
    out_dir: Annotated[Optional[Path], typer.Option("--out-dir", "-o", help="Default: <output dir>/codegen.")] = None,
    num_samples: Annotated[int, typer.Option("--num-samples", "-n")] = 10,
    seed: int = 34,
) -> None:
    """Draw reproducible eval samples as data.hpp (MCU) and .npy (Jetson/PC)."""
    from ble_cs_profiling.codegen import write_test_samples

    write_test_samples(
        cfg.EVAL_DATA_PATH,
        cfg.FEATURES_RAW_IQ_ALL,
        cfg.TARGET,
        out_dir or cfg.OUTPUT_DIR / "codegen",
        num_samples=num_samples,
        seed=seed,
    )
