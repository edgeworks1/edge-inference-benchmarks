"""Command line interface: ``edge-ib <command>`` (overview: ``edge-ib --help``).

The commands only parse arguments and delegate to the library modules. Heavy imports
(torch, LiteRT, ONNX Runtime) happen inside the commands so that ``--help`` stays fast.
"""

from pathlib import Path
from typing import Annotated

import typer

from edge_ib import config as cfg

app = typer.Typer(
    help="Edge inference benchmarks: train, export for edge targets, generate firmware sources.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
codegen_app = typer.Typer(help="Generate C/C++ sources for the firmware.", no_args_is_help=True)
app.add_typer(codegen_app, name="codegen")
results_app = typer.Typer(
    help="Collect measurements in <output dir>/<checkpoint>/results.json.", no_args_is_help=True
)
app.add_typer(results_app, name="results")

ExistingFile = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, readable=True, resolve_path=True)
]


@app.command("models")
def list_models() -> None:
    """List all registered model names with their default arguments."""
    from edge_ib.models import MODELS

    for name, preset in MODELS.items():
        print(f"{name:34s} {preset.func.__name__:28s} {preset.keywords}")


@app.command()
def train(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Registry name, see `edge-ib models`.")
    ] = "mobilenet_v3_small_2d",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Checkpoint path; default: <checkpoint dir>/<model>.pt"
        ),
    ] = None,
) -> None:
    """Train a registered model with the fixed settings from config.py and save a checkpoint."""
    from edge_ib.train import train as run_training

    run_training(model, output or cfg.CHECKPOINT_DIR / f"{model}.pt")


@app.command()
def export(
    checkpoint: ExistingFile,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            "-o",
            help="Root folder; files go to <out dir>/<checkpoint name>/. Default: <output dir>.",
        ),
    ] = None,
    onnx: Annotated[bool, typer.Option(help="FP32 + INT8 ONNX.")] = True,
    tflite: Annotated[bool, typer.Option(help="FP32 + full INT8 TFLite.")] = True,
    onnx_opset: int = 14,
    evaluate: Annotated[
        bool,
        typer.Option(
            "--eval/--no-eval", help="MAE of the FP32 and full INT8 TFLite on the eval set."
        ),
    ] = True,
) -> None:
    """Export a checkpoint to ONNX and TFLite (FP32 + INT8) and record sizes and MAE."""
    from edge_ib.export import ExportOptions, export_checkpoint

    options = ExportOptions(
        onnx=onnx,
        tflite=tflite,
        onnx_opset=onnx_opset,
        evaluate=evaluate,
    )
    export_checkpoint(checkpoint, out_dir or cfg.OUTPUT_DIR, options)


@codegen_app.command("model")
def codegen_model(
    tflite: ExistingFile,
    out_dir: Annotated[
        Path | None, typer.Option("--out-dir", "-o", help="Default: <tflite folder>/codegen.")
    ] = None,
) -> None:
    """Embed a .tflite model (+ its sha256) as model_data.h / model_data.cc."""
    from edge_ib.codegen import write_model_sources

    write_model_sources(tflite, out_dir or tflite.parent / "codegen")


@codegen_app.command("samples")
def codegen_samples(
    out_dir: Annotated[
        Path | None, typer.Option("--out-dir", "-o", help="Default: <output dir>/samples.")
    ] = None,
    num_samples: Annotated[int, typer.Option("--num-samples", "-n")] = 10,
    seed: int = cfg.SEED,
) -> None:
    """Draw reproducible eval samples as data.hpp (MCU) and .npy (Jetson/PC)."""
    from edge_ib.codegen import write_test_samples
    from edge_ib.data import draw_eval_samples

    features, targets = draw_eval_samples(num_samples, seed)
    write_test_samples(features, targets, out_dir or cfg.OUTPUT_DIR / "samples")


@results_app.command("record")
def results_record() -> None:
    """Read a device run (run.json) from stdin and add it to results.json of the model it ran.

    Example: cat build/hil/run.json | docker compose run --rm -T ml results record
    """
    import json
    import sys

    from edge_ib.results import record_run

    record_run(json.load(sys.stdin), cfg.OUTPUT_DIR)


@results_app.command("table")
def results_table(
    target: Annotated[
        str, typer.Option(help="Device whose measurements fill the table.")
    ] = "nrf5340dk",
) -> None:
    """Print all results.json as one Markdown table (e.g. for the README)."""
    from edge_ib.results import markdown_table

    print(markdown_table(cfg.OUTPUT_DIR, target))
