# Experiments

Notebooks and proofs of concept. What works moves into `ml/src/` and becomes a pipeline step.
Runs in the dev container, where the environment is the pipeline's (same `uv.lock`) plus the dev
tools; the package is installed as editable (`from edge_ib.data import load_training_set`).

## Trying packages: `uv run --with`

The environment is read-only. `uv run --with` adds a package on top of it for one command
(cached, `pyproject.toml` and `uv.lock` stay unchanged). Run it from inside `ml/`:

```bash
uv run --with seaborn python my_poc.py
uv run --with seaborn --with optuna python
make -C ml/experiments notebook WITH="seaborn optuna"   # Jupyter server for VS Code
```

`--with` can replace pipeline versions (`--with "numpy<2"`), so results may no longer reflect the
pipeline. Once a package proves useful: `uv add --no-sync <package>` (or `--dev`), commit
`pyproject.toml` and `uv.lock`, rebuild the image or dev container.
