# Examples for trying packages on top of the pipeline environment with `uv run --with`.
# Runs inside the dev container (VS Code: "Reopen in Container"), see README.md in this folder.
# Copy and change the targets as you like; nothing here is part of the pipeline.
#
# Usage: make -C ml/experiments <target> WITH="seaborn optuna"

# ---- Parameters ----
# WITH:   extra packages, separated by spaces; versions allowed ("numpy<2")
# SCRIPT: Python file for the target script
# PORT:   port of the Jupyter server
WITH   ?= seaborn
SCRIPT ?=
PORT   ?= 8888

# One --with per package. Quoted, so that version specifiers like "numpy<2" reach uv intact
# instead of being read by the shell as a redirection.
UV_RUN := uv run $(foreach package,$(WITH),--with '$(package)')

.PHONY: help python script notebook

help:
	@echo "Pipeline environment + WITH=\"$(WITH)\", nothing gets installed permanently:"
	@echo "  python     interactive Python             make python WITH=\"seaborn\""
	@echo "  script     run a Python file              make script SCRIPT=my_poc.py WITH=\"optuna\""
	@echo "  notebook   Jupyter server for VS Code     make notebook WITH=\"seaborn optuna\""

python:
	$(UV_RUN) python

script:
	$(if $(SCRIPT),,$(error Set SCRIPT=<file.py>))
	$(UV_RUN) python $(SCRIPT)

# A VS Code kernel cannot point to the temporary environment of --with (it is deleted after the
# command). So the server runs inside it: VS Code -> Select Kernel -> Existing Jupyter Server ->
# paste the http://127.0.0.1:... URL from the output. Stop with Ctrl+C.
notebook:
	$(UV_RUN) --with jupyter-server jupyter server --no-browser --ip 127.0.0.1 --port $(PORT)
