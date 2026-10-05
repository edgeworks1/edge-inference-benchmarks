"""Allows ``python -m ble_cs_profiling`` as an alternative to the ``ble-cs`` command."""

from ble_cs_profiling.cli import app

app(prog_name="ble-cs")
