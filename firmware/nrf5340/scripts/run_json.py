"""Turn one firmware run into a device-independent ``run.json``.

    python3 scripts/run_json.py <target> <build dir>      (e.g. nrf5340dk build/hil)

Reads ``<build dir>/uart.log`` (the firmware's ``RESULT {...}`` line) and
``<build dir>/zephyr/zephyr.elf`` (flash/RAM usage) and writes ``<build dir>/run.json``:

    {"target": ..., "model_sha256": ..., "cycles": ..., "latency_us": ...,
     "arena_used_bytes": ..., "flash_bytes": ..., "ram_bytes": ..., "prediction": ...}

This format is the only thing the ML side (``edge-ib results record``) knows about a device.
Standard library only: runs in the fw container without extra packages.
"""

import json
import re
import struct
import sys
from pathlib import Path

RESULT_LINE = re.compile(r"RESULT (\{.*\})")  # see src/main.cpp
SRAM = range(0x2000_0000, 0x4000_0000)  # SRAM region of the Cortex-M memory map
PT_LOAD = 1


def parse_uart_log(log_path: Path) -> dict:
    """Return the JSON of the firmware's ``RESULT {...}`` line."""
    for line in log_path.read_text(errors="replace").splitlines():
        if match := RESULT_LINE.search(line):
            return json.loads(match.group(1))
    raise SystemExit(f"No RESULT line in {log_path} (firmware crashed or timed out?)")


def elf_memory(elf_path: Path) -> tuple[int, int]:
    """Flash and RAM usage (bytes) of a Cortex-M ELF32, identical to the linker's memory report.

    Uses the loadable segments (program headers): their load address says what is stored
    in flash, their runtime address what occupies RAM. Initialized data counts for both.
    """
    data = elf_path.read_bytes()
    if data[:6] != b"\x7fELF\x01\x01":
        raise SystemExit(f"Kein ELF32 little-endian: {elf_path}")
    phoff, = struct.unpack_from("<I", data, 0x1C)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x2A)

    flash = ram = 0
    for index in range(phnum):
        p_type, _, vaddr, paddr, filesz, memsz = struct.unpack_from(
            "<6I", data, phoff + index * phentsize
        )
        if p_type != PT_LOAD:
            continue
        if paddr not in SRAM:
            flash += filesz  # code, constants (incl. the model), initial values of variables
        if vaddr in SRAM:
            ram += memsz  # variables incl. zero-initialized ones (e.g. the tensor arena)
    return flash, ram


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    target, build_dir = sys.argv[1], Path(sys.argv[2])

    measured = parse_uart_log(build_dir / "uart.log")
    flash, ram = elf_memory(build_dir / "zephyr" / "zephyr.elf")
    run = {
        "target": target,
        "model_sha256": measured["model_sha256"],
        "cycles": measured["cycles"],
        "latency_us": measured["latency_us"],
        "arena_used_bytes": measured["arena_used_bytes"],
        "flash_bytes": flash,
        "ram_bytes": ram,
        "prediction": measured["prediction"],
    }
    (build_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"{target}: {run['cycles']:,} Zyklen, Arena {run['arena_used_bytes']:,} B, "
          f"Flash {flash / 1024:.1f} KiB, RAM {ram / 1024:.1f} KiB -> {build_dir / 'run.json'}")


if __name__ == "__main__":
    main()
