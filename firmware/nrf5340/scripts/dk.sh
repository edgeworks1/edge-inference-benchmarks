#!/usr/bin/env bash
# Host-side helpers for the nRF5340 DK: flash, measure and read the console over the J-Link.
# Runs natively in WSL/Linux because it needs USB access, not in Docker. Called by the Makefile.
#
#   dk.sh flash <hex>              program the app core and reset it
#   dk.sh measure <hex> <log>      capture the UART, flash, stop at the firmware's RESULT line
#   dk.sh console                  print the app core UART until Ctrl+C
#
# Board selection: the first connected J-Link, or the one whose serial number ends in $SNR.
set -euo pipefail
shopt -s nullglob

readonly SERIAL_DIR=/dev/serial/by-id
readonly BAUD_RATE=115200
readonly MEASURE_TIMEOUT_S=30

usage() {
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

fail() {
    echo "dk.sh: $*" >&2
    exit 1
}

# App core console = J-Link interface 02 (vcom 1); interface 00 belongs to the network core.
# by-id paths stay the same no matter in which order the boards were detected.
find_uart() {
    local matches=("$SERIAL_DIR"/usb-SEGGER_J-Link_*"${SNR:-}"-if02)
    ((${#matches[@]} > 0)) || fail "no J-Link${SNR:+ with serial $SNR} in $SERIAL_DIR (attach the DK to WSL with usbipd)"
    echo "${matches[0]}"
}

# usb-SEGGER_J-Link_001050025601-if02 -> 1050025601 (nrfutil shows it without leading zeros)
serial_number() {
    local name
    name=$(basename "$1")
    name=${name#usb-SEGGER_J-Link_}
    name=${name%-if02}
    echo "$((10#$name))"
}

configure_uart() {
    stty -F "$1" "$BAUD_RATE" raw -echo
}

# Erases only the flash pages the image occupies, so the network core image stays intact.
flash() {
    local hex=$1 uart serial
    [[ -f $hex ]] || fail "firmware not found: $hex (build it first)"
    uart=$(find_uart)
    serial=$(serial_number "$uart")
    nrfutil device program --serial-number "$serial" --firmware "$hex" \
        --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE
    nrfutil device reset --serial-number "$serial"
}

# The capture starts before flashing, otherwise the beginning of the output would be missed.
# The timeout turns a hanging or crashed firmware into an error instead of waiting forever.
measure() {
    local hex=$1 log=$2 uart capture_pid
    uart=$(find_uart)
    configure_uart "$uart"
    timeout "$MEASURE_TIMEOUT_S" sed '/RESULT /q' "$uart" > "$log" &
    capture_pid=$!
    # Never leave the capture running. Double quotes: the PID is inserted now, because the
    # local variable no longer exists when the trap runs at script exit.
    # shellcheck disable=SC2064
    trap "kill $capture_pid 2> /dev/null || true" EXIT

    flash "$hex"
    wait "$capture_pid" || fail "no RESULT line within ${MEASURE_TIMEOUT_S} s, see $log"
}

console() {
    local uart
    uart=$(find_uart)
    configure_uart "$uart"
    echo "Reading $uart (Ctrl+C to stop)" >&2
    exec cat "$uart"
}

case "${1:-}" in
    flash) (($# == 2)) || usage; flash "$2" ;;
    measure) (($# == 3)) || usage; measure "$2" "$3" ;;
    console) (($# == 1)) || usage; console ;;
    *) usage ;;
esac
