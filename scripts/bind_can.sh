#!/usr/bin/env bash
set -euo pipefail

LEFT_CAN="${LEFT_CAN:-can3}"
RIGHT_CAN="${RIGHT_CAN:-can2}"
BITRATE="${BITRATE:-1000000}"
LEFT_SERIAL="${LEFT_SERIAL:-002F00354148571320343133}"
RIGHT_SERIAL="${RIGHT_SERIAL:-004500434148571320343133}"

for interface in "$LEFT_CAN" "$RIGHT_CAN"; do
    ip link show "$interface" >/dev/null || { printf 'CAN interface not found: %s\n' "$interface" >&2; exit 1; }
done

actual_left="$(udevadm info -q property -p "/sys/class/net/$LEFT_CAN" | awk -F= '$1 == "ID_SERIAL_SHORT" {print $2}')"
actual_right="$(udevadm info -q property -p "/sys/class/net/$RIGHT_CAN" | awk -F= '$1 == "ID_SERIAL_SHORT" {print $2}')"
if [[ "$actual_left" != "$LEFT_SERIAL" || "$actual_right" != "$RIGHT_SERIAL" ]]; then
    printf 'CAN binding mismatch:\n  %s=%s (expected %s)\n  %s=%s (expected %s)\n' \
        "$LEFT_CAN" "$actual_left" "$LEFT_SERIAL" "$RIGHT_CAN" "$actual_right" "$RIGHT_SERIAL" >&2
    exit 1
fi

for interface in "$LEFT_CAN" "$RIGHT_CAN"; do
    details="$(ip -details link show "$interface")"
    if [[ "$details" == *"state UP"* && "$details" == *"bitrate $BITRATE"* ]]; then
        printf '%s is already UP at %s bit/s; leaving the active SocketCAN interface unchanged\n' \
            "$interface" "$BITRATE"
        continue
    fi
    sudo ip link set "$interface" down || true
    sudo ip link set "$interface" type can bitrate "$BITRATE"
    sudo ip link set "$interface" up
    sudo ip link set "$interface" txqueuelen 1000
done

printf 'CAN binding ready: left=%s right=%s bitrate=%s\n' "$LEFT_CAN" "$RIGHT_CAN" "$BITRATE"
ip -details link show "$LEFT_CAN"
ip -details link show "$RIGHT_CAN"
