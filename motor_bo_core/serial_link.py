"""PC-side serial transport for the STM32 motor-tuning protocol."""

from __future__ import annotations

from collections import deque
import math
import time
from typing import Any

from .serial_protocol import (
    RUN_DURATION_MAX_MS,
    RUN_TARGET_MAX,
    ProtocolMessage,
    encode_ping,
    encode_run_step,
    encode_set_pid,
    encode_stop,
    parse_frame,
)


def parse_serial_config(config: dict) -> dict:
    """Validate the serial settings shared by the PING and tuning entry points."""
    settings = config.get("serial", {})
    if not isinstance(settings, dict):
        raise ValueError("serial must be a JSON object")
    port = str(settings.get("port", "")).strip()
    baudrate = settings.get("baudrate", 115200)
    read_timeout_s = settings.get("read_timeout_s", 0.05)
    ping_timeout_s = settings.get("ping_timeout_s", 1.0)
    listen_after_ping_s = settings.get("listen_after_ping_s", 0.0)
    if not port:
        raise ValueError("请在 motor_bo_hw_config.json 的 serial.port 填写 COM 口")
    if isinstance(baudrate, bool) or not isinstance(baudrate, int) or baudrate <= 0:
        raise ValueError("serial.baudrate must be a positive integer")
    for name, value in (
        ("read_timeout_s", read_timeout_s),
        ("ping_timeout_s", ping_timeout_s),
        ("listen_after_ping_s", listen_after_ping_s),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"serial.{name} must be a finite number")
    if read_timeout_s <= 0 or ping_timeout_s <= 0 or listen_after_ping_s < 0:
        raise ValueError("serial timeouts must be positive; listen_after_ping_s may be zero")
    return {
        "port": port,
        "baudrate": baudrate,
        "read_timeout_s": float(read_timeout_s),
        "ping_timeout_s": float(ping_timeout_s),
        "listen_after_ping_s": float(listen_after_ping_s),
    }


class MotorTuneLink:
    """Small request/response client; trial scoring is owned by the runner."""

    def __init__(self, serial_port: Any):
        self._serial = serial_port
        self._pending: deque[ProtocolMessage] = deque(maxlen=512)

    @classmethod
    def open(cls, port: str, baudrate: int = 115200, timeout_s: float = 0.05) -> "MotorTuneLink":
        if not port or not port.strip():
            raise ValueError("serial port is required, for example COM5")
        if baudrate <= 0 or timeout_s <= 0:
            raise ValueError("baudrate and timeout_s must be positive")
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required; install requirements-motor-bo.txt") from exc
        port_handle = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout_s,
            write_timeout=1.0,
        )
        return cls(port_handle)

    def close(self) -> None:
        if getattr(self._serial, "is_open", True):
            self._serial.close()

    def clear_pending_messages(self) -> None:
        """Discard stale telemetry between completed trials, never during a trial."""
        self._pending.clear()
        reset_input = getattr(self._serial, "reset_input_buffer", None)
        if callable(reset_input):
            reset_input()

    def _read_from_port(self, timeout_s: float) -> ProtocolMessage:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            line = self._serial.readline()
            if not line:
                continue
            try:
                return parse_frame(line)
            except ValueError:
                continue
        raise TimeoutError("timed out waiting for a valid STM32 frame")

    def read_message(self, timeout_s: float = 1.0) -> ProtocolMessage:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self._pending:
            return self._pending.popleft()
        return self._read_from_port(timeout_s)

    def _request_ack(
        self,
        frame: bytes,
        timeout_s: float,
    ) -> ProtocolMessage:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._serial.write(frame)
        flush = getattr(self._serial, "flush", None)
        if callable(flush):
            flush()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                message = self._read_from_port(max(0.01, deadline - time.monotonic()))
            except TimeoutError:
                break
            if message.kind == "ACK":
                return message
            self._pending.append(message)
        raise TimeoutError(
            "no OK received before timeout; the firmware may have silently ignored "
            "the command (state/value not accepted), or the serial link is unavailable"
        )

    def ping(self, timeout_s: float = 1.0) -> ProtocolMessage:
        return self._request_ack(encode_ping(), timeout_s)

    def set_pid(
        self,
        kp: float,
        ki: float,
        kd: float,
        timeout_s: float = 1.0,
        *,
        kp_range: tuple[float, float] | None = None,
        ki_range: tuple[float, float] | None = None,
        kd_range: tuple[float, float] | None = None,
        supports_kd: bool = False,
    ) -> ProtocolMessage:
        return self._request_ack(
            encode_set_pid(kp, ki, kd, kp_range=kp_range, ki_range=ki_range,
                           kd_range=kd_range, supports_kd=supports_kd),
            timeout_s,
        )

    def run_step(
        self,
        target: float,
        duration_ms: int,
        timeout_s: float = 1.0,
        *,
        target_max_rad_s: float = RUN_TARGET_MAX,
        duration_max_ms: int = RUN_DURATION_MAX_MS,
        target_min: float = 0.0,
        target_max: float | None = None,
        target_min_inclusive: bool = False,
        target_max_inclusive: bool = True,
    ) -> ProtocolMessage:
        return self._request_ack(
            encode_run_step(
                target,
                duration_ms,
                target_max_rad_s=target_max_rad_s,
                duration_max_ms=duration_max_ms,
                target_min=target_min, target_max=target_max,
                target_min_inclusive=target_min_inclusive,
                target_max_inclusive=target_max_inclusive,
            ),
            timeout_s,
        )

    def stop(self, timeout_s: float = 1.0) -> ProtocolMessage:
        return self._request_ack(encode_stop(), timeout_s)
