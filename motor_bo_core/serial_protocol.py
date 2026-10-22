"""Simple CRLF-delimited commands and generic target/feedback telemetry."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


MAX_LINE_LENGTH = 128
RUN_TARGET_MAX = 50.0
RUN_DURATION_MAX_MS = 3000


@dataclass(frozen=True)
class ProtocolMessage:
    kind: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class Telemetry:
    elapsed_ms: int
    target: float
    feedback: float


def encode_frame(tokens: Iterable[str]) -> bytes:
    """Join fields with commas and terminate the line with CRLF."""

    fields = tuple(str(token) for token in tokens)
    if not fields or any(char in field for field in fields for char in ",\r\n"):
        raise ValueError("protocol fields may not contain commas or line breaks")
    line = ",".join(fields)
    encoded = (line + "\r\n").encode("ascii")
    if len(encoded) > MAX_LINE_LENGTH:
        raise ValueError("protocol line is too long")
    return encoded


def _number_text(value: float, name: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return format(number, ".6g")


def encode_ping() -> bytes:
    return b"PING\r\n"


def encode_set_pid(
    kp: float,
    ki: float,
    kd: float,
    *,
    kp_range: tuple[float, float] | None = None,
    ki_range: tuple[float, float] | None = None,
    kd_range: tuple[float, float] | None = None,
    supports_kd: bool = False,
) -> bytes:
    kp_text = _number_text(kp, "Kp")
    ki_text = _number_text(ki, "Ki")
    kd_text = _number_text(kd, "Kd")
    kp_value = float(kp)
    ki_value = float(ki)
    kd_value = float(kd)

    if kp_value < 0.0 or ki_value < 0.0 or kd_value < 0.0:
        raise ValueError("Kp, Ki and Kd must be non-negative")

    for name, value, limits in (("Kp", kp_value, kp_range), ("Ki", ki_value, ki_range),
                                ("Kd", kd_value, kd_range)):
        if limits is None:
            continue
        try:
            lower, upper = (float(limit) for limit in limits)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} range must contain two finite limits") from exc
        if not math.isfinite(lower) or not math.isfinite(upper) or lower < 0.0 or lower > upper:
            raise ValueError(f"{name} range must be finite, non-negative, and increasing")
        if not lower <= value <= upper:
            raise ValueError(f"{name} must be in configured range {lower}..{upper}")
    # Kd uses the same numeric/boundary checks as Kp/Ki. Firmware decides
    # whether its chosen controller accepts D; supports_kd is legacy-only.

    return encode_frame(("SET_PID", kp_text, ki_text, kd_text))


def validate_target_range(target: float, lower: float, upper: float,
                          lower_inclusive: bool = False, upper_inclusive: bool = True) -> None:
    """Validate numeric endpoints without assuming a controller type or unit."""
    if (not isinstance(lower_inclusive, bool) or not isinstance(upper_inclusive, bool)
            or not all(math.isfinite(v) for v in (lower, upper)) or lower >= upper):
        raise ValueError("目标上下界必须为有限值且下限小于上限，边界开关必须为布尔值")
    if (not math.isfinite(target) or
            not (target >= lower if lower_inclusive else target > lower) or
            not (target <= upper if upper_inclusive else target < upper)):
        left, right = ("[" if lower_inclusive else "("), ("]" if upper_inclusive else ")")
        raise ValueError(f"target must be in range {left}{lower:g}, {upper:g}{right}")


def encode_run_step(
    target: float,
    duration_ms: int,
    *,
    target_max_rad_s: float = RUN_TARGET_MAX,
    duration_max_ms: int = RUN_DURATION_MAX_MS,
    target_min: float = 0.0,
    target_max: float | None = None,
    target_min_inclusive: bool = False,
    target_max_inclusive: bool = True,
) -> bytes:
    """Encode a generic step; legacy callers retain their positive-speed bounds."""
    target_text = _number_text(target, "target")
    target_value = float(target)
    target_limit = float(target_max_rad_s if target_max is None else target_max)
    validate_target_range(target_value, float(target_min), target_limit,
                          target_min_inclusive, target_max_inclusive)
    # Six-significant-digit wire formatting must not round a legal value onto
    # an excluded endpoint (e.g. 359.99999 becoming 360).
    validate_target_range(float(target_text), float(target_min), target_limit,
                          target_min_inclusive, target_max_inclusive)

    if isinstance(duration_ms, bool):
        raise ValueError("duration_ms must be an integer")
    try:
        duration = int(duration_ms)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("duration_ms must be an integer") from exc
    if duration != duration_ms:
        raise ValueError("duration_ms must be an integer")
    if isinstance(duration_max_ms, bool):
        raise ValueError("duration_max_ms must be a positive integer")
    try:
        duration_limit = int(duration_max_ms)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("duration_max_ms must be a positive integer") from exc
    if duration_limit != duration_max_ms or duration_limit < 1:
        raise ValueError("duration_max_ms must be a positive integer")
    if not 1 <= duration <= duration_limit:
        raise ValueError(f"duration_ms must be in range 1..{duration_limit}")

    return encode_frame(("RUN_STEP", target_text, str(duration)))


def encode_stop() -> bytes:
    return b"STOP\r\n"


def parse_frame(line: bytes | str) -> ProtocolMessage:
    """Parse one firmware response: OK or generic DATA telemetry."""

    if isinstance(line, bytes):
        text = line.decode("ascii")
    else:
        text = line
    if not text.endswith("\r\n") or len(text.encode("ascii")) > MAX_LINE_LENGTH:
        raise ValueError("line must end with CRLF and fit the protocol limit")
    text = text[:-2]
    if not text or "\r" in text or "\n" in text:
        raise ValueError("empty or oversized line")
    if text == "OK":
        return ProtocolMessage("ACK", ("OK",))
    parts = tuple(text.split(","))
    if parts[0] == "DATA" and len(parts) == 4:
        message = ProtocolMessage("DATA", parts[1:])
        parse_telemetry(message)
        return message
    raise ValueError("unknown firmware response or telemetry line")


def parse_telemetry(message: ProtocolMessage) -> Telemetry:
    if message.kind != "DATA" or len(message.fields) != 3:
        raise ValueError("expected DATA,elapsed,target,feedback")
    try:
        elapsed_ms = int(message.fields[0], 10)
        target, feedback = (float(value) for value in message.fields[1:])
    except ValueError as exc:
        raise ValueError("DATA contains an invalid value") from exc
    if elapsed_ms < 0 or not math.isfinite(target) or not math.isfinite(feedback):
        raise ValueError("DATA values must be finite and elapsed_ms non-negative")
    return Telemetry(elapsed_ms, target, feedback)
