"""Deterministic speed-loop experiment; none of these constants describe real hardware."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Gains:
    kp: float = 0.05
    ki: float = 0.10
    kd: float = 0.0


@dataclass(frozen=True)
class SimulationConfig:
    duration_s: float = 2.0
    step_rad_s: float = 10.0
    integration_dt_s: float = 0.00025
    speed_tick_hz: float = 930.0
    inertia_kg_m2: float = 0.01
    damping_nm_per_rad_s: float = 0.01
    torque_constant_nm_per_a: float = 0.10
    current_tau_s: float = 0.002
    iq_limit_a: float = 1.55
    speed_abort_rad_s: float = 15.0
    derivative_filter_tau_s: float = 0.02


def run_trial(gains: Gains, config: SimulationConfig) -> dict:
    """Reset all state, apply one speed step, and return a scored full trace.

    PI gains use A/(rad/s) and A/rad. Kd uses A/(rad/s²). D acts on
    measured speed, not the setpoint, and its derivative is low-pass filtered.
    """
    if not all(math.isfinite(value) and value >= 0.0 for value in asdict(gains).values()):
        raise ValueError("PID gains must be finite and non-negative")
    if not all(math.isfinite(value) for value in asdict(config).values()):
        raise ValueError("Simulation parameters must be finite")
    if (
        config.integration_dt_s <= 0 or config.speed_tick_hz <= 0 or config.duration_s <= 0
        or config.inertia_kg_m2 <= 0 or config.current_tau_s <= 0
        or config.integration_dt_s > config.current_tau_s
        or config.iq_limit_a <= 0 or config.step_rad_s <= 0
    ):
        raise ValueError("Simulation periods and physical parameters must be positive")

    speed = current = iq_ref = integral = derivative = last_speed = 0.0
    speed_period = 1.0 / config.speed_tick_hz
    schedule_elapsed = speed_period
    since_update = speed_period
    control_updates = 0
    trace: list[dict[str, float]] = []
    fault = ""
    steps = round(config.duration_s / config.integration_dt_s)

    for step in range(steps + 1):
        t = step * config.integration_dt_s
        if schedule_elapsed >= speed_period:
            dt_control = since_update
            error = config.step_rad_s - speed
            raw_derivative = (speed - last_speed) / dt_control
            alpha = dt_control / (config.derivative_filter_tau_s + dt_control)
            derivative += alpha * (raw_derivative - derivative)
            next_integral = integral + gains.ki * error * dt_control
            command = gains.kp * error + next_integral - gains.kd * derivative
            if (command > config.iq_limit_a and error > 0) or (
                command < -config.iq_limit_a and error < 0
            ):
                command = gains.kp * error + integral - gains.kd * derivative
            else:
                integral = next_integral
            iq_ref = max(-config.iq_limit_a, min(config.iq_limit_a, command))
            last_speed = speed
            schedule_elapsed -= speed_period  # Keep fractional time to preserve 930 Hz on average.
            since_update = 0.0
            control_updates += 1

        trace.append({
            "time_s": t,
            "speed_ref_rad_s": config.step_rad_s,
            "speed_rad_s": speed,
            "iq_ref_a": iq_ref,
            "iq_a": current,
        })
        if not all(math.isfinite(v) for v in (speed, current, iq_ref)):
            fault = "non_finite"
            break
        if abs(speed) > config.speed_abort_rad_s:
            fault = "simulated_overspeed"
            break
        if step == steps:
            break

        current += (iq_ref - current) * config.integration_dt_s / config.current_tau_s
        acceleration = (
            config.torque_constant_nm_per_a * current
            - config.damping_nm_per_rad_s * speed
        ) / config.inertia_kg_m2
        speed += acceleration * config.integration_dt_s
        schedule_elapsed += config.integration_dt_s
        since_update += config.integration_dt_s

    metrics = score_trace(trace, config, fault)
    metrics["speed_pi_updates"] = control_updates
    return {"trace": trace, "metrics": metrics}


def score_trace(trace: list[dict[str, float]], config: SimulationConfig, fault: str = "") -> dict:
    """Fixed, dimensionless objective; lower is better, failures score 10."""
    if not trace or fault:
        return {"valid": False, "fault": fault or "empty_trace", "score": 10.0}
    target = config.step_rad_s
    iae = sum(abs(target - row["speed_rad_s"]) for row in trace)
    iae_norm = iae * config.integration_dt_s / (target * config.duration_s)
    peak_speed = max(row["speed_rad_s"] for row in trace)
    overshoot = max(0.0, (peak_speed - target) / target)
    tolerance = target * 0.05
    last_outside = max(
        (i for i, row in enumerate(trace) if abs(target - row["speed_rad_s"]) > tolerance),
        default=-1,
    )
    settling_s = (
        trace[last_outside + 1]["time_s"]
        if last_outside + 1 < len(trace)
        else config.duration_s
    )
    settling_fraction = settling_s / config.duration_s
    saturated_fraction = sum(
        abs(row["iq_ref_a"]) >= 0.99 * config.iq_limit_a for row in trace
    ) / len(trace)
    score = (
        0.55 * iae_norm + 0.20 * overshoot + 0.15 * settling_fraction
        + 0.10 * saturated_fraction
    )
    if not math.isfinite(score):
        return {"valid": False, "fault": "invalid_score", "score": 10.0}
    return {
        "valid": True,
        "fault": "",
        "score": score,
        "iae_norm": iae_norm,
        "overshoot_ratio": overshoot,
        "settling_time_s": settling_s,
        "iq_saturation_fraction": saturated_fraction,
        "peak_speed_rad_s": peak_speed,
        "peak_iq_a": max(abs(row["iq_a"]) for row in trace),
    }
