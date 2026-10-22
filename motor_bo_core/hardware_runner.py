"""Bounded Bayesian tuning of unit-neutral, firmware-defined step trials."""

from __future__ import annotations

import csv
from datetime import datetime
import json
import math
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any

from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement

from .advisor import (
    ACTION_ZH,
    EXPLORATION_ZH,
    HardwareLiveAdvisor,
    XI_BY_MODE,
    validate_advice,
)
from .serial_protocol import (
    RUN_DURATION_MAX_MS,
    RUN_TARGET_MAX,
    encode_run_step,
    parse_telemetry,
    validate_target_range,
)
from .gui_support import expand_curves


DEFAULT_TUNE_CONFIG = Path(__file__).resolve().parent.parent / "motor_bo_hw_config.json"


class HardwareStopRequested(KeyboardInterrupt):
    """Cooperative desktop stop; preserve partial telemetry and execute STOP."""


def _pid_mode(config: dict) -> str:
    """Canonical search mode; retain compatibility with old snapshots/profiles."""
    return config.get("pid_mode", "pid" if config.get("tune_d", False) else "pi")


def _pid_axes(config: dict) -> tuple[str, ...]:
    return {"p": ("kp",), "pi": ("kp", "ki"), "pd": ("kp", "kd"),
            "pid": ("kp", "ki", "kd")}[_pid_mode(config)]


def _complete_gains(config: dict, gains: dict) -> dict:
    result = {"ki": config["baseline"]["ki"], "kd": _fixed_kd(config), **gains}
    if _pid_mode(config) == "p":
        result["ki"] = result["kd"] = 0.0
    elif _pid_mode(config) == "pd":
        result["ki"] = 0.0
    elif _pid_mode(config) == "pi":
        result["kd"] = _fixed_kd(config)
    return result


def _search_bounds(config: dict) -> dict:
    return {axis: config["ranges"][axis] for axis in _pid_axes(config)}


def _fixed_kd(config: dict) -> float:
    return float(config.get("kd", config.get("baseline", {}).get("kd", 0.0)))


def _targets(config: dict) -> list[float]:
    """A candidate is evaluated against the same ordered conditions each round."""
    return config["target_values"] if "target_values" in config else [config["target_rad_s"]]


def _response_config(config: dict) -> dict:
    """Scoring units/geometry are explicit, never inferred from PID ID or labels."""
    raw = config.get("response", {})
    if not isinstance(raw, dict):
        raise ValueError("response 必须是JSON对象")
    mode = raw.get("mode", "tracking")
    if mode not in {"tracking", "position"}:
        raise ValueError("response.mode must be tracking or position")
    result = {"mode": mode,
              "initial_reference": raw.get("initial_reference", "first_sample" if mode == "position" else "fixed"),
              "initial_value": float(raw.get("initial_value", 0.0)),
              "normalization_floor": float(raw.get("normalization_floor", 1.0 if mode == "position" else 1e-6)),
              "settling_tolerance_abs": float(raw.get("settling_tolerance_abs", 1.0 if mode == "position" else 0.0)),
              "settling_tolerance_ratio": float(raw.get("settling_tolerance_ratio", 0.05)),
              "cyclic_period": float(raw.get("cyclic_period", 0.0))}
    if result["initial_reference"] not in {"first_sample", "fixed"}:
        raise ValueError("response.initial_reference must be first_sample or fixed")
    if any(not math.isfinite(v) for v in result.values() if isinstance(v, float)):
        raise ValueError("评分参数必须为有限数值")
    if (result["normalization_floor"] <= 0 or result["settling_tolerance_abs"] < 0 or
            result["settling_tolerance_ratio"] <= 0 or result["cyclic_period"] < 0):
        raise ValueError("评分尺度必须为正，绝对容差/周期不能为负")
    return result


def _cycle_delta(value: float, period: float) -> float:
    return (value + period * 0.5) % period - period * 0.5 if period else value


def load_tune_config(path: Path = DEFAULT_TUNE_CONFIG) -> dict:
    """Validate protocol, scoring and search settings independently of the plant."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_targets = cfg["target_values"] if "target_values" in cfg else [cfg["target_rad_s"]]
    if not isinstance(raw_targets, list) or not 1 <= len(raw_targets) <= 10:
        raise ValueError("target_values 必须是1～10个目标值的列表")
    targets = [float(value) for value in raw_targets]
    if len(set(targets)) != len(targets):
        raise ValueError("target_values 不允许重复目标")
    target = targets[0]
    if "target_values" in cfg and "target_rad_s" in cfg and float(cfg["target_rad_s"]) != target:
        raise ValueError("target_rad_s 应与 target_values 首个目标一致；GUI保存会自动同步")
    worst_weight = float(cfg.get("multi_target_worst_weight", 0.3))
    if not math.isfinite(worst_weight) or not 0 <= worst_weight <= 1:
        raise ValueError("multi_target_worst_weight 必须在0～1之间")
    duration_ms = int(cfg["duration_ms"])
    delay_s = float(cfg["inter_trial_delay_s"])
    max_speed = float(cfg.get("feedback_abs_limit", cfg.get("max_feedback_rad_s", math.nan)))
    response = _response_config(cfg)
    protocol_limits_cfg = cfg.get("protocol_limits", {})
    target_max = float(protocol_limits_cfg.get("target_max", protocol_limits_cfg.get("target_max_rad_s", RUN_TARGET_MAX)))
    target_min = float(protocol_limits_cfg.get("target_min", 0.0))
    min_inclusive = protocol_limits_cfg.get("target_min_inclusive", False)
    max_inclusive = protocol_limits_cfg.get("target_max_inclusive", True)
    duration_max = int(protocol_limits_cfg.get("duration_max_ms", RUN_DURATION_MAX_MS))
    baseline = {key: float(cfg["baseline"][key]) for key in ("kp", "ki")}
    bounds = {key: tuple(float(v) for v in cfg["ranges"][key]) for key in ("kp", "ki")}
    pid_mode = _pid_mode(cfg)
    if not isinstance(pid_mode, str) or pid_mode not in {"p", "pi", "pd", "pid"}:
        raise ValueError("pid_mode must be p, pi, pd or pid")
    tune_d = pid_mode in {"pd", "pid"}
    supports_kd = protocol_limits_cfg.get("supports_kd", False)
    if "tune_d" in cfg and "pid_mode" not in cfg and not isinstance(cfg["tune_d"], bool):
        raise ValueError("tune_d must be a JSON boolean")
    kd = float(cfg["baseline"].get("kd", 0.0))
    if "kd" in cfg["ranges"] or tune_d:
        bounds["kd"] = tuple(float(v) for v in cfg["ranges"].get("kd", (0.0, 0.02)))
        baseline["kd"] = kd
    elif "kd" in cfg["baseline"]:
        baseline["kd"] = kd
    initial = int(cfg["initial_random_trials"])
    bayes = int(cfg.get("bayes_trials", max(0, int(cfg.get("rounds_per_batch", 1 + initial)) - 1 - initial)))
    seed = int(cfg["seed"])
    run_mode = str(cfg.get("run_mode", "bayes_only"))
    operation_mode = str(cfg.get("operation_mode", "bayes"))
    llm_interval = int(cfg.get("llm_interval_rounds", 5))
    rounds_per_batch = int(cfg.get("rounds_per_batch", 1 + initial + bayes))
    fixed_cfg = cfg.get("fixed_repeat", {})
    fixed_gains = {
        "kp": float(fixed_cfg.get("kp", 0.25170)),
        "ki": float(fixed_cfg.get("ki", 3.55416)),
    }
    fixed_repeats = int(fixed_cfg.get("repeats", 3))
    fixed_kd = float(fixed_cfg.get("kd", 0.0))
    if "kd" in fixed_cfg or "kd" in bounds:
        fixed_gains["kd"] = fixed_kd
    if pid_mode in {"p", "pd"}:
        baseline["ki"] = fixed_gains["ki"] = 0.0
    if pid_mode == "p":
        baseline["kd"] = fixed_gains["kd"] = kd = fixed_kd = 0.0
    if any(not math.isfinite(v) or v < 0 for v in (kd, fixed_kd)):
        raise ValueError("Kd must be finite and non-negative")
    pid_id = str(cfg.get("pid_id", "speed_pi")).strip()
    history_mode = str(cfg.get("history_mode", "new"))
    guard_cfg = cfg.get("trial_guard", {})
    trial_guard = {
        "enabled": guard_cfg.get("enabled", response["mode"] == "tracking"),
        "startup_grace_ms": int(guard_cfg.get("startup_grace_ms", min(100, duration_ms))),
        "reverse_hold_ms": int(guard_cfg.get("reverse_hold_ms", 100)),
        "stall_hold_ms": int(guard_cfg.get("stall_hold_ms", 500)),
    }

    if duration_max < 1:
        raise ValueError("protocol_limits.duration_max_ms must be positive")
    for value in targets:
        validate_target_range(value, target_min, target_max, min_inclusive, max_inclusive)
    if not 1 <= duration_ms <= duration_max:
        raise ValueError(f"duration_ms must be in 1..{duration_max}")
    if not 0.0 <= delay_s <= 10.0:
        raise ValueError("inter_trial_delay_s must be in 0..10")
    if not math.isfinite(max_speed) or max_speed <= 0 or max_speed < max(abs(v) for v in targets):
        raise ValueError("feedback_abs_limit 必须为正且覆盖全部目标的绝对值")
    if not isinstance(trial_guard["enabled"], bool):
        raise ValueError("trial_guard.enabled 必须为JSON布尔值")
    if trial_guard["enabled"] and any(v <= 0 for v in targets):
        raise ValueError("负向/近零判定只适用于正目标；位置或负目标请关闭 trial_guard.enabled")
    if not 0 <= trial_guard["startup_grace_ms"] <= duration_ms:
        raise ValueError("trial_guard.startup_grace_ms must be in 0..duration_ms")
    if trial_guard["reverse_hold_ms"] < 1 or trial_guard["stall_hold_ms"] < 1:
        raise ValueError("trial_guard hold times must be positive")
    if initial < 1 or bayes < 0 or not 2 <= rounds_per_batch <= 20:
        raise ValueError("rounds_per_batch must be in 2..20")
    if history_mode not in {"new", "continue"}:
        raise ValueError("history_mode must be 'new' or 'continue'")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", pid_id):
        raise ValueError("pid_id must be 1..40 ASCII letters, digits, _ or -")
    if run_mode not in {"bayes_only", "bayes_llm"}:
        raise ValueError("run_mode must be 'bayes_only' or 'bayes_llm'")
    if operation_mode not in {"bayes", "fixed_repeat"}:
        raise ValueError("operation_mode must be 'bayes' or 'fixed_repeat'")
    if llm_interval < 1:
        raise ValueError("llm_interval_rounds must be at least 1")
    if not all(math.isfinite(v) and v >= 0 for v in baseline.values()):
        raise ValueError("baseline gains must be finite and non-negative")
    for key, pair in bounds.items():
        if (
            len(pair) != 2
            or not all(math.isfinite(value) for value in pair)
            or not (0 <= pair[0] < pair[1])
        ):
            raise ValueError(f"{key} range must be finite, non-negative, and increasing")
        inactive = (pid_mode == "p" and key in {"ki", "kd"}) or (pid_mode == "pd" and key == "ki")
        if not inactive:
            if not pair[0] <= baseline[key] <= pair[1]:
                raise ValueError(f"baseline {key} is outside its search range")
    if operation_mode == "fixed_repeat":
        if not 1 <= fixed_repeats <= 20:
            raise ValueError("fixed_repeat.repeats must be in 1..20")
        if not all(math.isfinite(value) and value >= 0 for value in fixed_gains.values()):
            raise ValueError("fixed_repeat Kp/Ki must be finite and non-negative")
        if any(
            not pair[0] <= fixed_gains[key] <= pair[1]
            for key, pair in bounds.items()
            if not ((pid_mode == "p" and key in {"ki", "kd"}) or (pid_mode == "pd" and key == "ki"))
        ):
            raise ValueError("fixed_repeat Kp/Ki must be inside the configured firmware ranges")

    return {
        "target_rad_s": target,
        "target_values": targets,
        "multi_target_worst_weight": worst_weight,
        "duration_ms": duration_ms,
        "inter_trial_delay_s": delay_s,
        "max_feedback_rad_s": max_speed,
        "feedback_abs_limit": max_speed,
        "response": response,
        "trial_guard": trial_guard,
        "protocol_limits": {
            "target_max_rad_s": target_max,
            "duration_max_ms": duration_max,
            "supports_kd": supports_kd,
            **({"target_min": target_min, "target_max": target_max,
                "target_min_inclusive": min_inclusive, "target_max_inclusive": max_inclusive}
               if any(key in protocol_limits_cfg for key in ("target_min", "target_max")) else {}),
        },
        "baseline": baseline,
        "ranges": bounds,
        "initial_random_trials": initial,
        "bayes_trials": bayes,
        "seed": seed,
        "run_mode": run_mode,
        "operation_mode": operation_mode,
        "fixed_repeat": {**fixed_gains, "repeats": fixed_repeats},
        "llm_interval_rounds": llm_interval,
        "rounds_per_batch": rounds_per_batch,
        "pid_id": pid_id,
        "history_mode": history_mode,
        "kd": kd,
        "tune_d": tune_d,
        "pid_mode": pid_mode,
        "experiment_name": str(cfg.get("experiment_name", "速度环")),
        "feedback_unit": str(cfg.get("feedback_unit", "rad/s")),
    }


def score_speed_response(samples: list[dict], target_rad_s: float, duration_ms: int) -> dict:
    """Compatibility wrapper for the original zero-origin positive speed score."""
    return score_response(samples, target_rad_s, duration_ms)


def score_response(samples: list[dict], target: float, duration_ms: int,
                   response: dict | None = None) -> dict:
    """Unit-neutral signed-step score, with optional cyclic coordinate geometry.

    Initial feedback from the first DATA is an estimate (after command latency),
    not an exact pre-step sample. Store that fact; do not call a zero-size hold a
    measured step. Preserve the legacy positive-speed score through defaults.
    """
    settings = _response_config({"response": response or {}})
    if len(samples) < 3:
        raise ValueError("fewer than three telemetry samples received")
    samples = sorted(samples, key=lambda row: row["elapsed_ms"])
    if samples[-1]["elapsed_ms"] < duration_ms * 0.8:
        raise ValueError("telemetry ended before covering 80% of the test window")

    target = float(target)
    if not math.isfinite(target) or any(not math.isfinite(float(row["feedback"])) for row in samples):
        raise ValueError("目标和反馈必须为有限数值")
    period = settings["cyclic_period"]
    initial = (float(samples[0]["feedback"]) if settings["initial_reference"] == "first_sample"
               else settings["initial_value"])
    step = _cycle_delta(target - initial, period)
    amplitude = abs(step)
    scale = max(amplitude, settings["normalization_floor"])
    duration_s = duration_ms / 1000.0
    errors = [abs(_cycle_delta(target - row["feedback"], period)) for row in samples]
    iae = 0.0
    for index in range(1, len(samples)):
        dt = (samples[index]["elapsed_ms"] - samples[index - 1]["elapsed_ms"]) / 1000.0
        iae += (errors[index] + errors[index - 1]) * 0.5 * max(dt, 0.0)
    iae_norm = iae / (scale * duration_s)
    peak_speed = max(row["feedback"] for row in samples)
    direction = 1.0 if step > 0 else -1.0
    previous, displacement = initial, 0.0
    projected = []
    for row in samples:
        displacement += _cycle_delta(row["feedback"] - previous, period)
        previous = row["feedback"]
        projected.append(direction * displacement)
    overshoot = max(0.0, (max(projected) - amplitude) / scale) if amplitude else 0.0
    tolerance = max(settings["settling_tolerance_abs"], amplitude * settings["settling_tolerance_ratio"])
    last_outside = max(
        (i for i, error in enumerate(errors) if error > tolerance), default=-1
    )
    settling_s = (
        samples[last_outside + 1]["elapsed_ms"] / 1000.0
        if last_outside + 1 < len(samples)
        else duration_s
    )
    settling_fraction = min(1.0, settling_s / duration_s)
    score = 0.70 * iae_norm + 0.20 * overshoot + 0.10 * settling_fraction
    return {
        "valid": True,
        "score": score,
        "iae_norm": iae_norm,
        "overshoot_ratio": overshoot,
        "settling_time_s": settling_s,
        "final_speed_rad_s": samples[-1]["feedback"],
        "peak_speed_rad_s": peak_speed,
        "final_feedback": samples[-1]["feedback"],
        "peak_feedback": peak_speed,
        "initial_feedback": initial,
        "initial_reference": settings["initial_reference"],
        "step_amplitude": amplitude,
        "normalization_scale": scale,
        "settling_tolerance": tolerance,
        "final_error_abs": errors[-1],
        "cyclic_period": period,
        "hold_only": amplitude == 0.0,
        "sample_count": len(samples),
    }


def run_hardware_trial(
    link: Any,
    kp: float,
    ki: float,
    *,
    target_rad_s: float,
    duration_ms: int,
    max_feedback_rad_s: float,
    telemetry_timeout_s: float = 1.0,
    on_sample: Any = None,
    pid_ranges: dict[str, tuple[float, float]] | None = None,
    target_max_rad_s: float = RUN_TARGET_MAX,
    duration_max_ms: int = RUN_DURATION_MAX_MS,
    trial_guard: dict[str, int] | None = None,
    kd: float = 0.0,
    supports_kd: bool = False,
    target_min: float | None = None,
    target_max_inclusive: bool = True,
    target_min_inclusive: bool = False,
    response: dict | None = None,
) -> dict:
    """Run one step, stream DATA to the caller, and always try to send STOP."""
    if not math.isfinite(kd) or kd < 0:
        raise ValueError("Kd must be finite and non-negative")
    step_options = {"target_max_rad_s": target_max_rad_s, "duration_max_ms": duration_max_ms}
    if target_min is not None:
        step_options.update(target_min=target_min, target_max=target_max_rad_s,
                            target_min_inclusive=target_min_inclusive,
                            target_max_inclusive=target_max_inclusive)
    # Pure preflight: reject invalid wire commands before SET_PID or any I/O.
    encode_run_step(target_rad_s, duration_ms, **step_options)
    samples: list[dict] = []
    fault = ""
    run_may_be_active = False
    stop_error = None
    guard = trial_guard or {"startup_grace_ms": min(100, duration_ms),
                            "reverse_hold_ms": 100, "stall_hold_ms": 500}
    response_settings = _response_config({"response": response or {}})
    guard_enabled = guard.get("enabled", response_settings["mode"] == "tracking")
    reverse_since_ms = None
    stall_since_ms = None
    try:
        link.clear_pending_messages()
        pid_options = {} if pid_ranges is None else {
            "kp_range": pid_ranges["kp"], "ki_range": pid_ranges["ki"],
        }
        if pid_ranges and "kd" in pid_ranges:
            pid_options["kd_range"] = pid_ranges["kd"]
        link.set_pid(kp, ki, kd, **pid_options)
        run_may_be_active = True
        link.run_step(
            target_rad_s,
            duration_ms,
            **step_options,
        )
        deadline = time.monotonic() + duration_ms / 1000.0 + telemetry_timeout_s
        while time.monotonic() < deadline:
            try:
                message = link.read_message(min(0.1, max(0.01, deadline - time.monotonic())))
            except TimeoutError:
                continue
            if message.kind != "DATA":
                continue
            telemetry = parse_telemetry(message)
            row = {
                "elapsed_ms": telemetry.elapsed_ms,
                "target_rad_s": telemetry.target,
                "target": telemetry.target,
                "feedback": telemetry.feedback,
            }
            samples.append(row)
            if on_sample is not None:
                on_sample(row)
            if (len(samples) > 1 and telemetry.elapsed_ms <= samples[-2]["elapsed_ms"]):
                fault = "telemetry_time_not_increasing"
                break
            # Firmware sends three decimal places; allow its quantization but reject old-trial DATA.
            if abs(telemetry.target - target_rad_s) > 0.001:
                fault = "telemetry_target_mismatch"
                break
            if abs(telemetry.feedback) > max_feedback_rad_s:
                fault = "host_overspeed_stop"
                break
            if guard_enabled and telemetry.elapsed_ms >= guard["startup_grace_ms"]:
                reverse_limit = max(0.1, target_rad_s * 0.05)
                if telemetry.feedback < -reverse_limit:
                    reverse_since_ms = (telemetry.elapsed_ms if reverse_since_ms is None
                                        else reverse_since_ms)
                else:
                    reverse_since_ms = None
                if abs(telemetry.feedback) < target_rad_s * 0.05:
                    stall_since_ms = (telemetry.elapsed_ms if stall_since_ms is None
                                      else stall_since_ms)
                else:
                    stall_since_ms = None
                if (reverse_since_ms is not None and
                        telemetry.elapsed_ms - reverse_since_ms >= guard["reverse_hold_ms"]):
                    fault = "sustained_reverse_stop"
                    break
                if (stall_since_ms is not None and
                        telemetry.elapsed_ms - stall_since_ms >= guard["stall_hold_ms"]):
                    fault = "sustained_stall_stop"
                    break
            if telemetry.elapsed_ms >= duration_ms:
                break
        if fault:
            metrics = {
                "valid": False,
                "score": 10.0,
                "fault": fault,
                "sample_count": len(samples),
                "final_speed_rad_s": samples[-1]["feedback"] if samples else 0.0,
                "peak_speed_rad_s": max(
                    (abs(row["feedback"]) for row in samples), default=0.0
                ),
            }
        elif len(samples) < 3 or samples[-1]["elapsed_ms"] < duration_ms * 0.8:
            metrics = {
                "valid": False,
                "score": 10.0,
                "fault": "telemetry_incomplete",
                "sample_count": len(samples),
                "last_elapsed_ms": samples[-1]["elapsed_ms"] if samples else 0,
                "final_speed_rad_s": samples[-1]["feedback"] if samples else 0.0,
                "peak_speed_rad_s": max(
                    (abs(row["feedback"]) for row in samples), default=0.0
                ),
            }
        else:
            metrics = score_response(samples, target_rad_s, duration_ms, response)
    except HardwareStopRequested:
        metrics = {
            "valid": False, "score": 10.0, "fault": "user_stop",
            "sample_count": len(samples),
            "final_speed_rad_s": samples[-1]["feedback"] if samples else 0.0,
        }
    except BaseException:
        if run_may_be_active:
            try:
                link.stop()
            except Exception:
                pass
            run_may_be_active = False
        raise
    finally:
        if run_may_be_active:
            try:
                link.stop()
            except Exception as exc:
                stop_error = exc

    if stop_error is not None:
        previous_fault = metrics.get("fault", "")
        metrics["valid"] = False
        metrics["score"] = 10.0
        metrics["fault"] = (
            f"{previous_fault}; stop_ack_timeout" if previous_fault else "stop_ack_timeout"
        )
        metrics["stop_ack_error"] = str(stop_error)[:200]
    return {"metrics": metrics, "samples": samples}


def _evaluate_candidate(link: Any, config: dict, gains: dict, run_dir: Path,
                        number: int, prefix: str) -> dict:
    """Run every fixed target, persist even partial trials, aggregate only valid results.

    Each RUN_STEP has its own STOP acknowledgement. A failure ends the candidate
    and batch; it is never averaged away by successful conditions.
    """
    targets = _targets(config)
    results = []
    for index, target in enumerate(targets, 1):
        print(f"  目标测试 {index}/{len(targets)}：目标={target:g}")
        last_print_ms = [-100]

        def show_sample(row: dict) -> None:
            if row["elapsed_ms"] - last_print_ms[0] >= 100:
                print(f"  {row['elapsed_ms']:4d} ms 目标={row['target_rad_s']:.3f} "
                      f"反馈={row['feedback']:.3f} {config.get('feedback_unit', 'rad/s')}")
                last_print_ms[0] = row["elapsed_ms"]

        send_ranges = dict(config["ranges"])
        if _pid_mode(config) in {"p", "pd"}:
            send_ranges["ki"] = (0.0, 0.0)  # Fixed zero is not a searched Ki range.
        if _pid_mode(config) == "p":
            send_ranges["kd"] = (0.0, 0.0)
        trial = run_hardware_trial(
            link, gains["kp"], gains["ki"], kd=gains.get("kd", 0.0),
            target_rad_s=target, duration_ms=config["duration_ms"],
            max_feedback_rad_s=config["max_feedback_rad_s"], on_sample=show_sample,
            pid_ranges=send_ranges, trial_guard=config.get("trial_guard"),
            target_max_rad_s=config["protocol_limits"]["target_max_rad_s"],
            duration_max_ms=config["protocol_limits"]["duration_max_ms"],
            target_min=config["protocol_limits"].get("target_min"),
            target_min_inclusive=config["protocol_limits"].get("target_min_inclusive", False),
            target_max_inclusive=config["protocol_limits"].get("target_max_inclusive", True),
            response=config.get("response"),
        )
        suffix = f"_target_{index:02d}" if len(targets) > 1 else ""
        trace = f"traces/{prefix}_{number:03d}{suffix}.csv"
        with (run_dir / trace).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("elapsed_ms", "target", "feedback"))
            writer.writeheader()
            writer.writerows({key: row[key] for key in ("elapsed_ms", "target", "feedback")} for row in trial["samples"])
        results.append({"target": target, "metrics": trial["metrics"], "trace_file": trace})
        if not trial["metrics"].get("valid"):
            break
        if index < len(targets):
            time.sleep(config["inter_trial_delay_s"])

    metrics = dict(results[-1]["metrics"])
    if len(results) > 1 and all(item["metrics"].get("valid") for item in results):
        scores = [item["metrics"]["score"] for item in results]
        mean_score, worst_score = statistics.fmean(scores), max(scores)
        weight = config.get("multi_target_worst_weight", 0.3)
        metrics.update(score=(1 - weight) * mean_score + weight * worst_score,
                       mean_target_score=mean_score, worst_target_score=worst_score,
                       iae_norm=statistics.fmean(item["metrics"]["iae_norm"] for item in results),
                       overshoot_ratio=max(item["metrics"]["overshoot_ratio"] for item in results),
                       settling_time_s=max(item["metrics"]["settling_time_s"] for item in results),
                       sample_count=sum(item["metrics"]["sample_count"] for item in results))
    return {"metrics": metrics, "target_trials": results, "trace_file": results[0]["trace_file"]}


def run_fixed_hardware_validation(
    link: Any, config: dict, output_root: Path
) -> dict:
    """Repeat one fixed PID setting, score each trial, and keep a separate report."""
    fixed = config["fixed_repeat"]
    kp, ki, repeats = fixed["kp"], fixed["ki"], fixed["repeats"]
    kd = float(fixed.get("kd", 0.0))
    if _pid_mode(config) in {"p", "pd"}:
        ki = 0.0
    if _pid_mode(config) == "p":
        kd = 0.0
    unit = config.get("feedback_unit", "rad/s")
    target, duration_ms = config["target_rad_s"], config["duration_ms"]
    output_root = Path(output_root) / "motor_bo_validation"
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    (run_dir / "traces").mkdir(parents=True, exist_ok=False)

    manifest = {
        "status": "running",
        "mode": "fixed_repeat_validation",
        "ai_call_count": 0,
        "bayesian_optimization": False,
        "config": config,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rounds": repeats,
    }
    _write_json(run_dir / "manifest.json", manifest)
    records: list[dict] = []

    def write_summary() -> dict:
        valid_scores = [
            float(record["metrics"]["score"])
            for record in records
            if record["metrics"].get("valid", False)
        ]
        summary = {
            "mode": "fixed_repeat_validation",
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "target_rad_s": target,
            "duration_ms": duration_ms,
            "requested_repeats": repeats,
            "target_values": _targets(config),
            "multi_target_worst_weight": config.get("multi_target_worst_weight", 0.3),
            "completed_repeats": len(records),
            "valid_score_count": len(valid_scores),
            "mean_score": statistics.fmean(valid_scores) if valid_scores else None,
            "score_stddev": statistics.pstdev(valid_scores) if valid_scores else None,
            "trials": records,
        }
        _write_json(run_dir / "summary.json", summary)
        return summary

    try:
        for round_number in range(1, repeats + 1):
            print(
                f"\n固定复测 {round_number}/{repeats}：Kp={kp:.5f}, Ki={ki:.5f}, "
                f"Kd={kd:.5f}, 目标={target:.3f} {unit}"
            )
            trial = _evaluate_candidate(link, config, {"kp": kp, "ki": ki, "kd": kd},
                                        run_dir, round_number, "trial")
            record = {
                "round": round_number,
                "gains": {"kp": kp, "ki": ki, "kd": kd},
                "metrics": trial["metrics"],
                "trace_file": trial["trace_file"],
                "target_trials": trial["target_trials"],
            }
            records.append(record)
            with (run_dir / "rounds.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            _write_speed_response_svg(run_dir, records, target, duration_ms, feedback_unit=unit)

            metrics = trial["metrics"]
            if metrics.get("valid", False):
                print(
                    f"  评分={metrics['score']:.4f}，末反馈="
                    f"{metrics.get('final_speed_rad_s', 0.0):.3f} {unit}"
                )
            else:
                print(
                    f"  本轮无效：{metrics.get('fault', 'invalid_trial')}；"
                    "停止剩余复测。"
                )
                manifest["status"] = (
                    "stopped" if metrics.get("fault") == "user_stop" else "trial_aborted"
                )
                manifest["fault"] = metrics.get("fault", "invalid_trial")
                break

            manifest["completed_repeats"] = len(records)
            _write_json(run_dir / "manifest.json", manifest)
            if round_number < repeats:
                time.sleep(config["inter_trial_delay_s"])

        if manifest["status"] == "running":
            manifest["status"] = "completed"
        summary = write_summary()
        manifest["completed_repeats"] = len(records)
        manifest["mean_score"] = summary["mean_score"]
        manifest["score_stddev"] = summary["score_stddev"]
        _write_json(run_dir / "manifest.json", manifest)
        if summary["mean_score"] is not None:
            print(
                f"\n复测评分均值={summary['mean_score']:.4f}，"
                f"标准差={summary['score_stddev']:.4f}"
            )
        return {"status": manifest["status"], "run_dir": str(run_dir), "summary": summary}
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["completed_repeats"] = len(records)
        manifest["error"] = str(exc)[:300]
        write_summary()
        _write_json(run_dir / "manifest.json", manifest)
        raise


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _make_optimizer(config: dict, history: list[dict] | None = None) -> BayesianOptimization:
    optimizer = BayesianOptimization(
        f=None,
        pbounds={key: tuple(bounds) for key, bounds in _search_bounds(config).items()},
        acquisition_function=ExpectedImprovement(xi=XI_BY_MODE["balanced"]),
        random_state=config["seed"] + int(config.get("batch_index", 1)) - 1,
        verbose=0,
        allow_duplicate_points=True,
    )
    if history:
        selected = history[-100:]
        best = min(history, key=lambda row: row["score"])
        if best not in selected:
            selected = [best, *selected[:-1]]
        for row in selected:
            gains = row["gains"]
            if all(config["ranges"][axis][0] <= gains[axis] <= config["ranges"][axis][1]
                   for axis in _pid_axes(config)):
                optimizer.register(
                    params={axis: gains[axis] for axis in _pid_axes(config)},
                    target=-row["score"],
                )
    optimizer.set_bounds({axis: config.get("search_ranges", config["ranges"])[axis]
                          for axis in _pid_axes(config)})
    return optimizer


def make_hardware_advisor(config: dict) -> Any:
    """Create the live advisor only when explicitly selected in the hardware config."""
    if config.get("run_mode", "bayes_only") == "bayes_llm":
        return HardwareLiveAdvisor()
    return None


def load_paused_hardware_config(run_dir: Path) -> dict:
    """Read an immutable run snapshot, accepting only explicitly paused runs."""
    manifest_path = Path(run_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "paused":
        raise ValueError("only an AI-paused hardware run can be resumed")
    return manifest["config"]


def _write_advice_header(path: Path, config: dict) -> None:
    if config["run_mode"] == "bayes_only":
        text = "# 实机 AI 建议记录\n\n本次为纯贝叶斯模式，没有调用 AI。\n"
    else:
        text = (
            "# 实机 AI 建议记录\n\n"
            f"初始样本完成后首次咨询，之后每 {config['llm_interval_rounds']} 轮咨询一次。\n\n"
            "| 轮次 | 动作 | 探索倾向 | AI 理由 |\n| ---: | --- | --- | --- |\n"
        )
    path.write_text(text, encoding="utf-8")


def _append_advice(path: Path, round_number: int, decision: dict) -> None:
    reason = " ".join(str(decision["reason"]).replace("|", "｜").split())
    action = ACTION_ZH[decision["action"]]
    exploration = EXPLORATION_ZH[decision["exploration"]]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"| {round_number:02d} | {action} | {exploration} | {reason} |\n")
    print(f"AI建议 第{round_number:02d}轮：动作={action}，探索倾向={exploration}，理由：{reason}")


def _read_trace(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                "elapsed_ms": float(row["elapsed_ms"]),
                "target_rad_s": float(row.get("target", row.get("target_rad_s"))),
                "feedback": float(row["feedback"]),
            }
            for row in csv.DictReader(handle)
        ]


def _downsample_speed_points(samples: list[dict], maximum: int = 40) -> list[dict]:
    if len(samples) <= maximum:
        selected = samples
    else:
        indices = [round(i * (len(samples) - 1) / (maximum - 1)) for i in range(maximum)]
        selected = [samples[index] for index in indices]
    return [
        {"elapsed_ms": int(row["elapsed_ms"]), "feedback_rad_s": float(row["feedback"])}
        for row in selected
    ]


def _advisor_record(run_dir: Path, record: dict) -> dict:
    return {
        "round": record["round"],
        "gains": record["gains"],
        "metrics": record["metrics"],
        "target_trials": [
            {"target": trial["target"], "metrics": trial["metrics"],
             "feedback_points": _downsample_speed_points(_read_trace(run_dir / trial["trace_file"]))}
            for trial in record.get("target_trials", [])
        ],
        "speed_points": _downsample_speed_points(
            _read_trace(run_dir / record["trace_file"])
        ),
    }


def _query_ai_this_round(mode: str, round_number: int, total: int, initial_rounds: int, interval: int) -> bool:
    """Match simulation cadence: query after initial design, then at intervals, never at the end."""
    if mode != "bayes_llm" or round_number >= total or round_number < initial_rounds:
        return False
    return round_number == initial_rounds or (round_number - initial_rounds) % interval == 0


def _set_exploration(optimizer: BayesianOptimization, exploration: str) -> None:
    params = optimizer.acquisition_function.get_acquisition_params()
    params["xi"] = XI_BY_MODE[exploration]
    optimizer.acquisition_function.set_acquisition_params(params)


def _advice_validation_error(reply: object) -> str | None:
    """Explain why an advisor reply was rejected, without trusting it as control data."""
    if not isinstance(reply, dict):
        return "AI 回复不是 JSON 对象"
    forbidden = {"p", "i", "d", "kp", "ki", "kd", "gains", "parameters"}
    if forbidden.intersection(key.lower() for key in reply if isinstance(key, str)):
        return "AI 回复包含 PID 参数字段"
    if str(reply.get("action", "")).strip().lower() not in {"continue", "pause", "stop"}:
        return "AI 回复的 action 不在允许值内"
    if str(reply.get("exploration", "")).strip().lower() not in XI_BY_MODE:
        return "AI 回复的 exploration 不在允许值内"
    return None


def _write_speed_response_svg(
    run_dir: Path, records: list[dict], target_rad_s: float, duration_ms: int,
    *, feedback_unit: str = "rad/s",
) -> None:
    """Update one dependency-free SVG overlay with all recorded trial curves."""
    from html import escape

    traces = [(record, _read_trace(run_dir / record["trace_file"]))
              for record in expand_curves(records, target_rad_s)]
    targets = sorted({record["target"] for record, _ in traces}) or [target_rad_s]
    values = [*targets, 0.0]
    values.extend(row["feedback"] for _, trace in traces for row in trace)
    low, high = min(values), max(values)
    padding = max((high - low) * 0.08, 0.5)
    low -= padding
    high += padding

    width = 1200
    left, right, top = 82, 28, 72
    plot_width, plot_height = width - left - right, 500
    plot_bottom = top + plot_height
    legend_rows = max(1, (len(traces) + 1) // 2)
    legend_top = plot_bottom + 76
    height = legend_top + legend_rows * 24 + 32
    duration_s = duration_ms / 1000.0

    def map_x(elapsed_ms: float) -> float:
        return left + elapsed_ms / duration_ms * plot_width

    def map_y(speed: float) -> float:
        return top + (high - speed) / (high - low) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="82" y="32" font-family="sans-serif" font-size="20" font-weight="bold">'
        f'实机响应：目标 {escape(", ".join(f"{value:g}" for value in targets))} {escape(feedback_unit)}，每目标时长 {duration_s:g} s</text>',
    ]

    for tick in range(6):
        speed = low + (high - low) * tick / 5
        y = map_y(speed)
        svg.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" '
            'stroke="#e2e6eb"/><text x="8" y="{:.1f}" font-family="sans-serif" '
            'font-size="12" fill="#555">{:.2f}</text>'.format(y + 4, speed)
        )
    for tick in range(6):
        elapsed = duration_ms * tick / 5
        x = map_x(elapsed)
        svg.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{plot_bottom}" '
            'stroke="#eef0f3"/><text x="{:.1f}" y="{}" font-family="sans-serif" '
            'font-size="12" text-anchor="middle" fill="#555">{:.2f}</text>'.format(
                x, plot_bottom + 22, elapsed / 1000.0
            )
        )

    for target in targets:
        target_y = map_y(target)
        svg.append(f'<line x1="{left}" y1="{target_y:.1f}" x2="{left + plot_width}" '
                   f'y2="{target_y:.1f}" stroke="#333" stroke-dasharray="8 6"/>'
                   f'<text x="{left + 8}" y="{target_y - 5:.1f}" font-size="12">目标 {target:g}</text>')
    svg.append(
        f'<path d="M{left},{top} V{plot_bottom} H{left + plot_width}" '
        'fill="none" stroke="#444" stroke-width="1.5"/>'
    )
    svg.append(
        f'<text x="{left + plot_width / 2:.1f}" y="{plot_bottom + 48}" '
        'font-family="sans-serif" font-size="14" text-anchor="middle">时间 (s)</text>'
    )
    svg.append(
        f'<text x="{left}" y="{top - 10}" font-family="sans-serif" font-size="13">反馈 ({escape(feedback_unit)})</text>'
    )
    svg.append(
        f'<line x1="{left + plot_width - 120}" y1="42" x2="{left + plot_width - 88}" '
        'y2="42" stroke="#333" stroke-width="2" stroke-dasharray="8 6"/>'
        f'<text x="{left + plot_width - 82}" y="47" font-family="sans-serif" font-size="12">目标</text>'
    )

    for index, (record, trace) in enumerate(traces):
        hue = (index * 137.508) % 360
        color = f"hsl({hue:.1f}, 68%, 42%)"
        if len(trace) > 1:
            points = " ".join(
                f"{map_x(row['elapsed_ms']):.1f},{map_y(row['feedback']):.1f}"
                for row in trace
            )
            svg.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        elif trace:
            row = trace[0]
            svg.append(
                f'<circle cx="{map_x(row["elapsed_ms"]):.1f}" '
                f'cy="{map_y(row["feedback"]):.1f}" r="3.5" fill="{color}"/>'
            )
        column = index % 2
        row_index = index // 2
        legend_x = left + column * (plot_width / 2)
        legend_y = legend_top + row_index * 24
        gains = record["gains"]
        metrics = record["metrics"]
        label = (
            f"第{record['round']:02d}轮 目标={record['target']:g} Kp={gains['kp']:.4g} Ki={gains['ki']:.4g} Kd={gains.get('kd', 0):.4g} "
            f"评分={metrics['score']:.4g}"
        )
        if not trace:
            label += "（无遥测）"
        svg.append(
            f'<line x1="{legend_x:.1f}" y1="{legend_y + 7:.1f}" '
            f'x2="{legend_x + 28:.1f}" y2="{legend_y + 7:.1f}" stroke="{color}" stroke-width="3"/>'
            f'<text x="{legend_x + 36:.1f}" y="{legend_y + 12:.1f}" '
            f'font-family="sans-serif" font-size="13">{escape(label)}</text>'
        )
    svg.append("</svg>")
    (run_dir / "speed_response.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")


def _read_rounds(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _profile_paths(output_root: Path, pid_id: str, series_id: str | None = None) -> tuple[Path, Path]:
    profile_dir = output_root / "pid_profiles" / pid_id
    return profile_dir / "active.json", profile_dir / f"{series_id}.json"


def inspect_hardware_batch(config: dict, output_root: Path) -> dict:
    """Resolve the active PID history without modifying files or starting hardware."""
    output_root = Path(output_root)
    if config.get("history_mode", "new") == "new":
        return {"history": [], "search_ranges": _search_bounds(config), "batch_index": 1}
    active_path, _ = _profile_paths(output_root, config["pid_id"])
    if not active_path.exists():
        raise ValueError(f"PID {config['pid_id']} has no completed history; select history_mode=new first")
    active = json.loads(active_path.read_text(encoding="utf-8"))
    _, series_path = _profile_paths(output_root, config["pid_id"], active["series_id"])
    profile = json.loads(series_path.read_text(encoding="utf-8"))
    if profile["pid_id"] != config["pid_id"] or not profile["history"]:
        raise ValueError("active PID profile is empty or belongs to another PID")
    if (_pid_mode(profile) != _pid_mode(config) or
            profile.get("feedback_unit", "rad/s") != config.get("feedback_unit", "rad/s") or
            (_pid_mode(config) == "pi" and profile.get("fixed_kd", 0.0) != _fixed_kd(config))):
        raise ValueError("PID搜索维度、固定Kd或反馈单位已改变，请新建历史")
    if (_response_config(profile) != _response_config(config) or
            profile.get("trial_guard", {}).get("enabled", True) != config.get("trial_guard", {}).get("enabled", True)):
        raise ValueError("评分口径、起点、周期或异常规则已改变，请新建历史")
    if (_targets(profile) != _targets(config) or
            profile.get("multi_target_worst_weight", 0.3) != config.get("multi_target_worst_weight", 0.3) or
            profile["duration_ms"] != config["duration_ms"]):
        raise ValueError(
            f"历史试验目标为 {_targets(profile)}、{profile['duration_ms']} ms，"
            f"当前配置为 {_targets(config)}、{config['duration_ms']} ms；"
            "继续旧历史需恢复原目标列表、评分权重和时长，否则将 motor_bo_hw_config.json "
            "的 history_mode 设为 new"
        )
    best = min(profile["history"], key=lambda item: item["score"])
    if any(not config["ranges"][axis][0] <= best["gains"][axis] <= config["ranges"][axis][1]
           for axis in _pid_axes(config)):
        raise ValueError("current hard ranges exclude historical best; review them or start new history")
    proposed = profile.get("next_search_ranges", _search_bounds(config))
    if any(proposed[axis][0] < config["ranges"][axis][0] or
           proposed[axis][1] > config["ranges"][axis][1]
           for axis in _pid_axes(config)):
        raise ValueError("saved next search range exceeds current hard ranges; review config")
    return {
        "history": profile["history"],
        "search_ranges": {axis: tuple(proposed[axis]) for axis in _pid_axes(config)},
        "batch_index": len(profile["batches"]) + 1,
        "series_id": profile["series_id"],
        "profile": profile,
    }


def _validate_next_ranges(reply: object, hard_ranges: dict, best: dict) -> dict:
    """AI may propose a region, never new gains or bounds beyond firmware-approved limits."""
    parser_metadata = {"analysis_summary", "thought_process", "tuning_action"}
    if (not isinstance(reply, dict) or not {"ranges", "reason"}.issubset(reply) or
            set(reply) - parser_metadata - {"ranges", "reason"}):
        raise ValueError("AI must return only ranges and reason")
    proposal = reply["ranges"]
    if not isinstance(proposal, dict) or set(proposal) != set(hard_ranges):
        raise ValueError("AI ranges must contain exactly the optimized PID axes")
    result = {}
    for axis in hard_ranges:
        pair = proposal[axis]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"AI {axis} range must contain two numbers")
        low, high = pair
        if isinstance(low, bool) or isinstance(high, bool):
            raise ValueError(f"AI {axis} range must be numeric")
        low, high = float(low), float(high)
        hard_low, hard_high = hard_ranges[axis]
        if not (math.isfinite(low) and math.isfinite(high) and
                hard_low <= low < high <= hard_high):
            raise ValueError(f"AI {axis} range exceeds hard limits or is invalid")
        if high - low < 0.2 * (hard_high - hard_low):
            raise ValueError(f"AI {axis} range is narrower than 20% of hard range")
        if not low <= best["gains"][axis] <= high:
            raise ValueError(f"AI {axis} range excludes historical best")
        result[axis] = (low, high)
    if not isinstance(reply["reason"], str) or not reply["reason"].strip():
        raise ValueError("AI must explain the next range")
    return result


def _complete_pid_batch(run_dir: Path, manifest: dict, records: list[dict], advisor: Any) -> None:
    """Commit a completed batch to its PID series; paused/failed batches are not committed."""
    config = manifest["config"]
    output_root = run_dir.parent
    active_path, series_path = _profile_paths(output_root, config["pid_id"], manifest["series_id"])
    profile = json.loads(series_path.read_text(encoding="utf-8")) if series_path.exists() else {
        "pid_id": config["pid_id"],
        "series_id": manifest["series_id"],
        "target_rad_s": config["target_rad_s"],
        "duration_ms": config["duration_ms"],
        "history": [],
        "target_values": _targets(config),
        "multi_target_worst_weight": config.get("multi_target_worst_weight", 0.3),
        "batches": [],
        "tune_d": config.get("tune_d", False),
        "pid_mode": _pid_mode(config),
        "fixed_kd": _fixed_kd(config),
        "feedback_unit": config.get("feedback_unit", "rad/s"),
        "response": _response_config(config),
        "trial_guard": config.get("trial_guard", {}),
    }
    new_history = [
        {"gains": row["gains"], "score": row["metrics"]["score"],
         "metrics": row["metrics"], "target_trials": row.get("target_trials", []),
         "run_dir": str(run_dir), "round": row["round"]}
        for row in records if row["metrics"].get("valid")
    ]
    previous_best_score = min((row["score"] for row in profile["history"]), default=None)
    profile["history"].extend(new_history)
    batch_best_score = min(row["score"] for row in new_history)
    best = min(profile["history"], key=lambda item: item["score"])
    next_ranges = config.get("search_ranges", _search_bounds(config))
    advice = None
    advice_error = None
    if config["run_mode"] == "bayes_llm" and advisor is not None:
        summary = {
            "pid_id": config["pid_id"],
            "target_rad_s": config["target_rad_s"],
            "target_values": _targets(config),
            "score_rule": "分数越低越好；综合分=(1-w)*各目标平均分+w*最差目标分",
            "multi_target_worst_weight": config.get("multi_target_worst_weight", 0.3),
            "response": _response_config(config),
            "duration_ms": config["duration_ms"],
            "hard_ranges": _search_bounds(config),
            "pid_mode": _pid_mode(config),
            "experiment_name": config.get("experiment_name", "速度环"),
            "feedback_unit": config.get("feedback_unit", "rad/s"),
            "current_search_ranges": next_ranges,
            "historical_best": best,
            "recent_trials": profile["history"][-20:],
        }
        manifest["llm_call_count"] += 1
        try:
            advice = advisor.recommend_next_range(summary)
            next_ranges = _validate_next_ranges(advice, _search_bounds(config), best)
        except Exception as exc:
            advice_error = " ".join(str(exc).split())[:300]
            next_ranges = config.get("search_ranges", _search_bounds(config))
    profile["next_search_ranges"] = next_ranges
    profile["batches"].append({
        "run_dir": str(run_dir), "rounds": len(records),
        "batch_best_score": batch_best_score, "global_best_score": best["score"],
        "improved": previous_best_score is None or batch_best_score < previous_best_score,
        "range_advice": advice, "range_error": advice_error,
    })
    series_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(series_path, profile)
    _write_json(active_path, {"series_id": manifest["series_id"]})
    manifest["profile_file"] = str(series_path)
    manifest["historical_best"] = best
    manifest["next_search_ranges"] = next_ranges
    manifest["range_advice"] = advice
    manifest["range_advice_error"] = advice_error
    with (run_dir / "ai_advice.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## 下一批搜索范围\n\n")
        handle.write("；".join(f"{axis}={limits}" for axis, limits in next_ranges.items()) + "。")
        if advice_error:
            handle.write(f"AI 建议未采用：{advice_error}；沿用本批范围。\n")
        elif advice:
            handle.write(f"AI 理由：{advice['reason']}\n")
        else:
            handle.write("纯贝叶斯模式沿用本批范围。\n")
    print("下一批搜索范围：" + "，".join(f"{axis}={limits}" for axis, limits in next_ranges.items()))
    if advice_error:
        print(f"批末 AI 建议未采用：{advice_error}；沿用本批范围。")
    elif advice:
        print(f"批末 AI 评价：{' '.join(advice['reason'].split())[:500]}")


def _initial_candidates(config: dict) -> list[dict]:
    rng = random.Random(config["seed"])
    candidates = [_complete_gains(config, dict(config["baseline"]))]
    seen = {tuple(candidates[0].get(key, 0.0) for key in _pid_axes(config))}
    while len(candidates) < 1 + config["initial_random_trials"]:
        candidate = {
            key: rng.uniform(*config.get("search_ranges", config["ranges"])[key])
            for key in _pid_axes(config)
        }
        key = tuple(candidate[name] for name in _pid_axes(config))
        if key not in seen:
            seen.add(key)
            candidates.append(_complete_gains(config, candidate))
    return candidates


def _ensure_advice_log(run_dir: Path, config: dict, records: list[dict]) -> Path:
    path = run_dir / "ai_advice.md"
    if not path.exists():
        _write_advice_header(path, config)
        for record in records:
            if record.get("ai_called"):
                _append_advice(path, record["round"], record["decision"])
    return path


def _run_hardware_rounds(
    link: Any,
    config: dict,
    run_dir: Path,
    manifest: dict,
    optimizer: BayesianOptimization,
    records: list[dict],
    advisor: Any,
) -> dict:
    total = config.get("rounds_per_batch", 1 + config["initial_random_trials"] + config["bayes_trials"])
    continuing = manifest.get("history_count", 0) > 0
    initial_rounds = min(total, config["llm_interval_rounds"]) if continuing else min(
        total, 1 + config["initial_random_trials"]
    )
    candidates = [] if continuing else _initial_candidates(config)[:total]
    exploration = manifest.get("exploration", "balanced")
    if records and records[-1].get("decision"):
        exploration = records[-1]["decision"].get("exploration", exploration)
    _set_exploration(optimizer, exploration)
    manifest["status"] = "running"
    manifest["exploration"] = exploration
    advice_path = _ensure_advice_log(run_dir, config, records)

    try:
        for index in range(len(records), total):
            if index < len(candidates):
                gains = candidates[index]
            else:
                gains = {key: float(value) for key, value in optimizer.suggest().items()}
            gains = _complete_gains(config, gains)
            print(
                f"\n实机第 {index + 1}/{total} 轮：Kp={gains['kp']:.5f}, "
                f"Ki={gains['ki']:.5f}, Kd={gains['kd']:.5f}, "
                f"目标={config['target_rad_s']:.3f} {config.get('feedback_unit', 'rad/s')}"
            )
            round_number = index + 1
            trial = _evaluate_candidate(link, config, gains, run_dir, round_number, "round")
            metrics = trial["metrics"]
            if metrics.get("valid", False):
                optimizer.register(
                    params={key: gains[key] for key in _pid_axes(config)},
                    target=-metrics["score"],
                )

            record = {
                "round": round_number,
                "gains": dict(gains),
                "metrics": metrics,
                "trace_file": trial["trace_file"],
                "target_trials": trial["target_trials"],
                "ai_called": False,
            }
            should_query = metrics.get("valid", False) and _query_ai_this_round(
                config["run_mode"], round_number, total, initial_rounds,
                config["llm_interval_rounds"],
            )
            previous_exploration = exploration
            if should_query:
                current_ai_data = _advisor_record(run_dir, record)
                current_ai_data["target_rad_s"] = config["target_rad_s"]
                current_ai_data["pid_mode"] = _pid_mode(config)
                current_ai_data["response"] = _response_config(config)
                current_ai_data["target_values"] = _targets(config)
                current_ai_data["multi_target_worst_weight"] = config.get("multi_target_worst_weight", 0.3)
                current_ai_data["duration_ms"] = config["duration_ms"]
                current_ai_data["experiment_name"] = config.get("experiment_name", "速度环")
                current_ai_data["feedback_unit"] = config.get("feedback_unit", "rad/s")
                previous_ai_data = [
                    _advisor_record(run_dir, item) for item in records[-4:]
                ]
                manifest["llm_call_count"] = manifest.get("llm_call_count", 0) + 1
                try:
                    reply = advisor.advise(current_ai_data, previous_ai_data)
                except Exception as exc:
                    error = " ".join(str(exc).split())[:300]
                    reply = {"advisor_error": error}
                    decision = {
                        "action": "continue",
                        "exploration": previous_exploration,
                        "reason": (
                            f"AI 调用失败（{error}），沿用上次探索倾向并继续贝叶斯搜索。"
                        ),
                    }
                    record["ai_error"] = error
                else:
                    advice_error = _advice_validation_error(reply)
                    if advice_error:
                        record["ai_error"] = advice_error
                        decision = {
                            "action": "continue",
                            "exploration": previous_exploration,
                            "reason": (
                                f"AI 回复无效（{advice_error}），沿用上次探索倾向继续搜索。"
                            ),
                        }
                    else:
                        decision = validate_advice(
                            reply,
                            round_number,
                            initial_rounds,
                            allow_initial_stop=True,
                            fallback_exploration=previous_exploration,
                        )
                record["ai_called"] = True
                record["llm_reply"] = reply
                record["decision"] = decision
                _append_advice(advice_path, round_number, decision)
            else:
                decision = {
                    "action": "continue",
                    "exploration": previous_exploration,
                    "reason": "未到 AI 咨询轮次，沿用当前探索倾向。",
                }
                record["llm_reply"] = None
                record["decision"] = decision

            records.append(record)
            exploration = decision["exploration"]
            _set_exploration(optimizer, exploration)
            optimizer.save_state(run_dir / "optimizer.json")
            with (run_dir / "rounds.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            valid_records = [item for item in records if item["metrics"].get("valid", False)]
            best = min(valid_records, key=lambda item: item["metrics"]["score"]) if valid_records else None
            if best is not None:
                _write_json(run_dir / "best.json", best)
            _write_speed_response_svg(
                run_dir, records, config["target_rad_s"], config["duration_ms"],
                feedback_unit=config.get("feedback_unit", "rad/s"),
            )
            if metrics.get("valid", False):
                print(
                    f"  评分={metrics['score']:.4f}，本轮末反馈={metrics.get('final_speed_rad_s', 0):.3f}，"
                    f"当前最佳第 {best['round']} 轮"
                )
            else:
                print(f"  本轮无效：{metrics.get('fault', 'invalid_trial')}；已停止后续试验。")

            if not metrics.get("valid", False):
                manifest["status"] = (
                    "stopped" if metrics.get("fault") == "user_stop" else "trial_aborted"
                )
                manifest["fault"] = metrics.get("fault", "invalid_trial")
            elif decision["action"] in {"pause", "stop"}:
                manifest["status"] = "paused" if decision["action"] == "pause" else "stopped"
                manifest["ai_action_reason"] = decision["reason"]
            elif round_number >= total:
                manifest["status"] = "completed"
            else:
                manifest["status"] = "running"

            manifest["completed_rounds"] = len(records)
            manifest["best"] = best
            manifest["exploration"] = exploration
            if manifest["status"] == "completed" and "series_id" in manifest:
                _complete_pid_batch(run_dir, manifest, records, advisor)
            _write_json(run_dir / "manifest.json", manifest)
            if manifest["status"] != "running":
                break
            time.sleep(config["inter_trial_delay_s"])
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["completed_rounds"] = len(records)
        manifest["error"] = str(exc)[:300]
        _write_json(run_dir / "manifest.json", manifest)
        raise

    return {
        "status": manifest["status"],
        "run_dir": str(run_dir),
        "best": manifest.get("best"),
    }


def run_hardware_optimization(
    link: Any, config: dict, output_root: Path, advisor: Any = None
) -> dict:
    """Start a new physical Bayesian run; hardware trials always end with STOP."""
    advisor = (
        make_hardware_advisor(config)
        if config["run_mode"] == "bayes_llm" and advisor is None
        else advisor
    )
    if config["run_mode"] == "bayes_only":
        advisor = None
    output_root = Path(output_root)
    batch = inspect_hardware_batch(config, output_root)
    config = {**config, "search_ranges": batch["search_ranges"],
              "batch_index": batch["batch_index"]}
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "traces").mkdir()
    total = config.get("rounds_per_batch", 1 + config["initial_random_trials"] + config["bayes_trials"])
    series_id = batch.get("series_id", run_dir.name)
    manifest = {
        "status": "running",
        "mode": "physical_hardware_bayesian",
        "run_mode": config["run_mode"],
        "llm_interval_rounds": config["llm_interval_rounds"],
        "llm_call_count": 0,
        "exploration": "balanced",
        "config": config,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rounds": total,
        "pid_id": config["pid_id"],
        "series_id": series_id,
        "batch_index": batch["batch_index"],
        "history_count": len(batch["history"]),
        "warning": "Physical motor test; PC-side overspeed stop depends on live UART telemetry.",
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_advice_header(run_dir / "ai_advice.md", config)
    optimizer = _make_optimizer(config, batch["history"])
    manifest["optimizer_history_count"] = len(optimizer.space)
    _write_json(run_dir / "manifest.json", manifest)
    return _run_hardware_rounds(link, config, run_dir, manifest, optimizer, [], advisor)


def resume_hardware_optimization(link: Any, run_dir: Path, advisor: Any = None) -> dict:
    """Resume only an explicitly AI-paused run, restoring its saved optimizer state."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "paused":
        raise ValueError("only an AI-paused hardware run can be resumed")
    config = manifest["config"]
    if "series_id" not in manifest:
        # Runs created before PID-series support retain their saved experiment setup.
        config = {**config, "pid_id": "speed_pi", "history_mode": "new",
                  "rounds_per_batch": 1 + config["initial_random_trials"] + config["bayes_trials"]}
        manifest["config"] = config
        manifest["pid_id"] = config["pid_id"]
        manifest["series_id"] = run_dir.name
        manifest["batch_index"] = 1
        manifest["history_count"] = 0
        manifest["optimizer_history_count"] = 0
    advisor = (
        make_hardware_advisor(config)
        if config["run_mode"] == "bayes_llm" and advisor is None
        else advisor
    )
    if config["run_mode"] == "bayes_only":
        advisor = None
    records = _read_rounds(run_dir / "rounds.jsonl")
    if len(records) != manifest.get("completed_rounds"):
        raise ValueError("paused manifest and round log disagree")
    optimizer = _make_optimizer(config)
    optimizer.load_state(run_dir / "optimizer.json")
    if len(optimizer.space) != len(records) + manifest.get("optimizer_history_count", 0):
        raise ValueError("saved optimizer state and trial log disagree")
    return _run_hardware_rounds(link, config, run_dir, manifest, optimizer, records, advisor)
