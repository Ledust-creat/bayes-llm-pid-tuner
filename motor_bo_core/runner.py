"""Reproducible ask/evaluate/register loop, separate from upstream hardware mode."""

from __future__ import annotations

import csv
from datetime import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
from typing import Any

from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement

from .advisor import (
    ACTION_ZH,
    DEFAULT_DECISION,
    EXPLORATION_ZH,
    LiveAdvisor,
    MockAdvisor,
    ReplayAdvisor,
    XI_BY_MODE,
    validate_advice,
)
from .simulation import Gains, SimulationConfig, run_trial


UPSTREAM_COMMIT = "d885cd659e78234896ce0b851f479f8eab1ef218"
MOTOR_BO_VERSION = "0.1.0"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "motor_bo_config.json"


def make_config(tune_d: bool, path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Snapshot an entire experiment's configuration before its first round."""
    data = json.loads(path.read_text(encoding="utf-8"))
    bounds = {key: data["ranges"][key] for key in (("kp", "ki", "kd") if tune_d else ("kp", "ki"))}
    for key, pair in bounds.items():
        if (
            len(pair) != 2 or not all(math.isfinite(float(v)) for v in pair)
            or float(pair[0]) < 0 or float(pair[0]) >= float(pair[1])
        ):
            raise ValueError(f"Invalid {key} bounds")
    baseline = dict(data["baseline"])
    if not tune_d:
        baseline["kd"] = 0.0
    for key, pair in bounds.items():
        if not float(pair[0]) <= float(baseline[key]) <= float(pair[1]):
            raise ValueError(f"Baseline {key} is outside bounds")
    sim = SimulationConfig(**data["simulation"])
    if not (
        all(math.isfinite(value) for value in data["simulation"].values())
        and sim.duration_s > 0 and sim.step_rad_s > 0 and sim.integration_dt_s > 0
        and sim.speed_tick_hz > 0 and sim.inertia_kg_m2 > 0 and sim.current_tau_s > 0
        and sim.integration_dt_s <= sim.current_tau_s
        and sim.iq_limit_a > 0 and sim.speed_abort_rad_s > sim.step_rad_s
        and sim.derivative_filter_tau_s > 0 and sim.damping_nm_per_rad_s >= 0
        and sim.torque_constant_nm_per_a > 0
        and sim.duration_s / sim.integration_dt_s <= 2_000_000
    ):
        raise ValueError("Invalid simulation parameters")
    suffix = "3d" if tune_d else "2d"
    initial_random = int(data[f"initial_random_{suffix}"])
    bayes_rounds = int(data[f"bayes_rounds_{suffix}"])
    if initial_random < 1 or bayes_rounds < 0:
        raise ValueError("Invalid number of rounds")
    run_mode = data.get("run_mode", "bayes_only")
    llm_interval_rounds = int(data.get("llm_interval_rounds", 5))
    if run_mode not in {"bayes_only", "bayes_llm"}:
        raise ValueError("run_mode must be 'bayes_only' or 'bayes_llm'")
    if llm_interval_rounds < 1:
        raise ValueError("llm_interval_rounds must be at least 1")
    return {
        "run_mode": run_mode,
        "llm_interval_rounds": llm_interval_rounds,
        "seed": int(data["seed"]),
        "tune_d": tune_d,
        "baseline": baseline,
        "bounds": bounds,
        "simulation": data["simulation"],
        "initial_random": initial_random,
        "bayes_rounds": bayes_rounds,
    }


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_rounds(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _make_optimizer(config: dict) -> BayesianOptimization:
    return BayesianOptimization(
        f=None,
        pbounds={key: tuple(bounds) for key, bounds in config["bounds"].items()},
        acquisition_function=ExpectedImprovement(xi=XI_BY_MODE["balanced"]),
        random_state=config["seed"],
        verbose=0,
        allow_duplicate_points=False,
    )


def _initial_candidates(config: dict) -> list[dict[str, float]]:
    rng = random.Random(config["seed"])
    return [
        {key: rng.uniform(*bounds) for key, bounds in config["bounds"].items()}
        for _ in range(config["initial_random"])
    ]


def _make_advisor(mode: str, replay_from: str | None):
    if mode == "mock":
        return MockAdvisor()
    if mode == "live":
        return LiveAdvisor()
    if mode == "replay" and replay_from:
        return ReplayAdvisor(Path(replay_from))
    raise ValueError("Replay mode requires a source run directory")


def _query_advisor_this_round(
    mode: str, round_number: int, total_rounds: int, initial_rounds: int, interval: int
) -> bool:
    """Consult only after the initial design, at interval boundaries that affect later proposals."""
    if mode not in {"live", "replay"} or round_number >= total_rounds or round_number < initial_rounds:
        return False
    return round_number == initial_rounds or (round_number - initial_rounds) % interval == 0


def _run_mode_name(mode: str) -> str:
    return {"mock": "bayes_only", "live": "bayes_llm", "replay": "replay"}[mode]


def _write_ai_advice_header(path: Path, run_mode: str, interval: int) -> None:
    if run_mode == "bayes_only":
        text = "# AI 建议记录\n\n本次为纯贝叶斯模式，没有调用 AI。\n"
    else:
        mode_label = "AI 建议回放" if run_mode == "replay" else "贝叶斯 + AI"
        text = (
            f"# {mode_label}记录\n\n"
            f"初始样本完成后每 {interval} 轮咨询一次。\n\n"
            "| 轮次 | 动作 | 探索倾向 | AI 理由 |\n| ---: | --- | --- | --- |\n"
        )
    path.write_text(text, encoding="utf-8")


def _append_ai_advice(
    path: Path, round_number: int, decision: dict, *, print_summary: bool = True
) -> None:
    reason = " ".join(str(decision["reason"]).replace("|", "｜").split())
    action = ACTION_ZH[decision["action"]]
    exploration = EXPLORATION_ZH[decision["exploration"]]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"| {round_number:02d} | {action} | {exploration} | {reason} |\n")
    if print_summary:
        print(
            f"AI建议 第{round_number:02d}轮：动作={action}，探索倾向={exploration}，理由：{reason}"
        )


def _write_trace(path: Path, trace: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "time_s", "speed_ref_rad_s", "speed_rad_s", "iq_ref_a", "iq_a"
        ))
        writer.writeheader()
        writer.writerows(trace)


def _write_response_svg(run_dir: Path, best_trace_file: str, sim: SimulationConfig) -> None:
    """Small dependency-free view of baseline, best, and target speed."""
    def read_trace(relative_path: str) -> list[dict[str, float]]:
        with (run_dir / relative_path).open(newline="", encoding="utf-8") as handle:
            return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]

    baseline = read_trace("traces/round_001.csv")
    best = read_trace(best_trace_file)
    width, height = 800, 380
    left, right, top, bottom = 62, 24, 32, 52
    plot_width, plot_height = width - left - right, height - top - bottom
    y_max = max(sim.step_rad_s * 1.2, max(row["speed_rad_s"] for row in baseline + best))

    def points(trace: list[dict[str, float]]) -> str:
        stride = max(1, len(trace) // 600)
        selected = trace[::stride]
        if selected[-1] is not trace[-1]:
            selected.append(trace[-1])
        return " ".join(
            f"{left + row['time_s'] / sim.duration_s * plot_width:.1f},"
            f"{top + (1 - row['speed_rad_s'] / y_max) * plot_height:.1f}"
            for row in selected
        )

    target_y = top + (1 - sim.step_rad_s / y_max) * plot_height
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="white"/>
<text x="{left}" y="20" font-family="sans-serif" font-size="16">Simulated speed response — not real motor data</text>
<path d="M{left},{top} V{top + plot_height} H{left + plot_width}" fill="none" stroke="#555"/>
<line x1="{left}" y1="{target_y:.1f}" x2="{left + plot_width}" y2="{target_y:.1f}" stroke="#888" stroke-dasharray="6 4"/>
<polyline points="{points(baseline)}" fill="none" stroke="#dd7f18" stroke-width="2"/>
<polyline points="{points(best)}" fill="none" stroke="#1766b0" stroke-width="2"/>
<text x="{left}" y="{height - 16}" font-family="sans-serif" font-size="13">0 s</text>
<text x="{left + plot_width - 32}" y="{height - 16}" font-family="sans-serif" font-size="13">{sim.duration_s:g} s</text>
<text x="4" y="{top + 4}" font-family="sans-serif" font-size="13">{y_max:.1f}</text>
<text x="4" y="{top + plot_height}" font-family="sans-serif" font-size="13">0</text>
<text x="{left + 12}" y="{top + 18}" font-family="sans-serif" font-size="13" fill="#dd7f18">baseline</text>
<text x="{left + 92}" y="{top + 18}" font-family="sans-serif" font-size="13" fill="#1766b0">best</text>
<text x="{left + 140}" y="{top + 18}" font-family="sans-serif" font-size="13" fill="#777">target</text>
</svg>\n"""
    (run_dir / "response.svg").write_text(svg, encoding="utf-8")


def run_session(
    run_dir: Path,
    *,
    config: dict | None = None,
    mode: str | None = None,
    replay_from: str | None = None,
    resume: bool = False,
    advisor: Any = None,
) -> dict:
    """Run or resume one simulated search. Only files below run_dir are written."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    rounds_path = run_dir / "rounds.jsonl"
    state_path = run_dir / "optimizer.json"
    if resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "paused":
            raise ValueError("Only a paused run can be resumed")
        config = manifest["config"]
        mode = manifest.get("llm_mode", "mock")
        config.setdefault("run_mode", _run_mode_name(mode))
        config.setdefault("llm_interval_rounds", 1 if mode in {"live", "replay"} else 5)
        manifest.setdefault("run_mode", config["run_mode"])
        manifest.setdefault("llm_interval_rounds", config["llm_interval_rounds"])
        replay_from = manifest.get("replay_from")
        records = _read_rounds(rounds_path)
        optimizer = _make_optimizer(config)
        optimizer.load_state(state_path)
        if len(optimizer.space) != len(records):
            raise ValueError("Optimizer state and rounds log disagree")
    else:
        if config is None:
            config = make_config(False)
        if mode is None:
            mode = "mock" if config.get("run_mode", "bayes_only") == "bayes_only" else "live"
        elif mode == "bayes_only":
            mode = "mock"
        elif mode == "bayes_llm":
            mode = "live"
        if mode not in {"mock", "live", "replay"}:
            raise ValueError("mode must be mock, live, replay, bayes_only or bayes_llm")
        run_mode = _run_mode_name(mode)
        config = {**config, "run_mode": run_mode}
        config.setdefault("llm_interval_rounds", 5)
        if int(config["llm_interval_rounds"]) < 1:
            raise ValueError("llm_interval_rounds must be at least 1")
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "traces").mkdir()
        records = []
        optimizer = _make_optimizer(config)
        manifest = {
            "status": "running",
            "config": config,
            "run_mode": run_mode,
            "llm_mode": mode,
            "llm_interval_rounds": config["llm_interval_rounds"],
            "llm_call_count": 0,
            "replay_from": replay_from,
            "upstream_commit": UPSTREAM_COMMIT,
            "motor_bo_version": MOTOR_BO_VERSION,
            "bayesian_optimization_version": importlib.metadata.version("bayesian-optimization"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_json(manifest_path, manifest)
    run_mode = manifest.get("run_mode", _run_mode_name(mode))
    interval = int(manifest.get("llm_interval_rounds", config.get("llm_interval_rounds", 5)))
    if interval < 1:
        raise ValueError("llm_interval_rounds must be at least 1")
    if mode in {"live", "replay"}:
        advisor = advisor if advisor is not None else _make_advisor(mode, replay_from)
    else:
        advisor = None
    advice_path = run_dir / "ai_advice.md"
    if not advice_path.exists():
        _write_ai_advice_header(advice_path, run_mode, interval)
        for saved_record in records:
            if saved_record.get("ai_called"):
                _append_ai_advice(
                    advice_path, saved_record["round"], saved_record["decision"], print_summary=False
                )
    manifest["llm_call_count"] = sum(
        bool(record.get("ai_called", record.get("llm_reply") is not None)) for record in records
    )
    manifest["llm_interval_rounds"] = interval
    initial = _initial_candidates(config)
    total = 1 + config["initial_random"] + config["bayes_rounds"]
    initial_rounds = 1 + config["initial_random"]
    sim = SimulationConfig(**config["simulation"])

    for index in range(len(records), total):
        round_number = index + 1
        if index == 0:
            gains = dict(config["baseline"])
        elif index <= len(initial):
            gains = dict(initial[index - 1])
        else:
            gains = {key: float(value) for key, value in optimizer.suggest().items()}
        if not config["tune_d"]:
            gains["kd"] = 0.0
        for key, bounds in config["bounds"].items():
            if not (math.isfinite(gains[key]) and bounds[0] <= gains[key] <= bounds[1]):
                raise ValueError(f"Candidate {key} is outside the configured bounds")
        candidate = {key: gains[key] for key in config["bounds"]}
        if any(all(abs(record["gains"][key] - gains[key]) < 1e-10 for key in candidate) for record in records):
            raise ValueError("Optimizer proposed a duplicate candidate")

        result = run_trial(Gains(**gains), sim)
        metrics = result["metrics"]
        optimizer.register(params=candidate, target=-metrics["score"])
        trace_file = f"traces/round_{index + 1:03d}.csv"
        _write_trace(run_dir / trace_file, result["trace"])
        record = {"round": index + 1, "gains": gains, "metrics": metrics, "trace_file": trace_file}
        should_query = _query_advisor_this_round(mode, round_number, total, initial_rounds, interval)
        reply = None
        if should_query:
            try:
                reply = advisor.advise(record, records)
            except Exception as exc:
                reply = {"advisor_error": str(exc)[:200]}
            decision = validate_advice(reply, round_number, initial_rounds)
            manifest["llm_call_count"] += 1
            _append_ai_advice(advice_path, round_number, decision)
        elif mode == "mock":
            decision = dict(DEFAULT_DECISION)
        else:
            previous_exploration = (
                records[-1]["decision"]["exploration"] if records else "balanced"
            )
            decision = {
                "action": "continue",
                "exploration": previous_exploration,
                "reason": "未到 AI 咨询轮次，沿用当前探索倾向。",
            }
        record["ai_called"] = should_query
        record["llm_reply"] = reply
        record["decision"] = decision
        records.append(record)
        acquisition_params = optimizer.acquisition_function.get_acquisition_params()
        acquisition_params["xi"] = XI_BY_MODE[decision["exploration"]]
        optimizer.acquisition_function.set_acquisition_params(acquisition_params)
        with rounds_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        optimizer.save_state(state_path)
        best = min(records, key=lambda item: item["metrics"]["score"])
        _write_json(run_dir / "best.json", {
            "round": best["round"], "gains": best["gains"], "metrics": best["metrics"]
        })
        _write_response_svg(run_dir, best["trace_file"], sim)
        manifest["status"] = "running"
        if decision["action"] in {"pause", "stop"}:
            manifest["status"] = "paused" if decision["action"] == "pause" else "stopped"
        elif len(records) == total:
            manifest["status"] = "completed"
        _write_json(manifest_path, manifest)
        print(
            f"{index + 1:02d}/{total} P={gains['kp']:.5f} I={gains['ki']:.5f} "
            f"D={gains['kd']:.5f} score={metrics['score']:.5f} {manifest['status']}"
        )
        if manifest["status"] != "running":
            break
    return {
        "status": manifest["status"],
        "rounds": len(records),
        "best": json.loads((run_dir / "best.json").read_text(encoding="utf-8")),
        "run_dir": str(run_dir),
    }
