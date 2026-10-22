"""Desktop file operations. Loading or saving settings never opens a COM port."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
import tempfile


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("配置根节点必须是 JSON 对象")
    return data


def initialize_local_config(path: Path, template: Path) -> bool:
    """Create a missing local config; never replace user settings or open devices."""
    if path.exists():
        return False
    data = read_json(template)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive creation also protects settings created by another GUI.
        with path.open("x", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except FileExistsError:
        return False
    return True


def get_value(data: dict, key: str, default=""):
    value = data
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def with_settings(data: dict, settings: dict[str, tuple[str, type]]) -> dict:
    """Preserve comments and unexposed fields; reject invalid numbers before save."""
    result = json.loads(json.dumps(data))
    for key, (text, kind) in settings.items():
        if kind is list:
            try:
                value = [float(item.strip()) for item in text.replace("，", ",").split(",")]
            except ValueError as exc:
                raise ValueError(f"{key}：填写逗号分隔的目标，例如 15.5,25.5,65") from exc
            if not all(math.isfinite(item) for item in value):
                raise ValueError(f"{key}：不能填写 NaN 或 Inf")
        elif kind is bool:
            normalized = text.strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(f"{key}：必须为 true 或 false")
            value = normalized == "true"
        elif kind is str:
            value = text.strip()
        else:
            try:
                value = kind(text.strip())
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{key}：请输入{'整数' if kind is int else '数字'}") from exc
            if not math.isfinite(value):
                raise ValueError(f"{key}：不能填写 NaN 或 Inf")
        parts = key.split(".")
        target = result
        for part in parts[:-1]:
            if part.isdigit():
                target = target[int(part)]
            else:
                target = target.setdefault(part, {})
        if parts[-1].isdigit():
            target[int(parts[-1])] = value
        else:
            target[parts[-1]] = value
    if "target_values" in settings:
        result.pop("target_rad_s", None)  # New configs use the unit-neutral list.
    return result


def save_json(path: Path, data: dict, *, hardware=False) -> None:
    """Validate a staged file, then atomically replace the original JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False,
    ) as handle:
        staging = Path(handle.name)
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    try:
        if hardware:
            from .hardware_runner import load_tune_config
            from .serial_link import parse_serial_config

            load_tune_config(staging)
            parse_serial_config(data)
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def list_runs(root: Path) -> list[Path]:
    result = []
    for folder in ("motor_bo_hardware", "motor_bo_validation"):
        parent = root / "artifacts" / folder
        if parent.exists():
            result.extend(p for p in parent.iterdir() if p.is_dir() and (p / "manifest.json").exists())
    return sorted(result, key=lambda p: p.name, reverse=True)


def read_run(path: Path) -> tuple[dict, list[dict]]:
    manifest = read_json(path / "manifest.json")
    records = []
    log = path / "rounds.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # The runner may still be appending its final line.
            if isinstance(record, dict) and "metrics" in record:
                records.append(record)
    return manifest, records


def read_trace(run_dir: Path, record: dict) -> list[tuple[float, float]]:
    # Records are local files, but never follow a trace path outside the run.
    path = (run_dir / record["trace_file"]).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ValueError("曲线文件超出运行目录")
    points = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            x, y = float(row["elapsed_ms"]) / 1000.0, float(row["feedback"])
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
    return points


def expand_curves(records: list[dict], default_target: float = 0.0) -> list[dict]:
    """Flatten multi-condition records while retaining legacy single-trace runs."""
    result = []
    for record in records:
        trials = record.get("target_trials") or [{"target": default_target,
                    "metrics": record["metrics"], "trace_file": record["trace_file"]}]
        for index, trial in enumerate(trials, 1):
            result.append({**record, **trial, "curve_id": f"{record['round']}:{index}",
                           "target_index": index})
    return result


def current_pid_best(root: Path, pid_id: str) -> tuple[dict, dict]:
    """Read the active completed PID series, not the currently selected chart.

    Paths and finite scores are checked before importing local history. No
    configuration, profile or hardware state is changed by this lookup.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", pid_id):
        raise ValueError("PID历史名称格式无效")
    folder = root / "artifacts" / "motor_bo_hardware" / "pid_profiles" / pid_id
    if not (folder / "active.json").exists():
        raise ValueError(f"PID {pid_id} 尚无完成批次的历史；可导入右侧所选记录的最优")
    series_id = read_json(folder / "active.json").get("series_id", "")
    if not isinstance(series_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", series_id):
        raise ValueError("PID历史系列标识无效")
    profile = read_json(folder / f"{series_id}.json")
    if profile.get("pid_id") != pid_id:
        raise ValueError("历史文件不属于当前PID")
    valid = []
    for row in profile.get("history", []):
        if not isinstance(row, dict):
            continue
        try:
            score = float(row["score"])
            gains = {key: float(row["gains"][key]) for key in ("kp", "ki")}
            gains["kd"] = float(row["gains"].get("kd", 0))
            if (math.isfinite(score) and all(math.isfinite(v) and v >= 0 for v in gains.values())
                    and row.get("metrics", {}).get("valid", True)):
                valid.append({**row, "score": score, "gains": gains})
        except (KeyError, TypeError, ValueError):
            continue
    if not valid:
        raise ValueError("当前PID历史没有有效评分")
    best = min(valid, key=lambda row: row["score"])
    # Original run metadata includes its object name and full test conditions.
    # Do not follow profile paths outside this project's artifact directory.
    metadata = profile
    source = Path(best.get("run_dir", "")).resolve()
    if source.is_relative_to((root / "artifacts").resolve()) and (source / "manifest.json").is_file():
        metadata = read_json(source / "manifest.json").get("config", profile)
    return best, metadata
