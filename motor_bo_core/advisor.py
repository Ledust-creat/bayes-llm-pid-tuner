"""LLM may choose a bounded search strategy; it never supplies PID gains."""

from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_DECISION = {
    "action": "continue",
    "exploration": "balanced",
    "reason": "AI 建议无效，沿用默认的均衡探索策略。",
}
XI_BY_MODE = {"explore": 0.05, "balanced": 0.01, "exploit": 0.001}
ACTION_ZH = {"continue": "继续", "pause": "暂停", "stop": "停止"}
EXPLORATION_ZH = {"explore": "探索优先", "balanced": "均衡", "exploit": "利用优先"}


def validate_advice(
    reply: object,
    round_number: int,
    initial_rounds: int,
    *,
    allow_initial_stop: bool = False,
    fallback_exploration: str = "balanced",
) -> dict:
    """Discard malformed or gain-bearing replies and optionally allow an early stop."""
    if fallback_exploration not in XI_BY_MODE:
        fallback_exploration = "balanced"
    fallback = {
        "action": "continue",
        "exploration": fallback_exploration,
        "reason": DEFAULT_DECISION["reason"],
    }
    if not isinstance(reply, dict):
        return fallback
    forbidden = {"p", "i", "d", "kp", "ki", "kd", "controller_1", "controller_2"}
    if forbidden.intersection(key.lower() for key in reply if isinstance(key, str)):
        return fallback
    action = str(reply.get("action", "")).strip().lower()
    exploration = str(reply.get("exploration", "")).strip().lower()
    if action not in {"continue", "pause", "stop"} or exploration not in XI_BY_MODE:
        return fallback
    if action == "stop" and round_number < initial_rounds and not allow_initial_stop:
        action = "continue"
    return {
        "action": action,
        "exploration": exploration,
        "reason": str(reply.get("reason") or reply.get("analysis_summary") or "")[:500],
    }


class MockAdvisor:
    def advise(self, current: dict, history: list[dict]) -> dict:
        return {"action": "continue", "exploration": "balanced", "reason": "offline mock"}


class ReplayAdvisor:
    def __init__(self, source_dir: Path):
        records = [json.loads(line) for line in (source_dir / "rounds.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()]
        self.replies = {record["round"]: record["llm_reply"] for record in records}

    def advise(self, current: dict, history: list[dict]) -> object:
        return self.replies.get(current["round"])


class LiveAdvisor:
    def __init__(self):
        from llm.client import LLMTuner  # Keep upstream's provider out of offline runs.

        config_path = Path(__file__).resolve().parents[1] / "motor_bo_local.json"
        local_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        api_key = os.environ.get("MOTOR_BO_API_KEY", local_config.get("api_key", ""))
        model = os.environ.get("MOTOR_BO_MODEL", local_config.get("model", ""))
        base_url = os.environ.get("MOTOR_BO_BASE_URL", local_config.get("base_url", ""))
        provider = os.environ.get("MOTOR_BO_PROVIDER", local_config.get("provider", "openai"))
        if not (api_key and model and base_url):
            raise ValueError(
                "Live mode needs api_key, model and base_url in motor_bo_local.json "
                "or the matching MOTOR_BO_* environment variables"
            )
        self.client = LLMTuner(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=provider,
            timeout=15.0,
            emit_console=False,
        )

    def advise(self, current: dict, history: list[dict]) -> object:
        recent = [{
            "round": row["round"], "gains": row["gains"], "metrics": row["metrics"]
        } for row in history[-5:]]
        system = (
            "你负责分析电机 PID 仿真搜索结果。请始终使用简体中文，并且只返回 JSON 对象。"
            '字段为 action（continue/pause/stop）、exploration（explore/balanced/exploit）、reason（简体中文短理由）。'
            "不要输出或修改 PID 增益、搜索边界或评分权重；PID 候选由贝叶斯优化计算。"
            "只能依据提供的仿真数据，不得把仿真结果描述成真实电机实测结论。"
        )
        user = json.dumps({"current": current, "recent_history": recent}, ensure_ascii=False)
        return self.client.request_json(system_prompt=system, user_prompt=user)


class HardwareLiveAdvisor(LiveAdvisor):
    """Analyze real trial metrics and bounded samples; never provide PID gains."""

    def advise(self, current: dict, history: list[dict]) -> object:
        recent = [
            {
                "round": row["round"],
                "gains": row["gains"],
                "metrics": row["metrics"],
                "speed_points": row.get("speed_points", []),
                "target_trials": row.get("target_trials", []),
            }
            for row in history[-4:]
        ]
        system = (
            "你负责分析真实被控对象的实测 PID 调参结果。以experiment_name和feedback_unit理解工况；不得擅自假定是电机。请始终使用简体中文并只返回 JSON 对象。"
            '字段为 action（continue/pause/stop）、exploration（explore/balanced/exploit）、reason（简体中文短理由）。'
            "不得输出或修改 Kp、Ki、Kd、搜索边界或评分权重；增益候选只由贝叶斯优化器生成。"
            "pid_mode=pd时Ki固定为0，只优化Kp/Kd，不得建议增加积分项。"
            "pid_mode=p时只搜索Kp，Ki/Kd固定0。"
            "只能根据提供的遥测和指标分析，不确定时明确说明；不得忽略过冲、反转、通信故障或无效轮次。"
            "score越低越好；评分为0.70×归一化绝对误差积分+0.20×沿阶跃方向的超调率+0.10×调节时间占比。"
            "评分尺度=max(阶跃幅度,normalization_floor)，容差=max(绝对容差,幅度×相对容差)，具体依据response。"
            "位置允许负目标、零位及跨圈，不得把合法负位置或近零位置当作反转/停转。cyclic_period>0时按周期误差。"
            "initial_reference=first_sample只是指令后首帧起点估计；hold_only=true没有实际阶跃，不得误称证明动态响应。"
            "overshoot_ratio=0不能证明没有振荡或稳定，末反馈是坐标值不是速度，单位见feedback_unit。"
            "多目标时每组PID依次测试target_values全部条件，综合分=(1-w)*各目标平均分+w*最差目标分，"
            "w为multi_target_worst_weight；比较同一组固定条件，不把最后一个目标的末反馈当作全部目标的结果。"
            "pause/stop 仅在当前轮已结束并发送STOP后生效。"
        )
        current_data = {
            "round": current["round"],
            "pid_mode": current.get("pid_mode", "pi"),
            "response": current.get("response", {}),
            "target_rad_s": current["target_rad_s"],
            "target_values": current.get("target_values", [current["target_rad_s"]]),
            "multi_target_worst_weight": current.get("multi_target_worst_weight", 0.3),
            "target_trials": current.get("target_trials", []),
            "duration_ms": current["duration_ms"],
            "gains": current["gains"],
            "metrics": current["metrics"],
            "speed_points": current.get("speed_points", []),
            "experiment_name": current.get("experiment_name", "速度环"),
            "feedback_unit": current.get("feedback_unit", "rad/s"),
        }
        user = json.dumps(
            {"current": current_data, "recent_history": recent}, ensure_ascii=False
        )
        return self.client.request_json(system_prompt=system, user_prompt=user)

    def recommend_next_range(self, summary: dict) -> object:
        """Recommend only next batch's bounded search region after PWM is stopped."""
        system = (
            "你分析同一个真实被控对象 PID 的整批试验。以experiment_name和feedback_unit理解工况。只返回 JSON 对象："
            '{"ranges":{搜索轴:[下限,上限]},"reason":"简体中文理由"}。'
            "ranges的轴必须与hard_ranges完全一致：P仅kp，PI为kp/ki，PD为kp/kd，PID为kp/ki/kd。"
            "位置负值和零位是合法坐标，依据response使用带方向/周期的评分，不能当作反转/停转故障。"
            "pid_mode=pd时Ki固定为0，ranges不得包含ki，也不得建议增加积分项。"
            "这是下一批贝叶斯搜索范围，不是 PID 增益。必须在 hard_ranges 内，"
            "每轴宽度至少为对应硬范围的 20%，且包含 historical_best 对应的参数。"
            "注意历史最佳是否贴着人工硬边界；若贴边，明确指出不能越过配置边界，不把它误认为固件能力的绝对边界。"
            "结合本批误差、反转、振荡和过冲；不确定时保持 current_search_ranges。"
            "不得建议修改固件保护、评分公式、试验目标或时长。"
        )
        compact = {key: summary[key] for key in (
            "pid_id", "target_rad_s", "duration_ms", "hard_ranges",
            "current_search_ranges", "historical_best", "recent_trials",
        )}
        compact.update(experiment_name=summary.get("experiment_name", "速度环"),
                       pid_mode=summary.get("pid_mode", "pi"),
                       response=summary.get("response", {}),
                       feedback_unit=summary.get("feedback_unit", "rad/s"),
                       target_values=summary.get("target_values", [summary["target_rad_s"]]),
                       multi_target_worst_weight=summary.get("multi_target_worst_weight", 0.3),
                       score_rule=summary.get("score_rule", "分数越低越好"))
        return self.client.request_json(
            system_prompt=system,
            user_prompt=json.dumps(compact, ensure_ascii=False),
        )
