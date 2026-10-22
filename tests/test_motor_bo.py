"""Offline checks for the PC-only motor optimizer (no pytest or serial required)."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from motor_bo_core.advisor import LiveAdvisor, MockAdvisor, validate_advice
from motor_bo_core.runner import make_config, run_session
from motor_bo_core.simulation import Gains, SimulationConfig, run_trial


def small_config(tune_d: bool) -> dict:
    config = make_config(tune_d)
    config["initial_random"] = 1
    config["bayes_rounds"] = 2
    config["simulation"]["duration_s"] = 0.1
    return config


class PauseAdvisor:
    def advise(self, current: dict, history: list[dict]) -> dict:
        return {"action": "pause", "exploration": "explore", "reason": "test pause"}


class CountingAdvisor:
    def __init__(self):
        self.rounds = []

    def advise(self, current: dict, history: list[dict]) -> dict:
        self.rounds.append(current["round"])
        return {
            "action": "continue",
            "exploration": "explore",
            "reason": "前几轮数据较少，继续探索参数范围。",
        }


class MotorBoTests(unittest.TestCase):
    def test_default_model_and_score(self):
        trial = run_trial(Gains(), SimulationConfig())
        self.assertTrue(trial["metrics"]["valid"])
        self.assertGreater(trial["trace"][-1]["speed_rad_s"], 0)
        self.assertLessEqual(max(abs(row["iq_ref_a"]) for row in trial["trace"]), 1.55)
        self.assertGreaterEqual(trial["metrics"]["speed_pi_updates"], 1859)
        self.assertLessEqual(trial["metrics"]["speed_pi_updates"], 1862)
        self.assertGreater(trial["metrics"]["score"], 0)

    def test_derivative_has_no_setpoint_kick(self):
        trial = run_trial(Gains(kp=0, ki=0, kd=0.02), SimulationConfig())
        self.assertTrue(all(row["iq_ref_a"] == 0 for row in trial["trace"]))

    def test_llm_cannot_set_pid_or_stop_initialization(self):
        bad = {"action": "continue", "exploration": "balanced", "kp": 100}
        self.assertIn("默认", validate_advice(bad, 7, 5)["reason"])
        early = {"action": "stop", "exploration": "explore", "reason": "enough"}
        self.assertEqual(validate_advice(early, 4, 5)["action"], "continue")
        self.assertEqual(validate_advice(early, 5, 5)["action"], "stop")
        self.assertEqual(validate_advice(None, 6, 5)["exploration"], "balanced")

    def test_pause_resume_and_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = small_config(False)
            paused = run_session(root / "first", config=config, mode="live", advisor=PauseAdvisor())
            self.assertEqual((paused["status"], paused["rounds"]), ("paused", 2))
            finished = run_session(root / "first", resume=True, advisor=MockAdvisor())
            self.assertEqual((finished["status"], finished["rounds"]), ("completed", 4))
            replayed = run_session(root / "replay", config=config, mode="replay", replay_from=str(root / "first"))
            self.assertEqual(replayed["status"], "paused")
            replayed = run_session(root / "replay", resume=True)
            self.assertEqual(replayed["status"], "completed")
            self.assertEqual(finished["best"]["gains"], replayed["best"]["gains"])
            self.assertEqual(finished["best"]["metrics"]["score"], replayed["best"]["metrics"]["score"])
            records = [json.loads(line) for line in (root / "first" / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(row["gains"]["kd"] == 0 for row in records))

    def test_pure_bayes_does_not_create_or_call_advisor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = small_config(False)
            with patch("motor_bo_core.runner._make_advisor", side_effect=AssertionError("advisor called")):
                result = run_session(root / "pure", config=config)
            run_dir = Path(result["run_dir"])
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            records = [json.loads(line) for line in (run_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(manifest["run_mode"], "bayes_only")
            self.assertEqual(manifest["llm_call_count"], 0)
            self.assertTrue(all(not row["ai_called"] for row in records))
            self.assertIn("没有调用 AI", (run_dir / "ai_advice.md").read_text(encoding="utf-8"))

    def test_live_advisor_is_called_at_configured_rounds_and_logs_chinese(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = small_config(False)
            config["initial_random"] = 4
            config["bayes_rounds"] = 15
            config["llm_interval_rounds"] = 5
            advisor = CountingAdvisor()
            output = StringIO()
            with redirect_stdout(output):
                result = run_session(root / "hybrid", config=config, mode="live", advisor=advisor)

            run_dir = Path(result["run_dir"])
            records = [json.loads(line) for line in (run_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            advice_report = (run_dir / "ai_advice.md").read_text(encoding="utf-8")
            self.assertEqual(advisor.rounds, [5, 10, 15])
            self.assertEqual(manifest["llm_call_count"], 3)
            self.assertEqual([row["round"] for row in records if row["ai_called"]], [5, 10, 15])
            self.assertIn("AI建议 第05轮", output.getvalue())
            self.assertIn("探索优先", output.getvalue())
            self.assertIn("前几轮数据较少", advice_report)
            self.assertIn("bayes_llm", manifest["run_mode"])

    def test_live_advisor_interval_one_skips_final_round(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = small_config(False)
            config["llm_interval_rounds"] = 1
            advisor = CountingAdvisor()
            with redirect_stdout(StringIO()):
                result = run_session(
                    Path(temporary) / "hybrid", config=config, mode="live", advisor=advisor
                )
            self.assertEqual(advisor.rounds, [2, 3])
            self.assertEqual(result["rounds"], 4)

    def test_three_dimensional_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_session(Path(temporary) / "three_d", config=small_config(True))
            records = [json.loads(line) for line in (Path(result["run_dir"]) / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result["status"], "completed")
            self.assertEqual(records[0]["gains"]["kd"], 0)
            self.assertTrue(any(row["gains"]["kd"] > 0 for row in records[1:]))
            self.assertTrue(all(0 <= row["gains"]["kd"] <= 0.02 for row in records))
            self.assertNotIn("serial", sys.modules)

    def test_full_default_search_improves_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            outcome = run_session(Path(temporary) / "full", config=make_config(False))
            records = [json.loads(line) for line in (Path(outcome["run_dir"]) / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual((outcome["status"], outcome["rounds"]), ("completed", 20))
            self.assertLess(outcome["best"]["metrics"]["score"], records[0]["metrics"]["score"])
            self.assertIn("<svg", (Path(outcome["run_dir"]) / "response.svg").read_text(encoding="utf-8"))

    def test_live_adapter_uses_upstream_json_request(self):
        with patch.dict("os.environ", {
            "MOTOR_BO_API_KEY": "test-only", "MOTOR_BO_MODEL": "test-model",
            "MOTOR_BO_BASE_URL": "https://example.invalid/v1",
        }), patch("llm.client.LLMTuner") as client_type:
            client_type.return_value.request_json.return_value = {
                "action": "continue", "exploration": "exploit", "reason": "test"
            }
            reply = LiveAdvisor().advise({"round": 1, "metrics": {}}, [])
            self.assertEqual(reply["exploration"], "exploit")
            client_type.return_value.request_json.assert_called_once()
            self.assertIn("简体中文", client_type.return_value.request_json.call_args.kwargs["system_prompt"])


if __name__ == "__main__":
    unittest.main()
