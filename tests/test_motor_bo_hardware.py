from __future__ import annotations

from collections import deque
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from motor_bo_core.hardware_runner import (
    _downsample_speed_points,
    inspect_hardware_batch,
    load_tune_config,
    resume_hardware_optimization,
    run_fixed_hardware_validation,
    run_hardware_optimization,
    run_hardware_trial,
    score_speed_response,
)
from motor_bo_core.advisor import HardwareLiveAdvisor
from motor_bo_core.serial_protocol import ProtocolMessage


class FakeTuneLink:
    def __init__(self, feedback=0.0):
        self.feedback = feedback
        self.messages = deque()
        self.commands = []
        self.pid_ranges = None

    def clear_pending_messages(self):
        self.commands.append(("clear",))

    def set_pid(self, kp, ki, kd, **kwargs):
        self.pid_ranges = kwargs
        self.commands.append(("pid", kp, ki, kd))

    def run_step(self, target, duration_ms, **limits):
        self.commands.append(("run", target, duration_ms))
        for elapsed in range(10, duration_ms + 1, 10):
            self.messages.append(
                ProtocolMessage("DATA", (str(elapsed), str(target), str(self.feedback)))
            )

    def read_message(self, timeout_s=1.0):
        if not self.messages:
            raise TimeoutError
        return self.messages.popleft()

    def stop(self):
        self.commands.append(("stop",))


class PartialTelemetryLink(FakeTuneLink):
    def run_step(self, target, duration_ms, **limits):
        self.commands.append(("run", target, duration_ms))
        for elapsed in range(10, min(duration_ms // 2, 510) + 1, 10):
            self.messages.append(
                ProtocolMessage("DATA", (str(elapsed), str(target), "0.04"))
            )

    def stop(self):
        self.commands.append(("stop",))
        raise TimeoutError("no OK from firmware")


class SequenceTelemetryLink(FakeTuneLink):
    def __init__(self, sequence):
        super().__init__()
        self.sequence = sequence

    def run_step(self, target, duration_ms, **limits):
        self.commands.append(("run", target, duration_ms))
        for elapsed, reported_target, feedback in self.sequence:
            self.messages.append(ProtocolMessage(
                "DATA", (str(elapsed), str(reported_target), str(feedback))
            ))


class FakeAdvisor:
    def __init__(self, reply=None, error=None):
        self.reply = reply or {
            "action": "continue",
            "exploration": "explore",
            "reason": "继续探索实测较好的区域。",
        }
        self.error = error
        self.calls = []
        self.range_calls = []
        self.next_range_reply = None

    def advise(self, current, history):
        self.calls.append((current, history))
        if self.error:
            raise RuntimeError(self.error)
        return self.reply

    def recommend_next_range(self, summary):
        self.range_calls.append(summary)
        if self.error:
            raise RuntimeError(self.error)
        return self.next_range_reply or {
            "ranges": {"kp": [0.01, 0.2], "ki": [0.1, 1.5]},
            "reason": "围绕历史最佳收窄下一批范围。",
        }


class HardwareRunnerTests(unittest.TestCase):
    @staticmethod
    def small_config():
        return {
            "target_rad_s": 20.0, "duration_ms": 100,
            "inter_trial_delay_s": 0.0, "max_feedback_rad_s": 50.0,
            "baseline": {"kp": 0.01, "ki": 0.1},
            "ranges": {"kp": (0.01, 0.5), "ki": (0.1, 5.0)},
            "protocol_limits": {"target_max_rad_s": 50.0, "duration_max_ms": 3000},
            "initial_random_trials": 1, "bayes_trials": 1, "seed": 42,
            "run_mode": "bayes_only", "llm_interval_rounds": 5,
            "pid_id": "speed_pi", "history_mode": "new", "kd": 0.0,
        }

    def test_config_ranges_are_not_clamped_to_old_host_limits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "config.json"
            test_path.write_text(json.dumps({
                "target_rad_s": 20, "duration_ms": 3000,
                "inter_trial_delay_s": 1, "max_feedback_rad_s": 50,
                "baseline": {"kp": 0.01, "ki": 0.1},
                "ranges": {"kp": [0.01, 0.5], "ki": [0.1, 5.0]},
                "initial_random_trials": 2, "rounds_per_batch": 20,
                "seed": 42, "run_mode": "bayes_llm", "llm_interval_rounds": 5,
                "pid_id": "speed_pi", "history_mode": "new",
            }), encoding="utf-8")
            config = load_tune_config(test_path)
        self.assertEqual(config["baseline"], {"kp": 0.01, "ki": 0.1})
        self.assertEqual(config["ranges"]["kp"], (0.01, 0.5))
        self.assertEqual(config["ranges"]["ki"], (0.1, 5.0))
        self.assertEqual(config["kd"], 0.0)
        self.assertEqual(config["run_mode"], "bayes_llm")
        self.assertEqual(config["llm_interval_rounds"], 5)
        self.assertEqual(config["rounds_per_batch"], 20)
        self.assertEqual(config["protocol_limits"], {"target_max_rad_s": 50.0, "duration_max_ms": 3000, "supports_kd": False})
        self.assertEqual(config["operation_mode"], "bayes")
        self.assertEqual(
            config["fixed_repeat"],
            {"kp": 0.2517, "ki": 3.55416, "repeats": 3},
        )
        example = Path(__file__).resolve().parents[1] / "motor_bo_hw_config.example.json"
        self.assertIn("pid_id", load_tune_config(example))

    def test_fixed_repeat_config_is_selected_and_checks_firmware_bounds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "config.json"
            config = self.small_config()
            config["operation_mode"] = "fixed_repeat"
            config["fixed_repeat"] = {"kp": 0.2517, "ki": 3.55416, "repeats": 3}
            # This fixture tests its own bounds, independent of desktop edits.
            config["ranges"] = {"kp": [0.001, 0.5], "ki": [0.01, 5.0]}
            config["baseline"] = {"kp": 0.01, "ki": 0.1}
            test_path.write_text(json.dumps(config), encoding="utf-8")
            loaded = load_tune_config(test_path)
            self.assertEqual(loaded["operation_mode"], "fixed_repeat")
            self.assertEqual(loaded["fixed_repeat"]["repeats"], 3)

            config["fixed_repeat"]["kp"] = 1.0
            test_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configured firmware ranges"):
                load_tune_config(test_path)

    def test_fixed_validation_repeats_identical_gains_and_never_uses_ai_or_bayes(self):
        config = self.small_config()
        config["fixed_repeat"] = {"kp": 0.2, "ki": 0.3, "repeats": 3}
        link = FakeTuneLink(feedback=19.5)
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "motor_bo_core.hardware_runner.HardwareLiveAdvisor",
            side_effect=AssertionError("fixed validation must not create an AI client"),
        ), patch(
            "motor_bo_core.hardware_runner._make_optimizer",
            side_effect=AssertionError("fixed validation must not create a Bayesian optimizer"),
        ), redirect_stdout(io.StringIO()):
            result = run_fixed_hardware_validation(link, config, Path(temp_dir))
            run_dir = Path(result["run_dir"])
            pid_commands = [command for command in link.commands if command[0] == "pid"]
            run_commands = [command for command in link.commands if command[0] == "run"]
            self.assertEqual(pid_commands, [("pid", 0.2, 0.3, 0.0)] * 3)
            self.assertEqual(len(run_commands), 3)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["summary"]["completed_repeats"], 3)
            self.assertEqual(result["summary"]["valid_score_count"], 3)
            expected_score = score_speed_response(
                [{"elapsed_ms": t, "feedback": 19.5} for t in range(10, 101, 10)],
                20.0,
                100,
            )["score"]
            self.assertAlmostEqual(result["summary"]["mean_score"], expected_score)
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "speed_response.svg").exists())
            self.assertEqual(len(list((run_dir / "traces").glob("*.csv"))), 3)
            self.assertFalse((Path(temp_dir) / "pid_profiles").exists())

    def test_fixed_validation_aborts_remaining_runs_after_overspeed(self):
        config = self.small_config()
        config["max_feedback_rad_s"] = 20.5
        config["fixed_repeat"] = {"kp": 0.2, "ki": 0.3, "repeats": 3}
        link = FakeTuneLink(feedback=21.0)
        with tempfile.TemporaryDirectory() as temp_dir, redirect_stdout(io.StringIO()):
            result = run_fixed_hardware_validation(link, config, Path(temp_dir))

        self.assertEqual(result["status"], "trial_aborted")
        self.assertEqual(result["summary"]["completed_repeats"], 1)
        self.assertEqual(result["summary"]["valid_score_count"], 0)
        self.assertEqual(len([command for command in link.commands if command[0] == "run"]), 1)

    def test_score_rewards_response_closer_to_target(self):
        slow = [
            {"elapsed_ms": t, "feedback": 0.0}
            for t in range(10, 101, 10)
        ]
        tracking = [
            {"elapsed_ms": t, "feedback": 1.0}
            for t in range(10, 101, 10)
        ]
        self.assertLess(
            score_speed_response(tracking, 1.0, 100)["score"],
            score_speed_response(slow, 1.0, 100)["score"],
        )

    def test_trial_applies_pid_runs_step_collects_and_stops(self):
        link = FakeTuneLink(feedback=0.8)
        result = run_hardware_trial(
            link,
            0.02,
            0.2,
            target_rad_s=1.0,
            duration_ms=100,
            max_feedback_rad_s=3.0,
        )
        self.assertTrue(result["metrics"]["valid"])
        self.assertEqual(len(result["samples"]), 10)
        self.assertEqual(link.commands[1], ("pid", 0.02, 0.2, 0.0))
        self.assertEqual(link.commands[-1], ("stop",))

    def test_trial_stops_and_marks_host_overspeed(self):
        link = FakeTuneLink(feedback=3.1)
        result = run_hardware_trial(
            link,
            0.02,
            0.2,
            target_rad_s=1.0,
            duration_ms=100,
            max_feedback_rad_s=3.0,
        )
        self.assertEqual(result["metrics"]["fault"], "host_overspeed_stop")
        self.assertEqual(result["metrics"]["final_speed_rad_s"], 3.1)
        self.assertEqual(len(result["samples"]), 1)
        self.assertEqual(link.commands[-1], ("stop",))

    def test_partial_telemetry_and_stop_failure_are_both_preserved(self):
        link = PartialTelemetryLink()
        result = run_hardware_trial(
            link,
            0.01,
            0.1,
            target_rad_s=1.0,
            duration_ms=100,
            max_feedback_rad_s=3.0,
            telemetry_timeout_s=0.01,
        )
        self.assertEqual(result["metrics"]["fault"], "telemetry_incomplete; stop_ack_timeout")
        self.assertGreater(len(result["samples"]), 0)
        self.assertIn("no OK", result["metrics"]["stop_ack_error"])

    def test_sustained_reverse_stops_early_but_single_negative_sample_does_not(self):
        guard = {"startup_grace_ms": 100, "reverse_hold_ms": 100, "stall_hold_ms": 500}
        reverse = SequenceTelemetryLink([
            (elapsed, 20, -8.0) for elapsed in range(10, 301, 10)
        ])
        result = run_hardware_trial(reverse, 0.01, 0.1, target_rad_s=20,
                                    duration_ms=300, max_feedback_rad_s=50,
                                    trial_guard=guard)
        self.assertEqual(result["metrics"]["fault"], "sustained_reverse_stop")
        self.assertLess(result["samples"][-1]["elapsed_ms"], 300)
        self.assertEqual(reverse.commands[-1], ("stop",))

        jitter = SequenceTelemetryLink([
            (elapsed, 20, -2.0 if elapsed == 150 else 20.0)
            for elapsed in range(10, 301, 10)
        ])
        result = run_hardware_trial(jitter, 0.01, 0.1, target_rad_s=20,
                                    duration_ms=300, max_feedback_rad_s=50,
                                    trial_guard=guard)
        self.assertTrue(result["metrics"]["valid"])

    def test_sustained_stall_stops_and_rejects_telemetry_from_other_trial(self):
        guard = {"startup_grace_ms": 100, "reverse_hold_ms": 100, "stall_hold_ms": 500}
        stalled = SequenceTelemetryLink([
            (elapsed, 20, 0.0) for elapsed in range(10, 1001, 10)
        ])
        result = run_hardware_trial(stalled, 0.01, 0.1, target_rad_s=20,
                                    duration_ms=1000, max_feedback_rad_s=50,
                                    trial_guard=guard)
        self.assertEqual(result["metrics"]["fault"], "sustained_stall_stop")
        self.assertLess(result["samples"][-1]["elapsed_ms"], 1000)
        for sequence, fault in (
            ([(10, 19, 20)], "telemetry_target_mismatch"),
            ([(10, 20, 20), (10, 20, 20)], "telemetry_time_not_increasing"),
        ):
            with self.subTest(fault=fault):
                link = SequenceTelemetryLink(sequence)
                result = run_hardware_trial(link, 0.01, 0.1, target_rad_s=20,
                                            duration_ms=1000, max_feedback_rad_s=50,
                                            trial_guard=guard)
                self.assertEqual(result["metrics"]["fault"], fault)
                self.assertEqual(link.commands[-1], ("stop",))

    def test_invalid_first_trial_is_not_registered_or_selected_as_best(self):
        config = self.small_config()
        config["duration_ms"] = 300
        config["trial_guard"] = {"startup_grace_ms": 100, "reverse_hold_ms": 100,
                                 "stall_hold_ms": 500}
        link = SequenceTelemetryLink([(elapsed, 20, -8.0)
                                      for elapsed in range(10, 301, 10)])
        with tempfile.TemporaryDirectory() as temp_dir, redirect_stdout(io.StringIO()):
            result = run_hardware_optimization(link, config, Path(temp_dir))
            run_dir = Path(result["run_dir"])
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "trial_aborted")
            self.assertIsNone(result["best"])
            self.assertIsNone(manifest["best"])
            self.assertFalse((run_dir / "best.json").exists())
            self.assertFalse((Path(temp_dir) / "pid_profiles").exists())

    def test_hardware_ai_prompt_has_target_duration_and_score_direction(self):
        advisor = HardwareLiveAdvisor.__new__(HardwareLiveAdvisor)
        captured = {}

        class Client:
            def request_json(self, **kwargs):
                captured.update(kwargs)
                return {}

        advisor.client = Client()
        advisor.advise({"round": 1, "target_rad_s": 20.0, "duration_ms": 3000,
                        "gains": {}, "metrics": {}, "speed_points": []}, [])
        self.assertIn("越低越好", captured["system_prompt"])
        self.assertIn('"target_rad_s": 20.0', captured["user_prompt"])
        self.assertIn('"duration_ms": 3000', captured["user_prompt"])

    def test_short_optimization_writes_real_trial_records(self):
        config = self.small_config()
        config.update({"initial_random_trials": 1, "bayes_trials": 1, "duration_ms": 100, "inter_trial_delay_s": 0.0})
        link = FakeTuneLink(feedback=0.7)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_hardware_optimization(link, config, Path(temp_dir))
            run_dir = Path(result["run_dir"])
            records = [json.loads(line) for line in (run_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(records), 3)
            self.assertEqual(
                link.pid_ranges,
                {"kp_range": config["ranges"]["kp"], "ki_range": config["ranges"]["ki"]},
            )
            self.assertTrue((run_dir / "best.json").exists())
            self.assertTrue((run_dir / "traces" / "round_001.csv").exists())
            self.assertTrue((run_dir / "speed_response.svg").exists())
            svg = (run_dir / "speed_response.svg").read_text(encoding="utf-8")
            colors = re.findall(r'<polyline .*?stroke="(hsl\([^\"]+\))"', svg)
            self.assertEqual(len(colors), 3)
            self.assertEqual(len(set(colors)), 3)
            self.assertIn("第03轮", svg)

    def test_bayes_only_never_constructs_live_advisor(self):
        config = self.small_config()
        config.update({"initial_random_trials": 1, "bayes_trials": 0, "duration_ms": 100})
        link = FakeTuneLink(feedback=0.5)
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "motor_bo_core.hardware_runner.HardwareLiveAdvisor",
            side_effect=AssertionError("pure Bayes must not create an API client"),
        ), redirect_stdout(io.StringIO()):
            result = run_hardware_optimization(link, config, Path(temp_dir))
        self.assertEqual(result["status"], "completed")

    def test_llm_cadence_and_exploration_are_saved(self):
        config = self.small_config()
        config.update({
            "run_mode": "bayes_llm",
            "llm_interval_rounds": 2,
            "initial_random_trials": 2,
            "bayes_trials": 4,
            "duration_ms": 100,
            "inter_trial_delay_s": 0.0,
        })
        advisor = FakeAdvisor()
        advisor.next_range_reply = {
            "ranges": {"kp": [0.01, 0.2], "ki": [0.1, 1.5]},
            "reason": "围绕历史最佳收窄下一批范围。",
            "analysis_summary": "No analysis summary provided.",
            "thought_process": "No analysis summary provided.",
            "tuning_action": "ADJUST_PID",
        }
        with tempfile.TemporaryDirectory() as temp_dir, redirect_stdout(io.StringIO()):
            result = run_hardware_optimization(
                FakeTuneLink(feedback=0.8), config, Path(temp_dir), advisor=advisor
            )
            run_dir = Path(result["run_dir"])
            records = [json.loads(line) for line in (run_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            advice = (run_dir / "ai_advice.md").read_text(encoding="utf-8")
            optimizer_state = (run_dir / "optimizer.json").read_text(encoding="utf-8")
        self.assertEqual([row["round"] for row, _ in advisor.calls], [3, 5])
        self.assertEqual([row["round"] for row in records if row["ai_called"]], [3, 5])
        self.assertEqual(manifest["llm_call_count"], 3)
        self.assertEqual(records[2]["decision"]["exploration"], "explore")
        self.assertIn("探索实测", advice)
        self.assertIn("0.05", optimizer_state)

    def test_ai_failure_uses_previous_exploration_and_continues(self):
        config = self.small_config()
        config.update({
            "run_mode": "bayes_llm",
            "initial_random_trials": 1,
            "bayes_trials": 1,
            "duration_ms": 100,
            "inter_trial_delay_s": 0.0,
        })
        advisor = FakeAdvisor(error="offline")
        with tempfile.TemporaryDirectory() as temp_dir, redirect_stdout(io.StringIO()):
            result = run_hardware_optimization(
                FakeTuneLink(feedback=0.8), config, Path(temp_dir), advisor=advisor
            )
            records = [json.loads(line) for line in (Path(result["run_dir"]) / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(records[1]["decision"]["exploration"], "balanced")
        self.assertIn("advisor_error", records[1]["llm_reply"])
        self.assertEqual(records[1]["ai_error"], "offline")

    def test_ai_cannot_return_gains_or_change_exploration_on_invalid_reply(self):
        config = self.small_config()
        config.update({
            "run_mode": "bayes_llm",
            "initial_random_trials": 1,
            "bayes_trials": 1,
            "duration_ms": 100,
            "inter_trial_delay_s": 0.0,
        })
        advisor = FakeAdvisor({
            "action": "continue",
            "exploration": "exploit",
            "reason": "use these gains",
            "gains": {"kp": 1000.0, "ki": 1000.0},
        })
        with tempfile.TemporaryDirectory() as temp_dir, redirect_stdout(io.StringIO()):
            result = run_hardware_optimization(
                FakeTuneLink(feedback=0.8), config, Path(temp_dir), advisor=advisor
            )
            records = [json.loads(line) for line in (Path(result["run_dir"]) / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records[1]["decision"]["exploration"], "balanced")
        self.assertIn("PID 参数字段", records[1]["ai_error"])

    def test_pause_then_resume_starts_at_next_trial(self):
        config = self.small_config()
        config.update({
            "run_mode": "bayes_llm",
            "initial_random_trials": 1,
            "bayes_trials": 2,
            "duration_ms": 100,
            "inter_trial_delay_s": 0.0,
        })
        pause_advisor = FakeAdvisor({
            "action": "pause", "exploration": "balanced", "reason": "暂停检查台架。"
        })
        link = FakeTuneLink(feedback=0.7)
        with tempfile.TemporaryDirectory() as temp_dir, redirect_stdout(io.StringIO()):
            paused = run_hardware_optimization(
                link, config, Path(temp_dir), advisor=pause_advisor
            )
            run_dir = Path(paused["run_dir"])
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(sum(command[0] == "run" for command in link.commands), 2)
            completed = resume_hardware_optimization(
                link, run_dir, advisor=FakeAdvisor()
            )
            rows = [json.loads(line) for line in (run_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual([row["round"] for row in rows], [1, 2, 3, 4])
        self.assertEqual(sum(command[0] == "run" for command in link.commands), 4)

    def test_partial_aborted_trace_is_in_svg(self):
        config = self.small_config()
        config.update({"initial_random_trials": 1, "bayes_trials": 1, "duration_ms": 100})
        with tempfile.TemporaryDirectory() as temp_dir, redirect_stdout(io.StringIO()):
            result = run_hardware_optimization(
                PartialTelemetryLink(), config, Path(temp_dir)
            )
            svg = (Path(result["run_dir"]) / "speed_response.svg").read_text(encoding="utf-8")
        self.assertEqual(result["status"], "trial_aborted")
        self.assertIn("<polyline", svg)
        self.assertIn("第01轮", svg)

    def test_ai_stop_finishes_without_starting_another_trial(self):
        config = self.small_config()
        config.update({
            "run_mode": "bayes_llm",
            "initial_random_trials": 2,
            "bayes_trials": 4,
            "duration_ms": 100,
            "inter_trial_delay_s": 0.0,
        })
        advisor = FakeAdvisor({
            "action": "stop", "exploration": "balanced", "reason": "停止继续试验。"
        })
        link = FakeTuneLink(feedback=0.8)
        with tempfile.TemporaryDirectory() as temp_dir, redirect_stdout(io.StringIO()):
            result = run_hardware_optimization(
                link, config, Path(temp_dir), advisor=advisor
            )
            with self.assertRaisesRegex(ValueError, "paused"):
                resume_hardware_optimization(link, Path(result["run_dir"]), advisor=advisor)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(sum(command[0] == "run" for command in link.commands), 3)

    def test_advisor_trace_is_uniformly_downsampled_to_40_points(self):
        samples = [
            {"elapsed_ms": index * 10, "feedback": float(index)}
            for index in range(100)
        ]
        points = _downsample_speed_points(samples)
        self.assertEqual(len(points), 40)
        self.assertEqual(points[0]["elapsed_ms"], 0)
        self.assertEqual(points[-1]["elapsed_ms"], 990)

    def test_completed_pid_history_continues_without_repeating_initial_design(self):
        config = self.small_config()
        config.update({"rounds_per_batch": 3, "duration_ms": 100, "inter_trial_delay_s": 0.0})
        with tempfile.TemporaryDirectory() as temp_dir, redirect_stdout(io.StringIO()):
            root = Path(temp_dir)
            first = run_hardware_optimization(FakeTuneLink(feedback=0.8), config, root)
            first_manifest = json.loads((Path(first["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
            continued = {**config, "history_mode": "continue"}
            batch = inspect_hardware_batch(continued, root)
            self.assertEqual(len(batch["history"]), 3)
            self.assertEqual(batch["batch_index"], 2)
            second = run_hardware_optimization(FakeTuneLink(feedback=0.8), continued, root)
            second_manifest = json.loads((Path(second["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
            second_rows = [json.loads(line) for line in (Path(second["run_dir"]) / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(second_manifest["history_count"], 3)
            self.assertEqual(second_manifest["series_id"], first_manifest["series_id"])
            self.assertNotEqual(second_rows[0]["gains"]["kp"], config["baseline"]["kp"])
            final_batch = inspect_hardware_batch(continued, root)
            self.assertEqual(len(final_batch["history"]), 6)
            self.assertEqual(len(final_batch["profile"]["batches"]), 2)
            self.assertIn("batch_best_score", final_batch["profile"]["batches"][1])

    def test_new_history_is_isolated_and_preserves_previous_series(self):
        config = self.small_config()
        config.update({"rounds_per_batch": 2, "duration_ms": 100, "inter_trial_delay_s": 0.0})
        with tempfile.TemporaryDirectory() as temp_dir, redirect_stdout(io.StringIO()):
            root = Path(temp_dir)
            first = run_hardware_optimization(FakeTuneLink(feedback=0.8), config, root)
            old_profile = Path(json.loads((Path(first["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))["profile_file"])
            second = run_hardware_optimization(FakeTuneLink(feedback=0.8), config, root)
            new_profile = Path(json.loads((Path(second["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))["profile_file"])
            self.assertNotEqual(old_profile, new_profile)
            self.assertTrue(old_profile.exists())
            self.assertEqual(len(inspect_hardware_batch({**config, "history_mode": "continue"}, root)["history"]), 2)

    def test_batch_ai_range_is_checked_then_applied_to_next_batch(self):
        config = self.small_config()
        config.update({"run_mode": "bayes_llm", "rounds_per_batch": 3,
                       "duration_ms": 100, "inter_trial_delay_s": 0.0})
        advisor = FakeAdvisor()
        with tempfile.TemporaryDirectory() as temp_dir, redirect_stdout(io.StringIO()):
            root = Path(temp_dir)
            first = run_hardware_optimization(FakeTuneLink(feedback=0.8), config, root, advisor)
            self.assertEqual(len(advisor.range_calls), 1)
            continued = {**config, "history_mode": "continue"}
            batch = inspect_hardware_batch(continued, root)
            self.assertEqual(batch["search_ranges"], {"kp": (0.01, 0.2), "ki": (0.1, 1.5)})
            second = run_hardware_optimization(FakeTuneLink(feedback=0.8), continued, root, advisor)
            rows = [json.loads(line) for line in (Path(second["run_dir"]) / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(0.01 <= row["gains"]["kp"] <= 0.2 and
                                0.1 <= row["gains"]["ki"] <= 1.5 for row in rows))
            self.assertTrue(Path(first["run_dir"]).joinpath("ai_advice.md").exists())

    def test_invalid_ai_range_falls_back_without_expanding_hard_limit(self):
        config = self.small_config()
        config.update({"run_mode": "bayes_llm", "rounds_per_batch": 2,
                       "duration_ms": 100, "inter_trial_delay_s": 0.0})
        advisor = FakeAdvisor()
        advisor.next_range_reply = {"ranges": {"kp": [0, 100], "ki": [0.1, 5.0]}, "reason": "unsafe"}
        with tempfile.TemporaryDirectory() as temp_dir, redirect_stdout(io.StringIO()):
            root = Path(temp_dir)
            result = run_hardware_optimization(FakeTuneLink(feedback=0.8), config, root, advisor)
            manifest = json.loads((Path(result["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["next_search_ranges"]["kp"], list(config["ranges"]["kp"]))
            self.assertIn("hard limits", manifest["range_advice_error"])

    def test_changed_trial_target_cannot_reuse_pid_history(self):
        config = self.small_config()
        config.update({"rounds_per_batch": 2, "duration_ms": 100, "inter_trial_delay_s": 0.0})
        with tempfile.TemporaryDirectory() as temp_dir, redirect_stdout(io.StringIO()):
            root = Path(temp_dir)
            run_hardware_optimization(FakeTuneLink(feedback=0.8), config, root)
            with self.assertRaisesRegex(ValueError, "history_mode"):
                inspect_hardware_batch({**config, "history_mode": "continue", "target_rad_s": 10.0}, root)

    def test_legacy_paused_run_can_finish_and_seed_new_series(self):
        config = self.small_config()
        config.update({"initial_random_trials": 1, "bayes_trials": 1,
                       "duration_ms": 100, "inter_trial_delay_s": 0.0,
                       "run_mode": "bayes_llm"})
        pause_advisor = FakeAdvisor({"action": "pause", "exploration": "balanced", "reason": "检查。"})
        with tempfile.TemporaryDirectory() as temp_dir, redirect_stdout(io.StringIO()):
            root = Path(temp_dir)
            paused = run_hardware_optimization(FakeTuneLink(feedback=0.8), config, root, pause_advisor)
            path = Path(paused["run_dir"]) / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            for key in ("series_id", "pid_id", "history_count", "optimizer_history_count", "batch_index"):
                manifest.pop(key, None)
            for key in ("pid_id", "history_mode", "rounds_per_batch", "search_ranges", "batch_index"):
                manifest["config"].pop(key, None)
            path.write_text(json.dumps(manifest), encoding="utf-8")
            finished = resume_hardware_optimization(FakeTuneLink(feedback=0.8), path.parent, FakeAdvisor())
            self.assertEqual(finished["status"], "completed")
            self.assertEqual(len(inspect_hardware_batch({**config, "history_mode": "continue"}, root)["history"]), 3)


if __name__ == "__main__":
    unittest.main()
