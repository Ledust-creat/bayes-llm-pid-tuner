"""Opt-in D support and generic metadata; no hardware or API requests."""

from collections import deque
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from motor_bo_core.gui_support import expand_curves, save_json, with_settings
from motor_bo_core.hardware_runner import (
    _make_optimizer, _validate_next_ranges, inspect_hardware_batch, load_tune_config,
    resume_hardware_optimization, run_fixed_hardware_validation, run_hardware_optimization,
)
from motor_bo_core.serial_protocol import ProtocolMessage, encode_set_pid


class GenericLink:
    def __init__(self):
        self.frames = []
        self.messages = deque()

    def clear_pending_messages(self):
        self.messages.clear()

    def set_pid(self, p, i, d, **kwargs):
        self.frames.append(encode_set_pid(p, i, d, **kwargs))

    def run_step(self, target, duration, **kwargs):
        self.messages.extend(ProtocolMessage("DATA", (str(ms), str(target), str(target)))
                             for ms in range(10, duration + 1, 10))

    def read_message(self, *args, **kwargs):
        return self.messages.popleft()

    def stop(self, **kwargs):
        return ProtocolMessage("ACK", ("OK",))


def raw_config():
    return {
        "target_rad_s": 20, "duration_ms": 100, "inter_trial_delay_s": 0,
        "max_feedback_rad_s": 50, "baseline": {"kp": 0.02, "ki": 0.5, "kd": 0},
        "ranges": {"kp": [0.01, 0.1], "ki": [0.1, 1], "kd": [0, 0.02]},
        "initial_random_trials": 2, "rounds_per_batch": 4, "seed": 42,
        "run_mode": "bayes_only", "llm_interval_rounds": 5,
        "protocol_limits": {"target_max_rad_s": 50, "duration_max_ms": 3000, "supports_kd": True},
        "pid_id": "force_pid", "history_mode": "new", "tune_d": True,
        "experiment_name": "力控制", "feedback_unit": "N",
        "fixed_repeat": {"kp": 0.02, "ki": 0.5, "kd": 0.005, "repeats": 2},
    }


class PidSupportTests(unittest.TestCase):
    def load(self, raw, root):
        path = root / "config.json"
        save_json(path, raw)
        return load_tune_config(path)

    def test_d_uses_ordinary_gain_limits_without_capability_gate(self):
        self.assertEqual(encode_set_pid(0.02, 0.5, 0.005), b"SET_PID,0.02,0.5,0.005\r\n")
        self.assertEqual(encode_set_pid(0.02, 0.5, 0.005, supports_kd=True, kd_range=(0, 0.02)),
                         b"SET_PID,0.02,0.5,0.005\r\n")
        with self.assertRaises(ValueError):
            encode_set_pid(0.02, 0.5, 0.03, supports_kd=True, kd_range=(0, 0.02))
        with tempfile.TemporaryDirectory() as tmp:
            raw = raw_config()
            raw["protocol_limits"]["supports_kd"] = False
            self.assertTrue(self.load(raw, Path(tmp))["tune_d"])

    def test_pid_search_registers_three_axes_and_records_sent_d(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            cfg = self.load(raw_config(), root)
            self.assertEqual(set(_make_optimizer(cfg).space.keys), {"kp", "ki", "kd"})
            link = GenericLink()
            result = run_hardware_optimization(link, cfg, root / "runs")
            self.assertEqual(result["status"], "completed")
            rounds = [json.loads(line) for line in (Path(result["run_dir"]) / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(link.frames), 4)
            self.assertTrue(any(row["gains"]["kd"] > 0 for row in rounds[1:]))
            for row, frame in zip(rounds, link.frames):
                self.assertAlmostEqual(float(frame.decode().strip().split(",")[3]), row["gains"]["kd"], places=7)
                self.assertTrue(0 <= row["gains"]["kd"] <= 0.02)
            svg = (Path(result["run_dir"]) / "speed_response.svg").read_text(encoding="utf-8")
            self.assertIn("反馈 (N)", svg)
            self.assertIn("Kd=", svg)
            cfg["history_mode"] = "continue"
            cfg["pid_mode"] = "pi"
            with self.assertRaisesRegex(ValueError, "搜索维度"):
                inspect_hardware_batch(cfg, root / "runs")

    def test_two_axis_search_holds_configured_d_constant(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            raw = raw_config()
            raw["tune_d"] = False
            raw["baseline"]["kd"] = 0.005
            cfg = self.load(raw, root)
            self.assertEqual(set(_make_optimizer(cfg).space.keys), {"kp", "ki"})
            link = GenericLink()
            run_hardware_optimization(link, cfg, root / "runs")
            self.assertTrue(all(frame.endswith(b",0.005\r\n") for frame in link.frames))

    def test_pd_search_and_continuation_never_change_integral(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            raw = raw_config()
            raw.update(pid_mode="pd", target_rad_s=10, target_values=[10, 20])
            cfg = self.load(raw, root)
            self.assertEqual(cfg["baseline"]["ki"], 0)
            self.assertEqual(cfg["fixed_repeat"]["ki"], 0)
            self.assertEqual(set(_make_optimizer(cfg).space.keys), {"kp", "kd"})
            link = GenericLink()
            result = run_hardware_optimization(link, cfg, root / "runs")
            self.assertEqual(result["status"], "completed")
            records = [json.loads(line) for line in (Path(result["run_dir"]) / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(record["gains"]["ki"] == 0 for record in records))
            self.assertTrue(any(record["gains"]["kd"] > 0 for record in records[1:]))
            self.assertEqual(len(link.frames), 8)
            self.assertTrue(all(float(frame.decode().split(",")[2]) == 0 for frame in link.frames))
            cfg["history_mode"] = "continue"
            run_hardware_optimization(link, cfg, root / "runs")
            self.assertEqual(len(link.frames), 16)
            self.assertTrue(all(float(frame.decode().split(",")[2]) == 0 for frame in link.frames))
            with self.assertRaisesRegex(ValueError, "搜索维度"):
                inspect_hardware_batch({**cfg, "pid_mode": "pid"}, root / "runs")

    def test_pd_fixed_repeat_overrides_configured_nonzero_i(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            raw = raw_config()
            raw.update(pid_mode="pd", operation_mode="fixed_repeat")
            cfg = self.load(raw, root)
            link = GenericLink()
            result = run_fixed_hardware_validation(link, cfg, root)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["summary"]["ki"], 0)
            self.assertEqual(link.frames, [b"SET_PID,0.02,0,0.005\r\n"] * 2)

    def test_pd_ai_pause_resume_and_range_advice_only_optimize_p_d(self):
        class Advisor:
            def __init__(self, action):
                self.action = action
                self.range_summary = None
                self.currents = []

            def advise(self, current, history):
                self.currents.append(current)
                return {"action": self.action, "exploration": "exploit", "reason": "只优化P/D"}

            def recommend_next_range(self, summary):
                self.range_summary = summary
                return {"ranges": summary["hard_ranges"], "reason": "保持P/D边界"}

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            raw = raw_config()
            raw.update(pid_mode="pd", run_mode="bayes_llm")
            cfg = self.load(raw, root)
            first_link, paused_advisor = GenericLink(), Advisor("pause")
            paused = run_hardware_optimization(first_link, cfg, root / "runs", advisor=paused_advisor)
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(paused_advisor.currents[0]["pid_mode"], "pd")
            self.assertEqual(paused_advisor.currents[0]["gains"]["ki"], 0)
            second_link, advisor = GenericLink(), Advisor("continue")
            resumed = resume_hardware_optimization(second_link, Path(paused["run_dir"]), advisor=advisor)
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(len(second_link.frames), 1)
            self.assertEqual(set(advisor.range_summary["hard_ranges"]), {"kp", "kd"})
            hard = advisor.range_summary["hard_ranges"]
            best = advisor.range_summary["historical_best"]
            with self.assertRaises(ValueError):
                _validate_next_ranges({"ranges": {**hard, "ki": [0, 1]}, "reason": "不允许改变I"}, hard, best)

    def test_pid_mode_takes_precedence_over_legacy_d_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = raw_config()
            raw["pid_mode"] = "pi"
            self.assertEqual(set(_make_optimizer(self.load(raw, root)).space.keys), {"kp", "ki"})
            raw["pid_mode"] = "invalid"
            with self.assertRaisesRegex(ValueError, "pid_mode"):
                self.load(raw, root)

    def test_fixed_repeat_sends_fixed_d_each_time(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            raw = raw_config()
            raw["operation_mode"] = "fixed_repeat"
            cfg = self.load(raw, root)
            link = GenericLink()
            result = run_fixed_hardware_validation(link, cfg, root)
            self.assertEqual(link.frames, [b"SET_PID,0.02,0.5,0.005\r\n"] * 2)
            self.assertEqual(result["summary"]["kd"], 0.005)

    def test_ai_range_advice_checks_all_three_axes(self):
        hard = raw_config()["ranges"]
        best = {"gains": {"kp": 0.02, "ki": 0.5, "kd": 0.005}}
        reply = {"ranges": hard, "reason": "保留三轴范围。"}
        self.assertEqual(set(_validate_next_ranges(reply, hard, best)), {"kp", "ki", "kd"})
        with self.assertRaises(ValueError):
            _validate_next_ranges({"ranges": {"kp": hard["kp"], "ki": hard["ki"]}, "reason": "少了一轴"}, hard, best)
        reply["ranges"] = {**hard, "kd": [0.01, 0.02]}
        with self.assertRaises(ValueError):
            _validate_next_ranges(reply, hard, best)

    def test_multi_target_same_pid_and_weighted_score(self):
        class ConditionsLink(GenericLink):
            def __init__(self):
                super().__init__()
                self.targets, self.stops = [], 0

            def run_step(self, target, duration, **kwargs):
                self.targets.append(target)
                ratio = 1 if target == 10 else 0.8
                self.messages.extend(ProtocolMessage("DATA", (str(ms), str(target), str(target * ratio)))
                                     for ms in range(10, duration + 1, 10))

            def stop(self, **kwargs):
                self.stops += 1
                return super().stop(**kwargs)

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            raw = raw_config()
            raw.update(operation_mode="fixed_repeat", target_rad_s=10,
                       target_values=[10, 20], multi_target_worst_weight=0.3)
            link = ConditionsLink()
            result = run_fixed_hardware_validation(link, self.load(raw, root), root)
            self.assertEqual(link.targets, [10, 20, 10, 20])
            self.assertEqual(link.stops, 4)
            self.assertEqual(len(set(link.frames)), 1)
            records = result["summary"]["trials"]
            self.assertEqual(len(expand_curves(records)), 4)
            for record in records:
                scores = [trial["metrics"]["score"] for trial in record["target_trials"]]
                self.assertAlmostEqual(record["metrics"]["score"], 0.7 * sum(scores) / 2 + 0.3 * max(scores))
                self.assertEqual(len(record["target_trials"]), 2)
                for trial in record["target_trials"]:
                    self.assertTrue((Path(result["run_dir"]) / trial["trace_file"]).is_file())
            svg = (Path(result["run_dir"]) / "speed_response.svg").read_text(encoding="utf-8")
            self.assertEqual(svg.count("<polyline"), 4)
            self.assertIn("目标 10", svg)
            self.assertIn("目标 20", svg)

    def test_multi_target_failure_stops_remaining_conditions(self):
        class FaultLink(GenericLink):
            def __init__(self):
                super().__init__()
                self.targets = []

            def run_step(self, target, duration, **kwargs):
                self.targets.append(target)
                feedback = 999 if target == 20 else target
                self.messages.extend(ProtocolMessage("DATA", (str(ms), str(target), str(feedback)))
                                     for ms in range(10, duration + 1, 10))

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            raw = raw_config()
            raw.update(operation_mode="fixed_repeat", target_rad_s=10, target_values=[10, 20, 30])
            link = FaultLink()
            result = run_fixed_hardware_validation(link, self.load(raw, root), root)
            self.assertEqual(link.targets, [10, 20])
            self.assertEqual(result["status"], "trial_aborted")
            record = result["summary"]["trials"][0]
            self.assertFalse(record["metrics"]["valid"])
            self.assertEqual(len(record["target_trials"]), 2)
            self.assertFalse((root / "pid_profiles").exists())

    def test_target_list_validation_and_gui_csv_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = raw_config()
            updated = with_settings(raw, {"target_values": ("10，20,30", list)})
            cfg = self.load(updated, root)
            self.assertEqual(cfg["target_values"], [10, 20, 30])
            self.assertEqual(cfg["target_rad_s"], 10)
            for targets in ([], [10, 10], [10, 99], [10, float("nan")]):
                with self.subTest(targets=targets), self.assertRaises(ValueError):
                    self.load({**updated, "target_values": targets}, root)

    def test_multi_target_bayes_history_keeps_conditions_consistent(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            raw = raw_config()
            raw.update(target_rad_s=10, target_values=[10, 20])
            cfg = self.load(raw, root)
            link = GenericLink()
            result = run_hardware_optimization(link, cfg, root / "runs")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(link.frames), 8)  # Four candidates, two conditions each.
            cfg["history_mode"] = "continue"
            self.assertEqual(len(inspect_hardware_batch(cfg, root / "runs")["history"]), 4)
            for changes in ({"target_values": [10, 25]}, {"multi_target_worst_weight": 0.5}):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "目标列表"):
                    inspect_hardware_batch({**cfg, **changes}, root / "runs")

    def test_multi_target_pause_resumes_next_candidate_not_old_conditions(self):
        class Advisor:
            def __init__(self, action):
                self.action = action
                self.currents = []

            def advise(self, current, history):
                self.currents.append(current)
                return {"action": self.action, "exploration": "balanced", "reason": "模拟建议"}

            def recommend_next_range(self, summary):
                return {"ranges": summary["hard_ranges"], "reason": "保留范围"}

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            raw = raw_config()
            raw.update(target_rad_s=10, target_values=[10, 20], run_mode="bayes_llm")
            cfg = self.load(raw, root)
            paused_advisor = Advisor("pause")
            first_link = GenericLink()
            first = run_hardware_optimization(first_link, cfg, root / "runs", advisor=paused_advisor)
            self.assertEqual(first["status"], "paused")
            self.assertEqual(len(first_link.frames), 6)
            self.assertEqual([row["target"] for row in paused_advisor.currents[0]["target_trials"]], [10, 20])
            resumed_link = GenericLink()
            result = resume_hardware_optimization(resumed_link, Path(first["run_dir"]), advisor=Advisor("continue"))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(resumed_link.frames), 2)
            lines = (Path(first["run_dir"]) / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line)["round"] for line in lines], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
