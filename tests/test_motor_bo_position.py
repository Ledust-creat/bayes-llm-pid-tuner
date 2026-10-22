"""Position/other signed coordinates: offline regression, no hardware or API."""

from collections import deque
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from motor_bo_core.gui_support import read_json, save_json
from motor_bo_core.hardware_runner import (
    _read_trace, inspect_hardware_batch, load_tune_config, run_fixed_hardware_validation,
    run_hardware_optimization, run_hardware_trial, score_response,
)
from motor_bo_core.serial_link import MotorTuneLink
from motor_bo_core.serial_protocol import ProtocolMessage, encode_run_step, encode_set_pid


LIMITS = dict(target_min=-360, target_max=360, target_min_inclusive=True,
              target_max_inclusive=False, duration_max_ms=5000)
RESPONSE = dict(mode="position", initial_reference="first_sample", initial_value=0,
                normalization_floor=1, settling_tolerance_abs=1,
                settling_tolerance_ratio=0.05, cyclic_period=720)


def position_config():
    return dict(target_values=[10, 0, -10], duration_ms=100, inter_trial_delay_s=0,
                feedback_abs_limit=360, response=RESPONSE, trial_guard={"enabled": False},
                baseline=dict(kp=0.02, ki=0.1, kd=0),
                ranges=dict(kp=[0.001, 1], ki=[0.01, 1], kd=[0, 0.02]),
                fixed_repeat=dict(kp=0.02, ki=0.1, kd=0, repeats=2),
                protocol_limits=LIMITS, initial_random_trials=1, rounds_per_batch=3,
                seed=42, run_mode="bayes_only", pid_mode="pi", operation_mode="fixed_repeat",
                llm_interval_rounds=5, pid_id="position_pi", history_mode="new",
                experiment_name="位置环", feedback_unit="度",
                serial=dict(port="COM_FAKE", baudrate=115200, read_timeout_s=0.05,
                            ping_timeout_s=1, listen_after_ping_s=0))


class FakePositionLink:
    def __init__(self, feedback=None):
        self.frames, self.targets, self.messages = [], [], deque()
        self.stops = 0
        self.feedback = feedback

    def clear_pending_messages(self):
        self.messages.clear()

    def set_pid(self, kp, ki, kd, **options):
        self.frames.append(encode_set_pid(kp, ki, kd, **options))

    def run_step(self, target, duration, **options):
        self.frames.append(encode_run_step(target, duration, **options))
        self.targets.append(target)
        for ms in range(10, duration + 1, 10):
            value = target if self.feedback is None else self.feedback
            self.messages.append(ProtocolMessage("DATA", (str(ms), str(target), str(value))))

    def read_message(self, *args, **kwargs):
        return self.messages.popleft()

    def stop(self):
        self.stops += 1
        return ProtocolMessage("ACK", ("OK",))


class FakeSerial:
    def __init__(self):
        self.lines, self.writes = deque(), []

    def write(self, data):
        self.writes.append(data)
        self.lines.append(b"OK\r\n")
        return len(data)

    def flush(self):
        pass

    def readline(self):
        return self.lines.popleft() if self.lines else b""


class PositionSupportTests(unittest.TestCase):
    def load(self, raw, root):
        path = root / "settings.json"
        save_json(path, raw)
        return load_tune_config(path)

    def samples(self, target, values):
        return [dict(elapsed_ms=10 * (index + 1), target=target, feedback=value)
                for index, value in enumerate(values)]

    def test_signed_zero_and_exclusive_endpoints_are_encoded(self):
        port = FakeSerial()
        link = MotorTuneLink(port)
        for target in (-360, -90, 0, 90, 359.99):
            link.run_step(target, 5000, **LIMITS)
            self.assertEqual(port.writes[-1], f"RUN_STEP,{target:g},5000\r\n".encode())
        count = len(port.writes)
        for target in (-360.1, 360, 359.99999, float("nan"), float("inf")):
            with self.subTest(target=target), self.assertRaises(ValueError):
                link.run_step(target, 100, **LIMITS)
        self.assertEqual(len(port.writes), count)

    def test_invalid_step_is_rejected_before_pid_write(self):
        link = FakePositionLink()
        with self.assertRaises(ValueError):
            run_hardware_trial(link, 0.02, 0.1, target_rad_s=360, duration_ms=100,
                               max_feedback_rad_s=360, target_min=-360,
                               target_max_rad_s=360, target_min_inclusive=True,
                               target_max_inclusive=False, response=RESPONSE)
        self.assertEqual(link.frames, [])
        self.assertEqual(link.stops, 0)

    def test_position_advisor_receives_scoring_geometry_and_only_p_range(self):
        from motor_bo_core.advisor import HardwareLiveAdvisor
        from unittest.mock import Mock
        client = Mock()
        client.request_json.return_value = dict(action="continue", exploration="balanced", reason="继续")
        advisor = HardwareLiveAdvisor.__new__(HardwareLiveAdvisor)
        advisor.client = client
        advisor.advise(dict(round=1, gains=dict(kp=0.02, ki=0, kd=0),
                            metrics=dict(score=0.1, valid=True), response=RESPONSE,
                            pid_mode="p", target_rad_s=-10, target_values=[-10, 0],
                            duration_ms=100, feedback_unit="度"), [])
        payload = json.loads(client.request_json.call_args.kwargs["user_prompt"])["current"]
        self.assertEqual(payload["response"]["cyclic_period"], 720)
        self.assertEqual(payload["feedback_unit"], "度")
        self.assertIn("pid_mode=p", client.request_json.call_args.kwargs["system_prompt"])
        advisor.recommend_next_range(dict(pid_mode="p", pid_id="position_p", target_rad_s=-10, response=RESPONSE,
            feedback_unit="度", target_values=[-10, 0], duration_ms=100,
            hard_ranges={"kp": [0.001, 1]}, current_search_ranges={"kp": [0.001, 1]},
            historical_best=dict(gains=dict(kp=0.02, ki=0, kd=0), score=0.1), recent_trials=[]))
        payload = json.loads(client.request_json.call_args.kwargs["user_prompt"])
        self.assertEqual(payload["response"]["mode"], "position")
        self.assertEqual(set(payload["hard_ranges"]), {"kp"})

    def test_bad_bounds_are_rejected(self):
        for changes in (dict(target_min=360), dict(target_max=float("nan")),
                        dict(target_min_inclusive="true"), dict(target_max_inclusive=1)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                encode_run_step(0, 100, **{**LIMITS, **changes})

    def test_position_config_accepts_signed_targets_but_tracking_guard_rejects_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, raw = Path(tmp), position_config()
            config = self.load(raw, root)
            self.assertEqual(config["target_values"], [10, 0, -10])
            self.assertFalse(config["trial_guard"]["enabled"])
            for changes in ({"target_values": [360]}, {"trial_guard": {"enabled": True}},
                            {"response": {"normalization_floor": 0}}, {"response": []}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    self.load({**raw, **changes}, root)

    def test_mirrored_steps_have_identical_scores(self):
        settings = {**RESPONSE, "initial_reference": "fixed", "cyclic_period": 0}
        positive = score_response(self.samples(10, [0, 4, 9, 12, 10]), 10, 50, settings)
        negative = score_response(self.samples(-10, [0, -4, -9, -12, -10]), -10, 50, settings)
        for name in ("score", "iae_norm", "overshoot_ratio", "settling_time_s"):
            self.assertAlmostEqual(positive[name], negative[name])
        self.assertAlmostEqual(negative["overshoot_ratio"], 0.2)

    def test_zero_target_returns_to_origin_without_dividing_by_zero(self):
        metrics = score_response(self.samples(0, [10, 6, 2, -2, 0]), 0, 50, RESPONSE)
        self.assertTrue(math.isfinite(metrics["score"]))
        self.assertEqual(metrics["step_amplitude"], 10)
        self.assertAlmostEqual(metrics["overshoot_ratio"], 0.2)
        self.assertEqual(metrics["initial_reference"], "first_sample")
        self.assertEqual(metrics["settling_tolerance"], 1)

    def test_zero_size_hold_is_identified_and_has_finite_score(self):
        metrics = score_response(self.samples(0, [0, 0.1, -0.1, 0, 0]), 0, 50, RESPONSE)
        self.assertTrue(metrics["hold_only"])
        self.assertEqual(metrics["normalization_scale"], 1)
        self.assertEqual(metrics["overshoot_ratio"], 0)
        self.assertTrue(math.isfinite(metrics["score"]))

    def test_cycle_crossing_is_twenty_degrees_not_seven_hundred(self):
        metrics = score_response(self.samples(-350, [350, 355, -359, -350, -350]),
                                 -350, 50, RESPONSE)
        self.assertEqual(metrics["step_amplitude"], 20)
        self.assertEqual(metrics["overshoot_ratio"], 0)
        self.assertLess(metrics["iae_norm"], 1)
        self.assertEqual(metrics["final_error_abs"], 0)

    def test_negative_and_zero_positions_do_not_trigger_speed_guards(self):
        for target in (-10, 0):
            link = FakePositionLink()
            trial = run_hardware_trial(link, 0.02, 0.1, target_rad_s=target,
                                       duration_ms=100, max_feedback_rad_s=360,
                                       target_max_rad_s=360, target_min=-360,
                                       target_min_inclusive=True, target_max_inclusive=False,
                                       response=RESPONSE,
                                       trial_guard=dict(enabled=False, startup_grace_ms=0,
                                                        reverse_hold_ms=1, stall_hold_ms=1))
            self.assertTrue(trial["metrics"]["valid"])
            self.assertEqual(link.stops, 1)
        link = FakePositionLink(feedback=-361)
        trial = run_hardware_trial(link, 0.02, 0.1, target_rad_s=-10, duration_ms=100,
                                   max_feedback_rad_s=360, target_min=-360,
                                   target_max_rad_s=360, target_min_inclusive=True,
                                   response=RESPONSE, trial_guard={"enabled": False})
        self.assertFalse(trial["metrics"]["valid"])
        self.assertEqual(trial["metrics"]["fault"], "host_overspeed_stop")
        self.assertEqual(link.stops, 1)

    def test_multi_target_csv_svg_and_score_use_position_units(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root, link = Path(tmp), FakePositionLink()
            result = run_fixed_hardware_validation(link, self.load(position_config(), root), root)
            self.assertEqual(link.targets, [10, 0, -10] * 2)
            self.assertEqual(link.stops, 6)
            folder = Path(result["run_dir"])
            for record in result["summary"]["trials"]:
                self.assertTrue(record["metrics"]["valid"])
                for trial in record["target_trials"]:
                    path = folder / trial["trace_file"]
                    self.assertEqual(path.read_text().splitlines()[0], "elapsed_ms,target,feedback")
                    self.assertEqual(_read_trace(path)[0]["target_rad_s"], trial["target"])
            svg = (folder / "speed_response.svg").read_text(encoding="utf-8")
            self.assertEqual(svg.count("<polyline"), 6)
            self.assertIn("目标 -10", svg)
            self.assertIn("度", svg)

    def test_p_mode_searches_only_kp_and_keeps_i_d_zero_across_batches(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root, raw, link = Path(tmp), position_config(), FakePositionLink()
            raw.update(pid_mode="p", operation_mode="bayes", target_values=[10])
            config = self.load(raw, root)
            self.assertEqual(config["baseline"]["ki"], 0)
            first = run_hardware_optimization(link, config, root)
            raw["history_mode"] = "continue"
            second = run_hardware_optimization(link, self.load(raw, root), root)
            for result in (first, second):
                rows = (Path(result["run_dir"]) / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
                for row in map(json.loads, rows):
                    self.assertEqual(row["gains"]["ki"], 0)
                    self.assertEqual(row["gains"]["kd"], 0)
            for frame in link.frames:
                if frame.startswith(b"SET_PID"):
                    self.assertTrue(frame.endswith(b",0,0\r\n"))

    def test_history_rejects_changed_scoring_or_units(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root, raw = Path(tmp), position_config()
            raw.update(operation_mode="bayes", target_values=[10])
            run_hardware_optimization(FakePositionLink(), self.load(raw, root), root)
            raw["history_mode"] = "continue"
            for changes in (dict(feedback_unit="rad"),
                            dict(response={**RESPONSE, "cyclic_period": 360}),
                            dict(response={**RESPONSE, "settling_tolerance_abs": 2}),
                            dict(trial_guard={"enabled": True})):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    inspect_hardware_batch(self.load({**raw, **changes}, root), root)

    def test_gui_saves_signed_position_fields_and_rejects_speed_history_import(self):
        import tkinter as tk
        import motor_bo_gui as gui
        try:
            window = tk.Tk()
        except tk.TclError:
            self.skipTest("Tk display unavailable")
        window.withdraw()
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.object(gui, "HW_CONFIG", Path(tmp) / "config.json"), \
                    patch.object(gui, "API_CONFIG", Path(tmp) / "unused.json"), \
                    patch.object(gui, "UI_CONFIG", Path(tmp) / "ui.json"), \
                    patch.object(gui, "list_runs", return_value=[]), \
                    patch.object(gui.messagebox, "showerror") as showerror, \
                    patch.object(gui.messagebox, "showinfo"):
                save_json(gui.HW_CONFIG, position_config())
                app = gui.MotorTunerApp(window)
                self.assertEqual(app.fields["protocol_limits.target_min"][0].get(), "-360")
                self.assertEqual(app.fields["trial_guard.enabled"][0].get(), "关闭（允许负值/零位）")
                original = app.fields["fixed_repeat.kp"][0].get()
                with self.assertRaises(ValueError):
                    app._import_best(dict(kp=0.5, ki=1, kd=0),
                                     dict(feedback_unit="rad/s", response={"mode": "tracking"}), "测试历史")
                self.assertEqual(app.fields["fixed_repeat.kp"][0].get(), original)
                app.fields["pid_mode"][0].set("P调参（I=D=0）")
                app.sync_pid_mode()
                self.assertEqual(app.fields["fixed_repeat.ki"][0].get(), "0")
                self.assertTrue(app.save_hardware(), showerror.call_args)
                saved = read_json(gui.HW_CONFIG)
                self.assertEqual(saved["target_values"], [10, 0, -10])
                self.assertNotIn("target_rad_s", saved)
                self.assertEqual(saved["pid_mode"], "p")
                self.assertEqual(saved["response"]["cyclic_period"], 720)
                self.assertFalse(saved["protocol_limits"]["target_max_inclusive"])
                showerror.assert_not_called()
        finally:
            window.destroy()
