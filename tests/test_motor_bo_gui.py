"""Desktop integration checks; fake serial only, never connect a physical motor."""

from collections import deque
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import motor_bo_hw
from motor_bo_core.gui_support import current_pid_best, read_json, read_run, read_trace, save_json, with_settings
from motor_bo_core.hardware_runner import DEFAULT_TUNE_CONFIG, run_hardware_trial
from motor_bo_core.serial_protocol import ProtocolMessage


class FakeLink:
    def __init__(self, event=None, stop_error=False):
        self.commands = []
        self.messages = deque()
        self.stop_error = stop_error
        self.event = event

    def clear_pending_messages(self):
        self.messages.clear()

    def set_pid(self, *args, **kwargs):
        self.commands.append("pid")
        if self.event:
            self.event.set()

    def run_step(self, target, duration_ms, **kwargs):
        self.commands.append("run")
        for elapsed in range(10, duration_ms + 1, 10):
            self.messages.append(ProtocolMessage("DATA", (str(elapsed), str(target), str(target))))

    def read_message(self, *args, **kwargs):
        return self.messages.popleft()

    def stop(self, *args, **kwargs):
        self.commands.append("stop")
        if self.stop_error:
            raise TimeoutError("fake STOP not acknowledged")
        return ProtocolMessage("ACK", ("OK",))

    def ping(self, *args, **kwargs):
        return ProtocolMessage("ACK", ("OK",))

    def close(self):
        self.commands.append("close")


class DesktopIntegrationTests(unittest.TestCase):
    def test_gui_pid_fields_and_selective_multi_target_curves(self):
        import tkinter as tk
        import motor_bo_gui as gui
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("Tk display unavailable")
        root.withdraw()
        config = {"target_rad_s": 10, "target_values": [10, 20], "duration_ms": 100}
        records = [{"round": number, "gains": {"kp": 0.02, "ki": 0.5, "kd": 0.005},
                    "metrics": {"valid": True, "score": number * 0.1},
                    "target_trials": [{"target": target, "metrics": {"valid": True, "score": 0.1},
                                       "trace_file": f"{number}_{target}.csv"} for target in (10, 20)]}
                   for number in (1, 2)]
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.object(gui, "list_runs", return_value=[]), \
                    patch.object(gui, "HW_CONFIG", Path(tmp) / "hardware.json"), \
                    patch.object(gui.messagebox, "showerror") as showerror, \
                    patch.object(gui.messagebox, "showinfo"), \
                    patch.object(gui, "API_CONFIG", Path(tmp) / "unused.json"), \
                    patch.object(gui, "UI_CONFIG", Path(tmp) / "theme.json"):
                # Isolate test data from concurrent edits in the user's GUI.
                data = read_json(DEFAULT_TUNE_CONFIG)
                data.update(experiment_name="速度环", feedback_unit="rad/s", pid_mode="pi",
                            target_values=[20], duration_ms=3000,
                            response={"mode": "tracking"}, feedback_abs_limit=120,
                            protocol_limits={"target_min": 0, "target_max": 100,
                                             "target_min_inclusive": False, "target_max_inclusive": True,
                                             "duration_max_ms": 3000},
                            trial_guard={"enabled": True})
                data.pop("target_rad_s", None)
                save_json(gui.HW_CONFIG, data)
                app = gui.MotorTunerApp(root)
                transcript = Path(tmp) / "terminal.log"
                transcript.write_text("saved evidence\n", encoding="utf-8")
                app._append_log("old display\n")
                app.advice.configure(state="normal")
                app.advice.insert("end", "AI评价保留")
                app.advice.configure(state="disabled")
                app.events.put(("line", "new message\n"))
                with patch.object(gui, "LOG_PATH", transcript):
                    app.clear_log_button.invoke()
                    self.assertEqual(app.log.get("1.0", "end-1c"), "")
                    self.assertEqual(app.log.cget("state"), "disabled")
                    self.assertEqual(transcript.read_text(encoding="utf-8"), "saved evidence\n")
                    self.assertIn("AI评价保留", app.advice.get("1.0", "end-1c"))
                    app.poll_events()
                    self.assertIn("new message", app.log.get("1.0", "end-1c"))
                self.assertNotIn("protocol_limits.supports_kd", app.fields)
                for key in ("baseline.kd", "fixed_repeat.kd", "ranges.kd.0", "ranges.kd.1"):
                    self.assertEqual(str(app.field_widgets[key].cget("state")), "normal")
                self.assertIn("target_values", app.fields)
                previous_ki = app.fields["fixed_repeat.ki"][0].get()
                app.fields["pid_mode"][0].set("PD调参（I=0）")
                app.sync_pid_mode()
                self.assertEqual(app.fields["baseline.ki"][0].get(), "0")
                self.assertEqual(app.fields["fixed_repeat.ki"][0].get(), "0")
                self.assertEqual(str(app.field_widgets["fixed_repeat.ki"].cget("state")), "disabled")
                app.fields["pid_mode"][0].set("PI调参（D固定）")
                app.sync_pid_mode()
                self.assertEqual(app.fields["fixed_repeat.ki"][0].get(), previous_ki)
                best = {"gains": {"kp": 0.04, "ki": 0.5259, "kd": 0}, "score": 0.12}
                with patch.object(gui, "current_pid_best", return_value=(best, {**config, "pid_mode": "pi", "multi_target_worst_weight": 0.6})), \
                        patch.object(app, "prepare") as prepare:
                    app.import_best_button.invoke()
                    self.assertEqual(app.fields["fixed_repeat.kp"][0].get(), "0.04")
                    self.assertEqual(app.fields["fixed_repeat.ki"][0].get(), "0.5259")
                    self.assertEqual(app.fields["operation_mode"][0].get(), "固定参数复测")
                    self.assertEqual(app.fields["multi_target_worst_weight"][0].get(), "0.6")
                    prepare.assert_not_called()
                app.current_run = Path(tmp)
                # Map a transparent window to test real Tk geometry and hit-testing,
                # not mocked canvas sizes/row lookup that miss covered widgets.
                root.attributes("-alpha", 0.0)
                root.deiconify()
                root.update()
                with patch.object(gui, "read_run", return_value=({"status": "completed", "config": config}, records)), \
                        patch.object(gui, "read_trace", return_value=[(0, 0), (0.1, 10)]):
                    app.load_results(force=True)
                    root.update()
                    self.assertEqual(len(app.chart.find_withtag("feedback_curve")), 4)
                    self.assertEqual(app.table.master.master, app.chart.master)
                    for size in ("1380x800", "1100x650"):
                        root.geometry(size)
                        root.update()
                        self.assertTrue(app.table.winfo_ismapped())
                        self.assertTrue(app.curve_filters.winfo_ismapped())
                        self.assertLessEqual(app.table.winfo_rooty() + app.table.winfo_height(),
                                             root.winfo_rooty() + root.winfo_height())
                        self.assertLessEqual(app.curve_filters.winfo_rooty() + app.curve_filters.winfo_height(),
                                             app.chart.winfo_rooty())
                    x, y, width, height = app.table.bbox("1:1", "round")
                    app.table.event_generate("<Button-1>", x=x + 10, y=y + height // 2)
                    root.update()
                    self.assertNotIn("1:1", app.visible_curves)
                    self.assertEqual(len(app.chart.find_withtag("feedback_curve")), 3)
                    app.set_curve_visibility("best")
                    self.assertEqual(app.visible_curves, {"1:1", "1:2"})
                    self.assertEqual(len(app.chart.find_withtag("feedback_curve")), 2)
                    app.load_results(force=True)
                    self.assertEqual(len(app.visible_curves), 2)  # Refresh preserves selection.
                    app.set_curve_visibility("none")
                    self.assertEqual(len(app.chart.find_withtag("feedback_curve")), 0)
                    app.set_curve_visibility("all")
                    self.assertEqual(len(app.chart.find_withtag("feedback_curve")), 4)
                app.use_best()
                self.assertEqual(app.fields["target_values"][0].get(), "10, 20")
                self.assertEqual(app.fields["fixed_repeat.kd"][0].get(), "0.005")
                showerror.assert_not_called()
        finally:
            root.destroy()

    def test_invalid_save_preserves_original_and_valid_save_keeps_comments(self):
        data = read_json(DEFAULT_TUNE_CONFIG)
        data["extra_future_option"] = {"keep": True}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_json(path, data)
            original = path.read_bytes()
            invalid = with_settings(data, {"target_rad_s": ("999999", float)})
            with self.assertRaises(ValueError):
                save_json(path, invalid, hardware=True)
            self.assertEqual(path.read_bytes(), original)
            changed = with_settings(data, {"serial.port": ("COM_TEST", str), "ranges.kp.0": (str(data["ranges"]["kp"][0]), float)})
            save_json(path, changed, hardware=True)
            loaded = read_json(path)
            self.assertEqual(loaded["_注释"], data["_注释"])
            self.assertEqual(loaded["extra_future_option"], {"keep": True})
            self.assertEqual(loaded["serial"]["port"], "COM_TEST")

    def test_current_best_import_uses_active_pid_series_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "artifacts" / "motor_bo_hardware" / "pid_profiles" / "speed_pi"
            save_json(folder / "old.json", {"pid_id": "speed_pi", "history": [
                {"score": 0.001, "gains": {"kp": 0.5, "ki": 5}}]})
            profile = {"pid_id": "speed_pi", "pid_mode": "pi", "target_values": [20, 50, 80],
                       "duration_ms": 3000, "history": [
                           {"score": 0.2, "gains": {"kp": 0.03, "ki": 0.51}},
                           {"score": 0.12, "gains": {"kp": 0.04, "ki": 0.526}},
                           {"score": float("nan"), "gains": {"kp": 0.99, "ki": 0.99}}]}
            save_json(folder / "new.json", profile)
            save_json(folder / "active.json", {"series_id": "new"})
            original = (folder / "new.json").read_bytes()
            best, config = current_pid_best(root, "speed_pi")
            self.assertEqual(best["score"], 0.12)
            self.assertEqual(best["gains"], {"kp": 0.04, "ki": 0.526, "kd": 0})
            self.assertEqual(config["target_values"], [20, 50, 80])
            self.assertEqual((folder / "new.json").read_bytes(), original)
            with self.assertRaises(ValueError):
                current_pid_best(root, "other_pid")
            save_json(folder / "active.json", {"series_id": "../../elsewhere"})
            with self.assertRaises(ValueError):
                current_pid_best(root, "speed_pi")

    def test_cancel_during_trial_keeps_partial_samples_and_stops(self):
        event = threading.Event()
        raw = FakeLink()
        link = motor_bo_hw.DesktopControlledLink(raw, event)
        def sample(row):
            if row["elapsed_ms"] == 20:
                event.set()
        with redirect_stdout(io.StringIO()) as log:
            result = run_hardware_trial(link, 0.02, 0.5, target_rad_s=20,
                                        duration_ms=100, max_feedback_rad_s=120, on_sample=sample)
        self.assertEqual(result["metrics"]["fault"], "user_stop")
        self.assertEqual(len(result["samples"]), 2)
        self.assertEqual(raw.commands, ["pid", "run", "stop"])
        self.assertIn("停止命令已确认", log.getvalue())

    def test_cancel_between_pid_and_run_does_not_start_motor(self):
        event = threading.Event()
        raw = FakeLink(event)
        result = run_hardware_trial(motor_bo_hw.DesktopControlledLink(raw, event),
                                    0.02, 0.5, target_rad_s=20,
                                    duration_ms=100, max_feedback_rad_s=120)
        self.assertEqual(result["metrics"]["fault"], "user_stop")
        self.assertNotIn("run", raw.commands)

    def test_stop_ack_failure_is_not_reported_as_success(self):
        event = threading.Event()
        raw = FakeLink(stop_error=True)
        with redirect_stdout(io.StringIO()) as log:
            result = run_hardware_trial(motor_bo_hw.DesktopControlledLink(raw, event),
                                        0.02, 0.5, target_rad_s=20, duration_ms=100,
                                        max_feedback_rad_s=120,
                                        on_sample=lambda row: event.set())
        self.assertIn("stop_ack_timeout", result["metrics"]["fault"])
        self.assertNotIn("停止命令已确认", log.getvalue())

    def test_gui_cli_cancel_saves_run_and_issues_no_run_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = read_json(DEFAULT_TUNE_CONFIG)
            data["operation_mode"] = "fixed_repeat"
            path = root / "config.json"
            save_json(path, data)
            raw = FakeLink()
            with patch.object(motor_bo_hw, "DEFAULT_TUNE_CONFIG", path), \
                 patch.object(motor_bo_hw, "__file__", str(root / "motor_bo_hw.py")), \
                 patch.object(motor_bo_hw.MotorTuneLink, "open", return_value=raw), \
                 patch("sys.argv", ["motor_bo_hw.py", "--gui-control"]), \
                 patch("sys.stdin", io.StringIO("START\nSTOP\n")), \
                 redirect_stdout(io.StringIO()) as log:
                code = motor_bo_hw.main()
            self.assertEqual(code, 0)
            self.assertNotIn("run", raw.commands)
            self.assertIn("stop", raw.commands)
            self.assertIn("GUI_READY", log.getvalue())
            run_dir = next((root / "artifacts" / "motor_bo_validation").iterdir())
            manifest, rounds = read_run(run_dir)
            self.assertEqual(manifest["status"], "stopped")
            self.assertEqual(rounds[0]["metrics"]["fault"], "user_stop")

    def test_trace_path_cannot_escape_selected_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                read_trace(Path(tmp), {"trace_file": "../outside.csv"})


if __name__ == "__main__":
    unittest.main()
