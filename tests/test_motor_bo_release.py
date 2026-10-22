"""Public examples are runnable structures, never live serial or API tests."""

import json
from pathlib import Path
import tempfile
import unittest

from motor_bo_core.gui_support import initialize_local_config, read_json
from motor_bo_core.hardware_runner import load_tune_config, score_response
from motor_bo_core.serial_link import parse_serial_config


ROOT = Path(__file__).resolve().parents[1]


class ReleaseExampleTests(unittest.TestCase):
    def test_first_run_creates_local_settings_from_public_template(self):
        template = ROOT / "motor_bo_hw_config.example.json"
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "hardware.json"
            self.assertTrue(initialize_local_config(local, template))
            self.assertEqual(read_json(local), read_json(template))

    def test_existing_settings_are_never_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "hardware.json"
            original = b'{"serial":{"port":"COM_KEEP"},"custom":true}\n'
            local.write_bytes(original)
            self.assertFalse(initialize_local_config(local, Path(tmp) / "missing-template.json"))
            self.assertEqual(local.read_bytes(), original)

    def test_invalid_template_does_not_create_local_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "hardware.json"
            template = Path(tmp) / "template.json"
            template.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                initialize_local_config(local, template)
            self.assertFalse(local.exists())

    def test_hardware_examples_validate_without_opening_serial(self):
        for filename in ("motor_bo_hw_config.example.json", "motor_bo_position_config.example.json"):
            with self.subTest(filename=filename):
                path = ROOT / filename
                raw = json.loads(path.read_text(encoding="utf-8"))
                config = load_tune_config(path)
                self.assertEqual(config["operation_mode"], "fixed_repeat")
                self.assertEqual(config["run_mode"], "bayes_only")
                self.assertEqual(parse_serial_config(raw)["port"], "YOUR_COM_PORT")

    def test_api_example_contains_no_credential(self):
        data = json.loads((ROOT / "motor_bo_local.example.json").read_text(encoding="utf-8"))
        self.assertEqual(data["api_key"], "")
        self.assertEqual(data["provider"], "openai")
        self.assertIn("example.com", data["base_url"])

    def test_position_example_uses_absolute_not_wrapped_displacement(self):
        config = load_tune_config(ROOT / "motor_bo_position_config.example.json")
        self.assertEqual(config["response"]["cyclic_period"], 0)
        self.assertFalse(config["trial_guard"]["enabled"])
        samples = [{"elapsed_ms": t, "feedback": feedback} for t, feedback in
                   ((0, -195.1), (100, 0.0), (200, 359.0))]
        result = score_response(samples, 359.0, 200, config["response"])
        self.assertAlmostEqual(result["step_amplitude"], 554.1)
        self.assertEqual(result["overshoot_ratio"], 0)

    def test_position_example_requires_a_user_selected_gain(self):
        config = load_tune_config(ROOT / "motor_bo_position_config.example.json")
        self.assertEqual(config["baseline"]["kp"], 0)
        self.assertEqual(config["fixed_repeat"]["kp"], 0)

    def test_simulation_defaults_to_offline_search(self):
        data = json.loads((ROOT / "motor_bo_config.json").read_text(encoding="utf-8"))
        self.assertEqual(data["run_mode"], "bayes_only")


if __name__ == "__main__":
    unittest.main()
