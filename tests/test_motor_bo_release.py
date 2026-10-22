"""Public examples are runnable structures, never live serial or API tests."""

import json
from pathlib import Path
import unittest

from motor_bo_core.hardware_runner import load_tune_config, score_response
from motor_bo_core.serial_link import parse_serial_config


ROOT = Path(__file__).resolve().parents[1]


class ReleaseExampleTests(unittest.TestCase):
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
