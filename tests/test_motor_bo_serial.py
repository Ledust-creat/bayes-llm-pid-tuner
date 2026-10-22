"""Offline checks for the current plain-text PC/STM32 serial protocol."""

from __future__ import annotations

from collections import deque
import unittest

from motor_bo_core.serial_link import MotorTuneLink, parse_serial_config
from motor_bo_core.serial_protocol import (
    ProtocolMessage,
    encode_frame,
    encode_ping,
    encode_run_step,
    encode_set_pid,
    encode_stop,
    parse_frame,
    parse_telemetry,
)


class FakeSerial:
    def __init__(self, replies=(), respond_ok=True):
        self.lines = deque(replies)
        self.respond_ok = respond_ok
        self.writes = []
        self.is_open = True

    def write(self, payload):
        self.writes.append(payload)
        if self.respond_ok:
            self.lines.append(b"OK\r\n")
        return len(payload)

    def flush(self):
        pass

    def readline(self):
        return self.lines.popleft() if self.lines else b""

    def close(self):
        self.is_open = False


class MotorBoSerialTests(unittest.TestCase):
    def test_shared_serial_config_and_validation(self):
        settings = parse_serial_config({"serial": {
            "port": "COM9", "baudrate": 230400,
            "read_timeout_s": 0.1, "ping_timeout_s": 2.0,
        }})
        self.assertEqual(settings["port"], "COM9")
        self.assertEqual(settings["baudrate"], 230400)
        self.assertEqual(settings["ping_timeout_s"], 2.0)
        with self.assertRaisesRegex(ValueError, "serial.port"):
            parse_serial_config({"serial": {"port": ""}})
        with self.assertRaisesRegex(ValueError, "serial.baudrate"):
            parse_serial_config({"serial": {"port": "COM9", "baudrate": 0}})

    def test_encode_exact_command_lines(self):
        self.assertEqual(encode_ping(), b"PING\r\n")
        self.assertEqual(encode_set_pid(0.01, 0.1, 0.0), b"SET_PID,0.01,0.1,0\r\n")
        self.assertEqual(encode_run_step(5.0, 3000), b"RUN_STEP,5,3000\r\n")
        self.assertEqual(encode_stop(), b"STOP\r\n")

    def test_pid_limits_accept_inclusive_bounds(self):
        self.assertEqual(encode_set_pid(0.0, 0.0, 0.0), b"SET_PID,0,0,0\r\n")
        self.assertEqual(encode_set_pid(0.1, 1.0, 0.0), b"SET_PID,0.1,1,0\r\n")

    def test_pid_ranges_can_be_configured_above_default_protocol_limits(self):
        self.assertEqual(
            encode_set_pid(
                0.5,
                5.0,
                0.0,
                kp_range=(0.01, 0.5),
                ki_range=(0.1, 5.0),
            ),
            b"SET_PID,0.5,5,0\r\n",
        )

    def test_pid_rejects_nonfinite_or_out_of_range_values(self):
        invalid = (
            (float("nan"), 0.1, 0.0),
            (float("inf"), 0.1, 0.0),
            (-0.01, 0.1, 0.0),
            (0.01, float("inf"), 0.0),
            (0.01, -0.01, 0.0),
            (0.01, 0.1, -0.001),
            (0.01, 0.1, float("nan")),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                encode_set_pid(*values)

    def test_pid_rejects_values_outside_the_configured_range(self):
        with self.assertRaisesRegex(ValueError, "configured range"):
            encode_set_pid(0.51, 5.0, 0.0, kp_range=(0.01, 0.5), ki_range=(0.1, 5.0))
        with self.assertRaisesRegex(ValueError, "configured range"):
            encode_set_pid(0.5, 5.01, 0.0, kp_range=(0.01, 0.5), ki_range=(0.1, 5.0))

    def test_run_step_accepts_documented_bounds(self):
        self.assertEqual(encode_run_step(0.001, 1), b"RUN_STEP,0.001,1\r\n")
        self.assertEqual(encode_run_step(50.0, 3000), b"RUN_STEP,50,3000\r\n")

    def test_run_step_uses_configured_firmware_limits(self):
        self.assertEqual(
            encode_run_step(
                100.0,
                4000,
                target_max_rad_s=100.0,
                duration_max_ms=4000,
            ),
            b"RUN_STEP,100,4000\r\n",
        )

    def test_run_step_rejects_invalid_target_or_duration(self):
        for target in (0.0, -1.0, 50.001, float("inf"), float("nan")):
            with self.subTest(target=target), self.assertRaises(ValueError):
                encode_run_step(target, 1000)
        for duration in (0, 3001, 1.5, True, float("inf")):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                encode_run_step(5.0, duration)

    def test_parse_ok_and_generic_telemetry(self):
        self.assertEqual(parse_frame(b"OK\r\n"), ProtocolMessage("ACK", ("OK",)))
        message = parse_frame(b"DATA,50,2.0,1.8\r\n")
        telemetry = parse_telemetry(message)
        self.assertEqual(telemetry.elapsed_ms, 50)
        self.assertEqual(telemetry.target, 2.0)
        self.assertEqual(telemetry.feedback, 1.8)

    def test_parse_rejects_bad_lines_and_nonfinite_telemetry(self):
        invalid = (
            b"OK\n",
            b"ACK/PONG\r\n",
            b"DATA,50,2.0,1.8,0.4\r\n",
            b"DATA,-1,2.0,1.8\r\n",
            b"DATA,50,nan,1.8\r\n",
            b"DATA,50,2.0,inf\r\n",
            b"SET_PID,0.01,0.1,0\r\n",
            b"!V1,CMD,1,PING*EF\r\n",
            b"X" * 127 + b"\r\n",
        )
        for line in invalid:
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_frame(line)

    def test_ping_waits_for_ok_and_preserves_interleaved_data(self):
        telemetry = b"DATA,10,2.0,0.5\r\n"
        serial_port = FakeSerial((telemetry, b"debug noise\r\n"))
        link = MotorTuneLink(serial_port)
        reply = link.ping(timeout_s=0.2)
        self.assertEqual(reply, ProtocolMessage("ACK", ("OK",)))
        self.assertEqual(parse_telemetry(link.read_message()).elapsed_ms, 10)
        self.assertEqual(serial_port.writes, [b"PING\r\n"])

    def test_commands_wait_for_ok_and_match_firmware_lines(self):
        serial_port = FakeSerial()
        link = MotorTuneLink(serial_port)
        link.set_pid(0.01, 0.1, 0.0)
        link.run_step(5.0, 3000)
        link.stop()
        self.assertEqual(serial_port.writes, [
            b"SET_PID,0.01,0.1,0\r\n",
            b"RUN_STEP,5,3000\r\n",
            b"STOP\r\n",
        ])

    def test_link_run_step_passes_configured_firmware_limits(self):
        serial_port = FakeSerial()
        link = MotorTuneLink(serial_port)
        link.run_step(100.0, 4000, target_max_rad_s=100.0, duration_max_ms=4000)
        self.assertEqual(serial_port.writes, [b"RUN_STEP,100,4000\r\n"])

    def test_silent_firmware_rejection_times_out_without_retry(self):
        serial_port = FakeSerial(respond_ok=False)
        link = MotorTuneLink(serial_port)
        with self.assertRaisesRegex(TimeoutError, "silently ignored"):
            link.set_pid(0.01, 0.1, 0.0, timeout_s=0.01)
        self.assertEqual(serial_port.writes, [b"SET_PID,0.01,0.1,0\r\n"])

    def test_local_validation_prevents_writing_bad_command(self):
        serial_port = FakeSerial()
        link = MotorTuneLink(serial_port)
        with self.assertRaises(ValueError):
            link.run_step(51.0, 3000)
        self.assertEqual(serial_port.writes, [])


if __name__ == "__main__":
    unittest.main()
