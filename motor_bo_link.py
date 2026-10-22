"""Smoke-test the PC-to-STM32 motor tuning UART link (does not start a motor test)."""

from __future__ import annotations

import json
from pathlib import Path
import time

from motor_bo_core.serial_link import MotorTuneLink, parse_serial_config


CONFIG_PATH = Path(__file__).resolve().with_name("motor_bo_hw_config.json")


def main() -> int:
    if not CONFIG_PATH.exists():
        print(f"找不到串口配置文件：{CONFIG_PATH}")
        return 2
    try:
        config = parse_serial_config(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except (ValueError, TypeError) as exc:
        print(f"串口配置错误：{exc}")
        return 2

    link = MotorTuneLink.open(
        config["port"],
        config["baudrate"],
        timeout_s=config["read_timeout_s"],
    )
    try:
        reply = link.ping(config["ping_timeout_s"])
        print(f"PING 成功，固件已回复：{','.join(reply.fields)}")

        listen_seconds = config["listen_after_ping_s"]
        if listen_seconds > 0.0:
            deadline = time.monotonic() + listen_seconds
            while time.monotonic() < deadline:
                try:
                    message = link.read_message(min(0.5, deadline - time.monotonic()))
                except TimeoutError:
                    continue
                print(f"收到 {message.kind}: {','.join(message.fields)}")
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"串口握手失败：{exc}")
        return 1
    finally:
        link.close()


if __name__ == "__main__":
    raise SystemExit(main())
