"""Standalone PC-only motor PID optimizer; no serial or MCU access."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from motor_bo_core.runner import make_config, run_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated motor-speed PID tuning with Bayesian optimization")
    parser.add_argument("--tune-d", action="store_true", help="Search Kd too; default keeps Kd=0")
    parser.add_argument(
        "--llm", choices=("mock", "live", "replay"),
        help="Temporary mode override; normal mode is selected in motor_bo_config.json",
    )
    parser.add_argument("--replay-from", type=Path, help="Run directory whose LLM replies are replayed")
    parser.add_argument("--config", type=Path, help="Defaults to motor_bo_config.json")
    parser.add_argument("--run-dir", type=Path, help="New output directory, or paused directory with --resume")
    parser.add_argument("--resume", action="store_true", help="Continue a paused run")
    args = parser.parse_args()
    if args.resume and not args.run_dir:
        parser.error("--resume requires --run-dir")
    if not args.resume and args.llm == "replay" and not args.replay_from:
        parser.error("--llm replay requires --replay-from")
    if not args.resume and args.llm == "replay" and (args.config or args.tune_d):
        parser.error("Replay uses the source run's saved configuration")
    if args.resume and (args.config or args.tune_d or args.replay_from or args.llm):
        parser.error("Resume uses the saved configuration, LLM mode and reply source")
    run_dir = args.run_dir or Path("artifacts/motor_bo") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.resume:
        config = None
        mode = None
    elif args.llm == "replay":
        config = json.loads((args.replay_from / "manifest.json").read_text(encoding="utf-8"))["config"]
        mode = "replay"
    else:
        config = make_config(args.tune_d, args.config) if args.config else make_config(args.tune_d)
        mode = args.llm
    outcome = run_session(
        run_dir,
        config=config,
        mode=mode,
        replay_from=str(args.replay_from.resolve()) if args.replay_from else None,
        resume=args.resume,
    )
    print(f"Run: {outcome['run_dir']}")
    print(f"Best: {outcome['best']['gains']} score={outcome['best']['metrics']['score']:.5f}")


if __name__ == "__main__":
    main()
