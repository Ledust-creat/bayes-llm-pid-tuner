"""Run explicit hardware PID trials using firmware-defined target/feedback units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading

from motor_bo_core.hardware_runner import (
    DEFAULT_TUNE_CONFIG,
    HardwareStopRequested,
    _pid_axes,
    inspect_hardware_batch,
    load_tune_config,
    load_paused_hardware_config,
    make_hardware_advisor,
    run_fixed_hardware_validation,
    run_hardware_optimization,
    resume_hardware_optimization,
)
from motor_bo_core.serial_link import MotorTuneLink, parse_serial_config


class DesktopControlledLink:
    """Keep all serial I/O on the runner thread; stdin only requests cancellation."""

    def __init__(self, link, cancelled: threading.Event):
        self.link = link
        self.cancelled = cancelled

    def _check(self):
        if self.cancelled.is_set():
            raise HardwareStopRequested("用户请求停止")

    def clear_pending_messages(self):
        self._check()
        return self.link.clear_pending_messages()

    def set_pid(self, *args, **kwargs):
        self._check()
        return self.link.set_pid(*args, **kwargs)

    def run_step(self, *args, **kwargs):
        self._check()
        return self.link.run_step(*args, **kwargs)

    def read_message(self, *args, **kwargs):
        self._check()
        message = self.link.read_message(*args, **kwargs)
        self._check()
        return message

    def stop(self, *args, **kwargs):
        reply = self.link.stop(*args, **kwargs)
        if self.cancelled.is_set():
            print("停止命令已确认：OK", flush=True)
        return reply

    def __getattr__(self, name):
        return getattr(self.link, name)


def _listen_desktop_control(cancelled: threading.Event) -> None:
    # Only the CLI runner touches the COM port. EOF also requests orderly STOP.
    for line in sys.stdin:
        if line.strip() in {"STOP", "CANCEL"}:
            cancelled.set()
            return
    cancelled.set()


def main() -> int:
    parser = argparse.ArgumentParser(description="实机 PID 贝叶斯自动整定")
    parser.add_argument("--resume", action="store_true", help="恢复 AI 暂停的试验")
    parser.add_argument("--run-dir", type=Path, help="要恢复的运行目录")
    parser.add_argument("--gui-control", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        if args.resume:
            if args.run_dir is None:
                raise ValueError("恢复试验时必须提供 --run-dir <运行目录>")
            tune = load_paused_hardware_config(args.run_dir)
        else:
            if args.run_dir is not None:
                raise ValueError("--run-dir 只能与 --resume 一起使用")
            tune = load_tune_config(DEFAULT_TUNE_CONFIG)
        output_root = Path(__file__).resolve().parent / "artifacts" / "motor_bo_hardware"
        operation_mode = tune.get("operation_mode", "bayes")
        unit = tune.get("feedback_unit", "rad/s")
        if args.resume and operation_mode != "bayes":
            raise ValueError("--resume 仅适用于贝叶斯调参模式")
        batch = (
            inspect_hardware_batch(tune, output_root)
            if not args.resume and operation_mode == "bayes"
            else None
        )
        advisor = make_hardware_advisor(tune) if operation_mode == "bayes" else None
        serial_config = parse_serial_config(
            json.loads(DEFAULT_TUNE_CONFIG.read_text(encoding="utf-8"))
        )
        if operation_mode == "fixed_repeat":
            fixed = tune["fixed_repeat"]
            print(
                f"固定参数复测：{fixed['repeats']} 次；Kp={fixed['kp']:.5f}，"
                f"Ki={fixed['ki']:.5f}，Kd={fixed.get('kd', 0):.5f}；目标 {tune['target_rad_s']} {unit}，"
                f"每次 {tune['duration_ms']} ms。"
            )
            print("不运行贝叶斯优化、不调用 AI、不修改 PID 历史。")
        else:
            total = tune.get("rounds_per_batch", 1 + tune["initial_random_trials"] + tune["bayes_trials"])
            print(
                f"实机 PID {tune.get('pid_id', 'speed_pi')} 第 {batch['batch_index'] if batch else tune.get('batch_index', 1)} 批："
                f"{total} 轮；目标 {tune['target_rad_s']} {unit}；"
                f"每轮 {tune['duration_ms']} ms。"
            )
            active_ranges = batch['search_ranges'] if batch else tune.get('search_ranges', tune['ranges'])
            search_text = "，".join(f"{axis.upper()}={active_ranges[axis]}" for axis in _pid_axes(tune))
            fixed_text = "Ki=Kd=0（固定）" if tune.get("pid_mode") == "p" else ("Ki=0（固定）" if tune.get("pid_mode") == "pd" else (
                "全PID搜索" if len(_pid_axes(tune)) == 3 else f"Kd={tune.get('kd', 0)}（固定）")
            )
            print(
                f"历史模式：{tune.get('history_mode', 'new')}；有效历史 {len(batch['history']) if batch else tune.get('history_count', 0)} 轮；"
                f"本批搜索范围 {search_text}；{fixed_text}。"
            )
            if batch and batch["history"]:
                best = min(batch["history"], key=lambda row: row["score"])
                print(f"历史最佳：Kp={best['gains']['kp']:.5f}，Ki={best['gains']['ki']:.5f}，评分={best['score']:.5f}")
            elif not args.resume:
                print(f"新 PID 首轮基线：Kp={tune['baseline']['kp']}，Ki={tune['baseline']['ki']}")
                active_profile = output_root / "pid_profiles" / tune["pid_id"] / "active.json"
                if active_profile.exists():
                    print("注意：history_mode=new 会忽略当前 PID 已有历史并建立新系列；旧记录文件仍保留。")
            print(
                f"模式：{'贝叶斯 + AI' if tune['run_mode'] == 'bayes_llm' else '纯贝叶斯'}；"
                f"AI 咨询间隔：每 {tune['llm_interval_rounds']} 轮（仅混合模式生效）。"
            )
            print(
                (
                    "恢复使用暂停时保存的 PID 搜索边界；"
                    if args.resume
                    else "PID 搜索/发送边界以 motor_bo_hw_config.json 的 ranges 为准；"
                )
                + "请确认设备的 SET_PID 接受相同范围，否则命令可能被静默忽略。"
            )
        print(
            f"主机遥测绝对反馈停止阈值 {tune['max_feedback_rad_s']} {unit}。"
            "它依赖串口遥测，不替代设备本地保护与紧急停止措施。"
        )
        guard = tune.get("trial_guard")
        if guard and guard.get("enabled", True):
            print(
                f"试验判定：启动宽限 {guard['startup_grace_ms']} ms；"
                f"持续负向反馈异常 {guard['reverse_hold_ms']} ms 或近零反馈异常 "
                f"{guard['stall_hold_ms']} ms 后停止本批。"
            )
        else:
            print("负向/近零判定已关闭：负值及零值本身不视为故障；绝对反馈截止、遥测完整性和STOP确认仍有效。")
        limits = tune["protocol_limits"]
        left = "[" if limits.get("target_min_inclusive", False) else "("
        right = "]" if limits.get("target_max_inclusive", True) else ")"
        print(
            "RUN_STEP 配置范围："
            f"{left}{limits.get('target_min', 0):g}, {limits['target_max_rad_s']:g}{right} {unit}，"
            f"{tune['protocol_limits']['duration_max_ms']} ms；"
            "必须与设备实际接受的目标范围和时长一致。"
        )
        response = tune.get("response", {})
        print(f"评分类型：{response.get('mode', 'tracking')}；起点依据：{response.get('initial_reference', 'fixed')}；"
              f"归一化下限：{response.get('normalization_floor', 1e-6):g} {unit}；"
              f"绝对误差容差：{response.get('settling_tolerance_abs', 0):g} {unit}。")
        if response.get("initial_reference") == "first_sample":
            print("起点取首个DATA估计，不是准确的试验前反馈；严格比较需保持起点、负载及目标顺序一致。")
        targets = tune.get("target_values", [tune["target_rad_s"]])
        print(f"固定目标列表：{targets} {unit}；每组PID执行{len(targets)}次试验，"
              f"每目标{tune['duration_ms']} ms。")
        if len(targets) > 1:
            w = tune.get("multi_target_worst_weight", 0.3)
            print(f"综合分 = {1-w:g} × 平均分 + {w:g} × 最差目标分；每次试验结束均发送STOP。")

        link = MotorTuneLink.open(
            serial_config["port"], serial_config["baudrate"],
            timeout_s=serial_config["read_timeout_s"],
        )
        try:
            reply = link.ping(timeout_s=serial_config["ping_timeout_s"])
            print(f"串口握手成功：{','.join(reply.fields)}")
            prompt = "确认被控对象及实验条件已就绪后，输入 START 开始实机试验："
            if args.resume:
                prompt = "恢复前确认被控对象及实验条件已就绪；输入 START 从下一轮继续："
            if args.gui_control:
                print(prompt, flush=True)
                print("GUI_READY", flush=True)
                answer = input()
            else:
                answer = input(prompt)
            if answer.strip() != "START":
                print("已取消；没有启动实机试验。")
                return 0
            if args.gui_control:
                cancelled = threading.Event()
                link = DesktopControlledLink(link, cancelled)
                threading.Thread(
                    target=_listen_desktop_control, args=(cancelled,), daemon=True,
                ).start()
                print("GUI_RUNNING", flush=True)
            if args.resume:
                result = resume_hardware_optimization(link, args.run_dir, advisor=advisor)
            elif operation_mode == "fixed_repeat":
                result = run_fixed_hardware_validation(
                    link, tune, Path(__file__).resolve().parent / "artifacts"
                )
            else:
                result = run_hardware_optimization(link, tune, output_root, advisor=advisor)
            print(f"\n状态：{result['status']}\n记录目录：{result['run_dir']}")
            if result["status"] == "paused":
                print(
                    "AI 已暂停。确认安全后可恢复："
                    f"python motor_bo_hw.py --resume --run-dir \"{result['run_dir']}\""
                )
            if (Path(result["run_dir"]) / "speed_response.svg").exists():
                print(f"曲线：{Path(result['run_dir']) / 'speed_response.svg'}")
            if result.get("best"):
                best = result["best"]
                print(
                    f"当前最佳：Kp={best['gains']['kp']:.5f}, "
                    f"Ki={best['gains']['ki']:.5f}, "
                    f"Kd={best['gains'].get('kd', 0):.5f}, "
                    f"评分={best['metrics']['score']:.5f}"
                )
            if operation_mode == "fixed_repeat":
                summary = result["summary"]
                if summary["mean_score"] is not None:
                    print(
                        f"有效评分 {summary['valid_score_count']}/{summary['completed_repeats']}；"
                        f"均值={summary['mean_score']:.5f}，"
                        f"标准差={summary['score_stddev']:.5f}；"
                        f"汇总：{Path(result['run_dir']) / 'summary.json'}"
                    )
            else:
                manifest_path = Path(result["run_dir"]) / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if operation_mode == "bayes" and result["status"] == "completed":
                global_best = manifest["historical_best"]
                print(
                    f"跨批次最优：Kp={global_best['gains']['kp']:.5f}，"
                    f"Ki={global_best['gains']['ki']:.5f}，Kd={global_best['gains'].get('kd', 0):.5f}，评分={global_best['score']:.5f}\n"
                    f"跨批次历史：{manifest['profile_file']}\n"
                    f"下一批范围：{manifest['next_search_ranges']}\n"
                    "继续同一 PID：将 motor_bo_hw_config.json 的 history_mode 改为 continue 后再运行。"
                )
            return 0
        finally:
            try:
                link.stop(timeout_s=1.0)
            except Exception as exc:
                if args.gui_control:
                    print(f"STOP 未确认：{exc}", flush=True)
            link.close()
    except KeyboardInterrupt:
        print("\n已中断；上位机已尝试发送 STOP。")
        return 130
    except (OSError, RuntimeError, TimeoutError, ValueError, KeyError) as exc:
        print(f"实机调参未完成：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
