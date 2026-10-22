"""Local Windows desktop for the existing motor_bo_hw.py experiment runner."""

from __future__ import annotations

import colorsys
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser

from motor_bo_core.gui_support import (
    current_pid_best, expand_curves, get_value, initialize_local_config, list_runs,
    read_json, read_run, read_trace, save_json, with_settings,
)
from motor_bo_core.hardware_runner import _response_config

ROOT = Path(__file__).resolve().parent
HW_CONFIG = ROOT / "motor_bo_hw_config.json"
API_CONFIG = ROOT / "motor_bo_local.json"
UI_CONFIG = ROOT / "motor_bo_gui_settings.json"
LOG_PATH = ROOT / "motor_bo_terminal_latest.log"
THEMES = {
    "浅色": {"code": "light", "outer": "#eef2f7", "body": "#f4f6fa", "panel": "#ffffff",
             "text": "#20344b", "muted": "#66778c", "accent": "#275886", "grid": "#e7ecf2"},
    "深色": {"code": "dark", "outer": "#131b29", "body": "#1c2738", "panel": "#243246",
             "text": "#e6edf7", "muted": "#aebed2", "accent": "#7cc6ff", "grid": "#34465c"},
    "浅绿": {"code": "mint", "outer": "#e4efeb", "body": "#eef6f2", "panel": "#fafffc",
             "text": "#24463b", "muted": "#5e7b70", "accent": "#176d52", "grid": "#dcebe3"},
}
CHOICES = {
    "operation_mode": {"贝叶斯调参": "bayes", "固定参数复测": "fixed_repeat"},
    "run_mode": {"纯贝叶斯": "bayes_only", "贝叶斯＋AI": "bayes_llm"},
    "history_mode": {"新建历史": "new", "继续同一PID": "continue"},
    "pid_mode": {"P调参（I=D=0）": "p", "PI调参（D固定）": "pi", "PD调参（I=0）": "pd", "PID调参": "pid"},
    "response.mode": {"设定值跟踪": "tracking", "坐标阶跃（如位置角度）": "position"},
    "response.initial_reference": {"首帧反馈估计": "first_sample", "固定初始值": "fixed"},
    "protocol_limits.target_min_inclusive": {"包含下限": "true", "不包含下限": "false"},
    "protocol_limits.target_max_inclusive": {"包含上限": "true", "不包含上限": "false"},
    "trial_guard.enabled": {"关闭（允许负值/零位）": "false", "开启（仅正向跟踪）": "true"},
}
STATE_ZH = {
    "running": "运行中", "completed": "完成", "paused": "AI暂停",
    "stopped": "已停止", "interrupted": "已中断", "failed": "失败",
    "trial_aborted": "试验异常中止",
}

# key, Chinese caption, Python type; all saved values use the original JSON schema.
GROUPS = {
    "运行配置": [
        ("串口", [("serial.port", "串口", str), ("serial.baudrate", "波特率", int)]),
        ("实验", [("experiment_name", "被控对象 / 环路", str), ("feedback_unit", "反馈单位", str),
                  ("operation_mode", "运行方式", str), ("target_values", "目标列表（逗号分隔）", list),
                  ("duration_ms", "每次时长 ms", int),
                  ("inter_trial_delay_s", "轮间等待 s", float)]),
        ("贝叶斯与AI", [("run_mode", "搜索模式", str),
                       ("rounds_per_batch", "每批总轮数 2～20", int),
                       ("llm_interval_rounds", "AI咨询间隔 / 轮", int),
                       ("pid_id", "PID历史名称", str), ("history_mode", "历史方式", str),
                       ("pid_mode", "搜索参数", str),
                       ("multi_target_worst_weight", "最差目标评分权重", float)]),
        ("固定参数复测", [("fixed_repeat.kp", "固定 Kp", float),
                         ("fixed_repeat.ki", "固定 Ki", float),
                         ("fixed_repeat.kd", "固定 Kd", float),
                         ("fixed_repeat.repeats", "重复次数 1～20", int)]),
        ("首批基线", [("baseline.kp", "基线 Kp", float), ("baseline.ki", "基线 Ki", float),
                     ("baseline.kd", "基线 Kd", float)]),
        ("搜索硬边界", [("ranges.kp.0", "Kp下限", float), ("ranges.kp.1", "Kp上限", float),
                       ("ranges.ki.0", "Ki下限", float), ("ranges.ki.1", "Ki上限", float),
                       ("ranges.kd.0", "Kd下限", float), ("ranges.kd.1", "Kd上限", float)]),
    ],
    "边界与时序": [
        ("下位机协议范围", [("protocol_limits.target_min", "目标下限", float),
                           ("protocol_limits.target_min_inclusive", "下限边界", bool),
                           ("protocol_limits.target_max", "目标上限", float),
                           ("protocol_limits.target_max_inclusive", "上限边界", bool),
                           ("protocol_limits.duration_max_ms", "时长上限 ms", int)]),
        ("评分与坐标", [("response.mode", "响应类型", str),
                         ("response.initial_reference", "起点依据", str),
                         ("response.initial_value", "固定起点值", float),
                         ("response.normalization_floor", "归一化幅度下限", float),
                         ("response.settling_tolerance_abs", "绝对误差容差", float),
                         ("response.settling_tolerance_ratio", "相对误差容差", float),
                         ("response.cyclic_period", "反馈周期（0不循环）", float)]),
        ("反馈异常判定", [("feedback_abs_limit", "绝对反馈截止值", float),
                         ("trial_guard.enabled", "负向/近零判定", bool),
                         ("trial_guard.startup_grace_ms", "启动宽限 ms", int),
                         ("trial_guard.reverse_hold_ms", "持续负向异常 ms", int),
                         ("trial_guard.stall_hold_ms", "持续近零异常 ms", int)]),
        ("搜索初始化", [("initial_random_trials", "首批随机参数组数", int), ("seed", "随机种子（复现用）", int)]),
        ("串口时序", [("serial.read_timeout_s", "读取超时 s", float),
                     ("serial.ping_timeout_s", "握手超时 s", float),
                     ("serial.listen_after_ping_s", "PING工具监听 s", float)]),
    ],
}
API_FIELDS = [("provider", "协议提供方", str), ("base_url", "API地址", str),
              ("model", "模型名称", str), ("api_key", "API密钥", str)]


class ScrollForm(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, background="#f4f6fa", yscrollincrement=1)
        self._scroll_job = None
        self._scroll_target = 0.0
        self._wheel_tag = f"SmoothForm{id(self)}"
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas)
        window = self.canvas.create_window(0, 0, window=self.body, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.body.bind("<Configure>", lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(window, width=event.width))
        self.bind_class(self._wheel_tag, "<MouseWheel>", self._wheel)
        self.bind_class(self._wheel_tag, "<Button-4>", self._wheel)
        self.bind_class(self._wheel_tag, "<Button-5>", self._wheel)
        self.after_idle(self._install_wheel_bindings)

    def _install_wheel_bindings(self):
        # Intercept wheel before widget defaults, including combobox selection.
        def visit(widget):
            tags = widget.bindtags()
            if self._wheel_tag not in tags:
                widget.bindtags((self._wheel_tag, *tags))
            for child in widget.winfo_children():
                visit(child)
        visit(self)

    def _scroll_bounds(self):
        box = self.canvas.bbox("all")
        height = float(box[3] - box[1]) if box else 1.0
        return max(1.0, height), max(0.0, height - self.canvas.winfo_height())

    def _wheel(self, event):
        delta = event.delta if getattr(event, "delta", 0) else (120 if getattr(event, "num", 0) == 4 else -120)
        height, maximum = self._scroll_bounds()
        current = self.canvas.yview()[0] * height
        if self._scroll_job is None:
            self._scroll_target = current
        self._scroll_target = min(maximum, max(0.0, self._scroll_target - delta / 120.0 * 45))
        if self._scroll_job is None:
            self._scroll_job = self.after(16, self._animate_scroll)
        return "break"

    def _animate_scroll(self):
        height, maximum = self._scroll_bounds()
        self._scroll_target = min(maximum, self._scroll_target)
        current = self.canvas.yview()[0] * height
        difference = self._scroll_target - current
        if abs(difference) < 1.5:
            self.canvas.yview_moveto(self._scroll_target / height)
            self._scroll_job = None
            return
        self.canvas.yview_moveto((current + difference * 0.35) / height)
        self._scroll_job = self.after(16, self._animate_scroll)


class MotorTunerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("PID 自动整定台 · 贝叶斯优化 / AI辅助 / 固定复测")
        width = min(1380, root.winfo_screenwidth() - 80)
        height = min(800, root.winfo_screenheight() - 100)
        root.geometry(f"{width}x{height}+{max(0, (root.winfo_screenwidth() - width) // 2)}+0")
        root.minsize(1100, 650)
        root.configure(background="#eef2f7")
        root.option_add("*Font", ("Microsoft YaHei UI", 10))
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f6fa")
        style.configure("TLabel", background="#f4f6fa", foreground="#20344b")
        style.configure("TLabelframe", background="#f4f6fa")
        style.configure("TLabelframe.Label", background="#f4f6fa", foreground="#275886")
        style.configure("TButton", padding=(8, 5))
        style.configure("Treeview", rowheight=25)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 17, "bold"))

        self.fields: dict[str, tuple[tk.StringVar, type]] = {}
        self.field_widgets = {}
        self.ki_before_pd = {}
        self.api_fields: dict[str, tk.StringVar] = {}
        self.events = queue.SimpleQueue()
        self.process: subprocess.Popen | None = None
        self.awaiting_start = False
        self.stopping = False
        self.closing = False
        self.run_paths: list[Path] = []
        self.current_run: Path | None = None
        self.active_run: Path | None = None
        self.records: list[dict] = []
        self.curves: list[tuple[str, list[tuple[float, float]]]] = []
        self.curve_records = []
        self.visible_curves = set()
        self.known_curves = set()
        self.visibility_run = None
        self.show_new_curves = True
        self.manifest = {}
        self.live_points: list[tuple[float, float]] = []
        self.live_round = 0
        self.live_target = 0.0
        self.result_stamp = None
        self.existing_runs: set[Path] = set()
        self.status = tk.StringVar(value="就绪：编辑配置后，点击“连接并准备”")
        self.result_caption = tk.StringVar(value="选择历史运行即可查看完整曲线与评分")
        self.run_choice = tk.StringVar()
        self.palette = THEMES["浅色"]
        self.theme_choice = tk.StringVar(value="浅色")
        self._build()
        try:
            theme_code = read_json(UI_CONFIG).get("theme", "light") if UI_CONFIG.exists() else "light"
        except (OSError, ValueError):
            theme_code = "light"
        self.theme_choice.set(next((name for name, palette in THEMES.items() if palette["code"] == theme_code), "浅色"))
        self.apply_theme(save=False)
        self.load_settings()
        self.refresh_runs()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(120, self.poll_events)
        root.after(900, self.poll_results)

    @property
    def busy(self):
        return self.process is not None and self.process.poll() is None

    def _build(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="PID 自动整定台", style="Title.TLabel").pack(side="left", padx=(0, 20))
        self.reload_button = ttk.Button(top, text="重新加载", command=self.load_settings)
        self.save_button = ttk.Button(top, text="保存配置", command=self.save_hardware)
        self.prepare_button = ttk.Button(top, text="连接并准备", command=self.prepare)
        self.confirm_button = ttk.Button(top, text="确认开始", command=self.confirm_start, state="disabled")
        self.stop_button = ttk.Button(top, text="停止", command=self.stop, state="disabled")
        self.resume_button = ttk.Button(top, text="恢复所选暂停", command=lambda: self.prepare(resume=True))
        for button in (self.reload_button, self.save_button, self.prepare_button,
                       self.confirm_button, self.stop_button, self.resume_button):
            button.pack(side="left", padx=3)
        theme_box = ttk.Combobox(top, textvariable=self.theme_choice, values=list(THEMES), state="readonly", width=6)
        theme_box.pack(side="right")
        theme_box.bind("<<ComboboxSelected>>", lambda _: self.apply_theme())
        ttk.Label(top, text="主题 ").pack(side="right")
        ttk.Label(self.root, textvariable=self.status, padding=(14, 4)).pack(fill="x")

        split = ttk.Panedwindow(self.root, orient="horizontal")
        split.pack(fill="both", expand=True, padx=10, pady=(6, 10))
        settings = ttk.Notebook(split, width=365)
        split.add(settings, weight=0)
        for name, groups in GROUPS.items():
            page = ScrollForm(settings)
            settings.add(page, text=name)
            for title, fields in groups:
                frame = ttk.LabelFrame(page.body, text=title, padding=8)
                frame.pack(fill="x", padx=7, pady=5)
                frame.columnconfigure(1, weight=1)
                for index, (key, caption, kind) in enumerate(fields):
                    var = tk.StringVar()
                    self.fields[key] = (var, kind)
                    ttk.Label(frame, text=caption).grid(row=index, column=0, sticky="w", pady=4, padx=(0, 8))
                    if key in CHOICES:
                        entry = ttk.Combobox(frame, textvariable=var, values=list(CHOICES[key]), state="readonly", width=17)
                    elif key == "serial.port":
                        row = ttk.Frame(frame)
                        row.grid(row=index, column=1, sticky="ew")
                        self.port_box = ttk.Combobox(row, textvariable=var, width=11)
                        self.port_box.pack(side="left", fill="x", expand=True)
                        ttk.Button(row, text="刷新", width=5, command=self.scan_ports).pack(side="left", padx=3)
                        continue
                    else:
                        entry = ttk.Entry(frame, textvariable=var, width=17)
                    entry.grid(row=index, column=1, sticky="ew", pady=4)
                    self.field_widgets[key] = entry
                    if key == "pid_mode":
                        entry.bind("<<ComboboxSelected>>", lambda _: self.sync_pid_mode())
                if title == "固定参数复测":
                    self.import_best_button = ttk.Button(frame, text="当前PID历史最优 → 复测", command=self.use_current_best)
                    self.import_best_button.grid(row=len(fields), column=0, columnspan=2, sticky="ew", pady=(6, 0))
            if name == "运行配置":
                ttk.Label(page.body, text="每组PID依次测试全部目标。\n目标与反馈使用设备约定的同一单位。\nP模式I/D固定0；PD模式I固定0。\n保存只更新设置，确认开始才发送试验命令。\n对象名称用于标记记录，不切换设备控制方式。",
                          justify="left", padding=10).pack(fill="x")
            else:
                ttk.Label(page.body, text="负值或零值合法时，应关闭负向/近零判定。\n评分尺度=max(阶跃幅度,归一化下限)。\n容差=max(绝对容差,幅度×相对容差)。\n周期/幅度/容差使用反馈单位。\n首帧反馈只是起点估计，时间由设备采样决定。\n随机参数组用于探索，种子用于复现候选序列。",
                          justify="left", padding=10).pack(fill="x")

        api_page = ttk.Frame(settings, padding=12)
        settings.add(api_page, text="API")
        for key, caption, _ in API_FIELDS:
            ttk.Label(api_page, text=caption).pack(anchor="w", pady=(10, 3))
            var = tk.StringVar()
            self.api_fields[key] = var
            entry = ttk.Entry(api_page, textvariable=var, show="*" if key == "api_key" else "")
            entry.pack(fill="x")
            if key == "api_key":
                self.key_entry = entry
        self.show_key = tk.BooleanVar()
        ttk.Checkbutton(api_page, text="显示密钥", variable=self.show_key,
                        command=lambda: self.key_entry.configure(show="" if self.show_key.get() else "*")).pack(anchor="w", pady=10)
        self.api_save_button = ttk.Button(api_page, text="保存API配置", command=self.save_api)
        self.api_save_button.pack(fill="x", pady=5)
        ttk.Label(api_page, text="保存到 motor_bo_local.json，仅混合模式使用。\nMOTOR_BO_* 环境变量仍优先于文件。\n保存配置不会发起API请求。",
                  wraplength=310, justify="left").pack(anchor="w", pady=15)

        right = ttk.Frame(split)
        split.add(right, weight=1)
        chooser = ttk.Frame(right)
        chooser.pack(fill="x", pady=(0, 5))
        ttk.Label(chooser, text="运行记录").pack(side="left", padx=(0, 7))
        self.run_box = ttk.Combobox(chooser, textvariable=self.run_choice, state="readonly", width=51)
        self.run_box.pack(side="left", fill="x", expand=True)
        self.run_box.bind("<<ComboboxSelected>>", self.select_run)
        ttk.Button(chooser, text="刷新记录", command=self.refresh_runs).pack(side="left", padx=4)
        actions = ttk.Frame(right)
        actions.pack(fill="x", pady=(0, 7))
        ttk.Button(actions, text="所选记录最优 → 复测", command=self.use_best).pack(side="left")
        ttk.Button(actions, text="打开记录目录", command=self.open_run).pack(side="left", padx=5)
        ttk.Button(actions, text="打开SVG曲线", command=self.open_svg).pack(side="left")
        ttk.Button(actions, text="浏览记录…", command=self.browse_run).pack(side="right")
        ttk.Label(right, textvariable=self.result_caption, wraplength=850).pack(fill="x", pady=4)
        tabs = ttk.Notebook(right)
        tabs.pack(fill="both", expand=True)
        results = ttk.Frame(tabs, padding=6)
        tabs.add(results, text="曲线与评分")
        results.columnconfigure(0, weight=1)
        results.rowconfigure(1, weight=1)
        # Reserve the selector/table before giving spare height to the chart.
        # Treeview and scrollbar must be children of their actual layout frame;
        # pack(in_=...) with siblings can leave them covered by the newer frame.
        filters = ttk.Frame(results)
        filters.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(filters, text="全部显示", command=lambda: self.set_curve_visibility("all")).pack(side="left")
        ttk.Button(filters, text="全部隐藏", command=lambda: self.set_curve_visibility("none")).pack(side="left", padx=4)
        ttk.Button(filters, text="仅显示最优组", command=lambda: self.set_curve_visibility("best")).pack(side="left")
        ttk.Label(filters, text="首列勾选显示曲线").pack(side="right")
        self.curve_filters = filters
        self.chart = tk.Canvas(results, background="white", highlightthickness=0, height=320)
        self.chart.grid(row=1, column=0, sticky="nsew")
        self.chart.bind("<Configure>", lambda _: self.draw_chart())
        table_row = ttk.Frame(results)
        table_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.table = ttk.Treeview(table_row, columns=("round", "pid", "score", "settle"), show="headings", height=7)
        for key, title, width in (("round", "显示 / 轮次 / 目标", 175), ("pid", "Kp / Ki / Kd", 240),
                                  ("score", "评分 ↓", 110),
                                  ("settle", "调节时间 s", 120)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, anchor="center", stretch=True)
        scroll = ttk.Scrollbar(table_row, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.table.bind("<Button-1>", self.toggle_curve)
        self.log = self._text_tab(tabs, "实时日志", clear=True)
        self.advice = self._text_tab(tabs, "AI中文评价")
        self.log.tag_configure("ai", foreground="#235f95")

    def _text_tab(self, tabs, title, *, clear=False):
        frame = ttk.Frame(tabs, padding=6)
        tabs.add(frame, text=title)
        if clear:
            toolbar = ttk.Frame(frame)
            toolbar.pack(fill="x", pady=(0, 5))
            self.clear_log_button = ttk.Button(toolbar, text="清空日志", command=self.clear_log)
            self.clear_log_button.pack(side="left")
            ttk.Label(toolbar, text="仅清空窗口显示，不删除文件或停止试验").pack(side="left", padx=8)
        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True)
        text = tk.Text(body, wrap="word", state="disabled", background="white", foreground="#213547", relief="flat")
        scroll = ttk.Scrollbar(body, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        return text

    def clear_log(self):
        """Clear display only; queued/new messages and saved evidence are untouched."""
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def apply_theme(self, save=True):
        """Apply visual preferences independently of motor and API configuration."""
        self.palette = palette = THEMES[self.theme_choice.get()]
        style = ttk.Style(self.root)
        style.configure(".", background=palette["body"], foreground=palette["text"])
        for name in ("TFrame", "TLabel", "TLabelframe", "TCheckbutton", "TNotebook"):
            style.configure(name, background=palette["body"], foreground=palette["text"])
        style.configure("TLabelframe.Label", background=palette["body"], foreground=palette["accent"])
        for name in ("TEntry", "TCombobox"):
            style.configure(name, fieldbackground=palette["panel"], foreground=palette["text"], selectbackground=palette["accent"])
            style.map(name, fieldbackground=[("readonly", palette["panel"]), ("disabled", palette["body"])],
                      foreground=[("readonly", palette["text"]), ("disabled", palette["muted"])])
        for name in ("TButton", "TNotebook.Tab", "Treeview.Heading"):
            style.configure(name, background=palette["body"], foreground=palette["text"])
            style.map(name, background=[("selected", palette["panel"]), ("active", palette["grid"])],
                      foreground=[("disabled", palette["muted"])])
        style.configure("Treeview", background=palette["panel"], fieldbackground=palette["panel"], foreground=palette["text"])
        style.map("Treeview", background=[("selected", palette["accent"])], foreground=[("selected", palette["panel"])])
        self.root.configure(background=palette["outer"])

        def paint(widget):
            if isinstance(widget, tk.Canvas):
                widget.configure(background=palette["panel"] if widget is self.chart else palette["body"])
            elif isinstance(widget, tk.Text):
                widget.configure(background=palette["panel"], foreground=palette["text"], insertbackground=palette["text"])
            for child in widget.winfo_children():
                paint(child)
        paint(self.root)
        self.root.option_add("*TCombobox*Listbox.background", palette["panel"])
        self.root.option_add("*TCombobox*Listbox.foreground", palette["text"])
        self.log.tag_configure("ai", foreground=palette["accent"])
        for record in self.records:
            self.table.tag_configure(str(record["round"]), foreground=self.color(record["round"]))
        self.draw_chart()
        if save:
            try:
                data = read_json(UI_CONFIG) if UI_CONFIG.exists() else {}
                data["theme"] = palette["code"]
                save_json(UI_CONFIG, data)
                self.status.set(f"主题已切换为{self.theme_choice.get()}，下次启动自动恢复")
            except (OSError, ValueError) as exc:
                self.status.set(f"主题已应用，但偏好未保存：{exc}")

    def load_settings(self):
        if self.busy:
            return
        try:
            initialized = initialize_local_config(HW_CONFIG, ROOT / "motor_bo_hw_config.example.json")
            self.hardware_source = HW_CONFIG.read_text(encoding="utf-8")
            data = read_json(HW_CONFIG)
            self.ki_before_pd = {}
            for key in ("baseline.ki", "fixed_repeat.ki", "baseline.kd", "fixed_repeat.kd"):
                self.field_widgets[key].configure(state="normal")
            for key, (var, _) in self.fields.items():
                if key.startswith("ranges."):
                    _, axis, index = key.split(".")
                    value = data["ranges"].get(axis, [0, 0.02])[int(index)]
                else:
                    defaults = {"experiment_name": "未命名对象", "feedback_unit": "rad/s",
                                "pid_mode": "pid" if data.get("tune_d", False) else "pi", "multi_target_worst_weight": 0.3,
                                "target_values": [data.get("target_rad_s", 20)],
                                "baseline.kd": 0.0, "fixed_repeat.kd": 0.0}
                    value = get_value(data, key, defaults.get(key, ""))
                    value = self._rule_values(data).get(key, value)
                if isinstance(value, bool):
                    value = "true" if value else "false"
                if isinstance(value, list):
                    value = ", ".join(str(item) for item in value)
                choices = CHOICES.get(key, {})
                var.set(next((label for label, code in choices.items() if code == value), str(value)))
            self.api_source = API_CONFIG.read_text(encoding="utf-8") if API_CONFIG.exists() else ""
            api = read_json(API_CONFIG) if API_CONFIG.exists() else {"provider": "openai"}
            for key, var in self.api_fields.items():
                var.set(str(api.get(key, "")))
            self.status.set("首次启动已生成默认配置；请在左侧选择串口并核对试验参数" if initialized
                            else "已加载配置；界面启动时不会连接串口")
            self.sync_pid_mode()
            self.scan_ports()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            messagebox.showerror("读取配置失败", str(exc), parent=self.root)

    def sync_pid_mode(self):
        """P fixes I/D; PD fixes I. Other modes restore unsaved fixed-gain fields."""
        mode = CHOICES["pid_mode"].get(self.fields["pid_mode"][0].get())
        for axis in ("ki", "kd"):
            fixed = mode == "p" or (axis == "ki" and mode == "pd")
            for key in (f"baseline.{axis}", f"fixed_repeat.{axis}"):
                widget, var = self.field_widgets[key], self.fields[key][0]
                was_disabled = str(widget.cget("state")) == "disabled"
                if fixed:
                    if not was_disabled:
                        self.ki_before_pd[key] = var.get()
                    var.set("0")
                elif was_disabled:
                    var.set(self.ki_before_pd.get(key, var.get()))
                widget.configure(state="disabled" if fixed else "normal")
            for index in (0, 1):
                self.field_widgets[f"ranges.{axis}.{index}"].configure(state="disabled" if fixed else "normal")

    @staticmethod
    def _rule_values(data):
        limits = data.get("protocol_limits", {})
        response = _response_config(data)
        return {"protocol_limits.target_min": limits.get("target_min", 0.0),
                "protocol_limits.target_max": limits.get("target_max", limits.get("target_max_rad_s", 50.0)),
                "protocol_limits.target_min_inclusive": limits.get("target_min_inclusive", False),
                "protocol_limits.target_max_inclusive": limits.get("target_max_inclusive", True),
                "protocol_limits.duration_max_ms": limits.get("duration_max_ms", 3000),
                "feedback_abs_limit": data.get("feedback_abs_limit", data.get("max_feedback_rad_s", 120.0)),
                "trial_guard.enabled": data.get("trial_guard", {}).get("enabled", response["mode"] == "tracking"),
                **{f"trial_guard.{key}": data.get("trial_guard", {}).get(key, default) for key, default in
                   (("startup_grace_ms", min(100, data.get("duration_ms", 3000))), ("reverse_hold_ms", 100), ("stall_hold_ms", 500))},
                **{f"response.{key}": value for key, value in response.items()}}

    def save_hardware(self, quiet=False):
        if self.busy:
            return False
        try:
            if HW_CONFIG.read_text(encoding="utf-8") != self.hardware_source:
                raise ValueError("JSON已被其他程序修改，请先点击“重新加载”，避免覆盖新配置。")
            settings = {}
            for key, (var, kind) in self.fields.items():
                value = var.get()
                settings[key] = (CHOICES.get(key, {}).get(value, value), kind)
            data = with_settings(read_json(HW_CONFIG), settings)
            data.pop("tune_d", None)  # pid_mode supersedes the old D-only switch.
            data.get("protocol_limits", {}).pop("supports_kd", None)
            data.get("protocol_limits", {}).pop("target_max_rad_s", None)
            data.pop("max_feedback_rad_s", None)
            save_json(HW_CONFIG, data, hardware=True)
            self.hardware_source = HW_CONFIG.read_text(encoding="utf-8")
            if not quiet:
                self.status.set("实机配置已保存到 motor_bo_hw_config.json")
            return True
        except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
            messagebox.showerror("配置未保存", str(exc), parent=self.root)
            return False

    def save_api(self):
        if self.busy:
            return
        try:
            source = API_CONFIG.read_text(encoding="utf-8") if API_CONFIG.exists() else ""
            if source != self.api_source:
                raise ValueError("API文件已被修改，请先重新加载。")
            data = read_json(API_CONFIG) if API_CONFIG.exists() else {}
            data.update({key: var.get().strip() for key, var in self.api_fields.items()})
            if not all(data.get(key) for key in ("provider", "base_url", "model", "api_key")):
                raise ValueError("请填写协议提供方、API地址、模型名称和密钥。")
            if not data["base_url"].startswith(("http://", "https://")):
                raise ValueError("API地址必须以 http:// 或 https:// 开头。")
            save_json(API_CONFIG, data)
            self.api_source = API_CONFIG.read_text(encoding="utf-8")
            self.status.set("API配置已保存；密钥不会显示在运行日志中")
        except (OSError, ValueError) as exc:
            messagebox.showerror("API配置未保存", str(exc), parent=self.root)

    def scan_ports(self):
        try:
            from serial.tools import list_ports
            self.port_box.configure(values=[port.device for port in list_ports.comports()])
        except ImportError:
            self.status.set("未安装pyserial；仍可编辑配置，运行前请安装requirements-motor-bo.txt")

    def _set_busy(self, value):
        for button in (self.prepare_button, self.save_button, self.reload_button,
                       self.resume_button, self.api_save_button, self.import_best_button):
            button.configure(state="disabled" if value else "normal")
        self.stop_button.configure(state="normal" if value else "disabled")
        if not value:
            self.confirm_button.configure(state="disabled")
            self.awaiting_start = False

    def prepare(self, resume=False):
        if self.busy:
            return
        args = []
        if resume:
            try:
                if not self.current_run or read_run(self.current_run)[0].get("status") != "paused":
                    raise ValueError("请先选择一条状态为“AI暂停”的运行记录。")
            except (OSError, ValueError) as exc:
                messagebox.showerror("不能恢复", str(exc), parent=self.root)
                return
            args = ["--resume", "--run-dir", str(self.current_run)]
        if not self.save_hardware(quiet=True):
            return
        self.launch_config = read_json(HW_CONFIG)
        self.stopping = False
        self.existing_runs = set(list_runs(ROOT))
        self.active_run = self.current_run if resume else None
        self.live_round, self.live_points = 0, []
        self.result_stamp = None
        if not resume:
            self.current_run = None
            self.records, self.curves, self.manifest = [], [], {}
            self.table.delete(*self.table.get_children())
        python = ROOT / "venv" / "Scripts" / "python.exe"
        executable = str(python) if python.exists() else sys.executable
        command = [executable, "-X", "utf8", "-u", str(ROOT / "motor_bo_hw.py"), "--gui-control", *args]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        try:
            self.process = subprocess.Popen(
                command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            messagebox.showerror("无法启动", str(exc), parent=self.root)
            return
        self._set_busy(True)
        self.status.set("正在准备并握手；成功后点击“确认开始”")
        secrets = tuple(value for value in (self.api_fields["api_key"].get(), os.getenv("MOTOR_BO_API_KEY", "")) if value)
        threading.Thread(target=self._read_process, args=(self.process, secrets), daemon=True).start()

    def _read_process(self, process, secrets):
        try:
            with LOG_PATH.open("w", encoding="utf-8") as handle:
                for line in process.stdout:
                    for secret in secrets:
                        line = line.replace(secret, "[密钥已隐藏]")
                    handle.write(line)
                    handle.flush()
                    self.events.put(("line", line))
        except OSError as exc:
            # Continue draining stdout even if saving a transcript failed.
            self.events.put(("line", f"日志文件写入失败：{exc}\n"))
            for line in process.stdout:
                for secret in secrets:
                    line = line.replace(secret, "[密钥已隐藏]")
                self.events.put(("line", line))
        finally:
            code = process.wait()
            process.stdout.close()
            process.stdin.close()
            self.events.put(("done", code))

    def _send(self, command):
        if not self.busy:
            return False
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
            return True
        except (OSError, ValueError):
            self.status.set("后台进程已退出，请查看实时日志")
            return False

    def confirm_start(self):
        if self.awaiting_start and self._send("START"):
            self.awaiting_start = False
            self.confirm_button.configure(state="disabled")
            self.status.set("运行中；停止按钮通过后台串口线程请求STOP")

    def stop(self):
        if self._send("STOP"):
            self.stopping = True
            self.awaiting_start = False
            self.confirm_button.configure(state="disabled")
            self.status.set("已请求停止，等待后台STOP确认与记录保存…")

    def _append_log(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line, "ai" if "AI" in line else "")
        if int(self.log.index("end-1c").split(".")[0]) > 4000:
            self.log.delete("1.0", "500.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def poll_events(self):
        while not self.events.empty():
            kind, value = self.events.get()
            if kind == "done":
                self._set_busy(False)
                self.process = None
                if self.active_run:
                    self.current_run = self.active_run
                self.refresh_runs()
                self.load_results(force=True)
                self.status.set(f"后台进程结束，退出码 {value}；结果保存在运行记录中")
                if self.closing:
                    self.root.destroy()
                    return
                continue
            line = value
            if line.strip() == "GUI_READY":
                if self.stopping:
                    continue
                self.awaiting_start = True
                self.confirm_button.configure(state="normal")
                self.status.set("握手成功：点击“确认开始”后才执行调参或复测试验")
                continue
            if line.strip() == "GUI_RUNNING":
                continue
            self._append_log(line)
            match = re.search(r"(?:实机第|固定复测)\s+(\d+)/(\d+).*?Kp=([\d.eE+\-]+),\s*Ki=([\d.eE+\-]+)(?:,\s*Kd=([\d.eE+\-]+))?", line)
            if match:
                self.live_round = int(match[1])
                self.live_points = []
                if not self.stopping:
                    self.status.set(f"第 {match[1]}/{match[2]} 轮 · Kp={match[3]} · Ki={match[4]} · Kd={match[5] or '0'}")
            condition = re.search(r"目标测试\s+\d+/\d+：目标=([\d.eE+\-]+)", line)
            if condition:
                self.live_target = float(condition[1])
                self.live_points = []
            sample = re.search(r"(\d+) ms 目标=([\d.eE+\-]+) 反馈=([\d.eE+\-]+)", line)
            if sample:
                self.live_target = float(sample[2])
                self.live_points.append((int(sample[1]) / 1000.0, float(sample[3])))
                self.draw_chart()
        self.root.after(120, self.poll_events)

    def refresh_runs(self):
        self.run_paths = list_runs(ROOT)
        captions = []
        for path in self.run_paths:
            try:
                manifest, _ = read_run(path)
                config = manifest.get("config", {})
                kind = "复测" if "validation" in path.parent.name else "调参"
                captions.append(f"{path.name} | {kind} | {config.get('target_values', [config.get('target_rad_s', '?')])} {config.get('feedback_unit', 'rad/s')} | {STATE_ZH.get(manifest.get('status'), manifest.get('status', '?'))}")
            except (OSError, ValueError):
                captions.append(path.name)
        self.run_box.configure(values=captions)
        if self.current_run in self.run_paths:
            self.run_box.current(self.run_paths.index(self.current_run))
        elif self.run_paths and not self.busy:
            self.run_box.current(0)
            self.current_run = self.run_paths[0]
        self.load_results(force=True)

    def select_run(self, _=None):
        index = self.run_box.current()
        if 0 <= index < len(self.run_paths):
            self.current_run = self.run_paths[index]
            self.result_stamp = None
            self.load_results(force=True)

    def poll_results(self):
        if self.busy:
            if self.active_run is None:
                fresh = [path for path in list_runs(ROOT) if path not in self.existing_runs]
                if fresh:
                    self.active_run = self.current_run = fresh[0]
                    self.refresh_runs()
            self.load_results()
        self.root.after(900, self.poll_results)

    def load_results(self, force=False):
        path = self.current_run
        if path is None:
            self.draw_chart()
            return
        try:
            stamp = (path, *[(path / name).stat().st_mtime_ns if (path / name).exists() else 0
                             for name in ("manifest.json", "rounds.jsonl", "ai_advice.md")])
            if stamp == self.result_stamp and not force:
                return
            manifest, records = read_run(path)
            curve_records = expand_curves(records, manifest.get("config", {}).get("target_rad_s", 0))
            curves = [(record["curve_id"], read_trace(path, record)) for record in curve_records]
            if self.visibility_run != path:
                self.visible_curves, self.known_curves = set(), set()
                self.visibility_run = path
                self.show_new_curves = True
            ids = {record["curve_id"] for record in curve_records}
            if self.show_new_curves:
                self.visible_curves.update(ids - self.known_curves)
            self.known_curves = ids
            self.curve_records = curve_records
            self.manifest, self.records, self.curves = manifest, records, curves
            self.result_stamp = stamp
            selected = self.table.selection()
            self.table.delete(*self.table.get_children())
            for record in curve_records:
                number, gains, metrics = record["round"], record["gains"], record["metrics"]
                score = f"{metrics['score']:.5f}" if metrics.get("valid") else "无效"
                settle = metrics.get("settling_time_s")
                self.table.tag_configure(str(number), foreground=self.color(number))
                checked = "☑" if record["curve_id"] in self.visible_curves else "☐"
                self.table.insert("", "end", iid=record["curve_id"], tags=(str(number),), values=(
                    f"{checked} 第{number}轮 · {record['target']:g}", f"{gains['kp']:.5f} / {gains['ki']:.5f} / {gains.get('kd', 0):.5f}", score,
                    f"{settle:.3f}" if settle is not None else "—"))
            if selected and self.table.exists(selected[0]):
                self.table.selection_set(selected[0])
            valid = [r for r in records if r["metrics"].get("valid")]
            text = f"{STATE_ZH.get(manifest.get('status'), manifest.get('status', '?'))} · 已记录 {len(records)} 轮"
            if valid:
                best = min(valid, key=lambda r: r["metrics"]["score"])
                text += f" · 最佳第{best['round']}轮：{best['metrics']['score']:.5f}"
            if manifest.get("mean_score") is not None:
                text += f" · 均值 {manifest['mean_score']:.5f} ± {manifest['score_stddev']:.5f}"
            if manifest.get("fault"):
                fault = manifest["fault"].replace("sustained_reverse_stop", "持续负向反馈异常").replace("sustained_stall_stop", "持续近零反馈异常").replace("host_overspeed_stop", "绝对反馈超限")
                text += f" · 原因 {fault}"
            self.result_caption.set(text)
            advice_path = path / "ai_advice.md"
            advice = advice_path.read_text(encoding="utf-8") if advice_path.exists() else "固定复测不调用AI。"
            self.advice.configure(state="normal")
            self.advice.delete("1.0", "end")
            self.advice.insert("end", advice)
            self.advice.configure(state="disabled")
            self.draw_chart()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.result_caption.set(f"记录正在写入或无法读取：{exc}")

    def color(self, number):
        rgb = colorsys.hsv_to_rgb((number * 0.61803398875) % 1.0, 0.72,
                                 0.9 if self.palette["code"] == "dark" else 0.66)
        return "#" + "".join(f"{round(value * 255):02x}" for value in rgb)

    def toggle_curve(self, event):
        """Only the first table column toggles display; selection never changes width."""
        key = self.table.identify_row(event.y)
        if key and self.table.identify_column(event.x) == "#1":
            if key in self.visible_curves:
                self.visible_curves.remove(key)
            else:
                self.visible_curves.add(key)
            self.update_curve_checks()
            return "break"

    def update_curve_checks(self):
        for record in self.curve_records:
            key = record["curve_id"]
            if self.table.exists(key):
                checked = "☑" if key in self.visible_curves else "☐"
                self.table.set(key, "round", f"{checked} 第{record['round']}轮 · {record['target']:g}")
        self.draw_chart()

    def set_curve_visibility(self, mode):
        self.show_new_curves = mode == "all"
        if mode == "all":
            self.visible_curves = set(self.known_curves)
        elif mode == "none":
            self.visible_curves = set()
        else:
            valid = [record for record in self.records if record["metrics"].get("valid")]
            best = min(valid, key=lambda record: record["metrics"]["score"]) if valid else None
            self.visible_curves = {r["curve_id"] for r in self.curve_records
                                   if best and r["round"] == best["round"]}
        self.update_curve_checks()

    def draw_chart(self):
        canvas = self.chart
        palette = self.palette
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 120 or height < 100:
            return
        config = self.manifest.get("config", {})
        curve_map = {record["curve_id"]: record for record in self.curve_records}
        curves = [(curve_map[key]["round"], points, curve_map[key]["target"])
                  for key, points in self.curves if key in self.visible_curves]
        if (self.show_new_curves and self.busy and self.current_run == self.active_run
                and self.live_points and self.live_round > len(self.records)):
            curves.append((self.live_round, self.live_points, self.live_target))
            if not config:
                config = self.launch_config
        if not curves:
            canvas.create_text(width / 2, height / 2, text="勾选下方表格显示曲线；或点击全部显示\n运行时预览当前目标，轮结束后显示完整CSV", fill=palette["muted"], justify="center")
            return
        all_points = [p for _, points, _ in curves for p in points]
        if not all_points:
            return
        xmax = max([float(config.get("duration_ms", 0)) / 1000.0, *[x for x, _ in all_points], 0.1])
        low = min(0.0, min(y for _, y in all_points))
        targets = sorted({target for _, _, target in curves})
        high = max(*targets, max(y for _, y in all_points), low + 1)
        pad = (high - low) * 0.08
        low, high = low - pad, high + pad
        left, right, top, bottom = 60, width - 18, 30, height - 38
        xy = lambda x, y: (left + x / xmax * (right - left), bottom - (y - low) / (high - low) * (bottom - top))
        for i in range(6):
            x, y = xmax * i / 5, low + (high - low) * i / 5
            px, py = xy(x, y)
            canvas.create_line(px, top, px, bottom, fill=palette["grid"])
            canvas.create_line(left, py, right, py, fill=palette["grid"])
            canvas.create_text(px, bottom + 14, text=f"{x:.1f}", fill=palette["muted"])
            canvas.create_text(left - 8, py, text=f"{y:.1f}", anchor="e", fill=palette["muted"])
        unit = config.get("feedback_unit", "rad/s")
        canvas.create_text(left, 12, text=f"反馈值 ({unit})", anchor="w", fill=palette["text"])
        canvas.create_text(right, height - 8, text="时间 s", anchor="e", fill=palette["text"])
        for target in targets:
            _, ty = xy(0, target)
            canvas.create_line(left, ty, right, ty, fill=palette["text"], dash=(6, 4), width=1)
            canvas.create_text(right - 5, ty - 11, text=f"目标 {target:g}", anchor="e", fill=palette["text"])
        for number, points, _ in curves:
            coords = [coord for x, y in points for coord in xy(x, y)]
            if len(coords) >= 4:
                canvas.create_line(*coords, fill=self.color(number), width=1.5, tags=("feedback_curve",))
        canvas.create_text(right, 12, text="颜色对应轮次；勾选表格控制显示", anchor="e", fill=palette["muted"])

    def use_best(self):
        if self.busy:
            return
        valid = [record for record in self.records if record["metrics"].get("valid")]
        if not valid:
            messagebox.showinfo("没有可用参数", "请先选择一条包含有效评分的运行记录。", parent=self.root)
            return
        best = min(valid, key=lambda record: record["metrics"]["score"])
        try:
            self._import_best(best["gains"], self.manifest.get("config", {}),
                              f"所选记录第{best['round']}轮，综合分{best['metrics']['score']:.5f}")
        except ValueError as exc:
            messagebox.showinfo("不能导入不同试验", str(exc), parent=self.root)

    def use_current_best(self):
        if self.busy:
            return
        pid_id = self.fields["pid_id"][0].get().strip()
        try:
            best, config = current_pid_best(ROOT, pid_id)
            self._import_best(best["gains"], config, f"PID {pid_id} 跨批次历史最优，综合分{best['score']:.5f}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            messagebox.showinfo("不能导入历史最优", str(exc), parent=self.root)

    def _import_best(self, gains, config, source):
        mode = config.get("pid_mode", "pid" if config.get("tune_d", False) else "pi")
        if mode not in {"p", "pi", "pd", "pid"}:
            raise ValueError("历史记录的PID模式无效")
        source_rules = self._rule_values(config)
        source_unit = config.get("feedback_unit", "rad/s")
        current_response = CHOICES["response.mode"].get(self.fields["response.mode"][0].get())
        if source_unit != self.fields["feedback_unit"][0].get() or source_rules["response.mode"] != current_response:
            raise ValueError("所选记录的单位/响应类型与当前配置不同；请先选择匹配的试验规则，不能混用不同被控量的最优参数")
        was_pd = CHOICES["pid_mode"].get(self.fields["pid_mode"][0].get()) in {"p", "pd"}
        self.fields["pid_mode"][0].set(next(label for label, code in CHOICES["pid_mode"].items() if code == mode))
        self.sync_pid_mode()
        if was_pd and mode not in {"p", "pd"}:
            # A saved PD config has a zero baseline I; restore a valid original
            # baseline for PI/PID rather than blocking unrelated fixed replay.
            self.fields["baseline.ki"][0].set(str(config.get("baseline", {}).get("ki", gains["ki"])))
        for axis in ("kp", "ki", "kd"):
            self.fields[f"fixed_repeat.{axis}"][0].set(f"{gains.get(axis, 0):.8g}")
        if mode in {"p", "pd"}:
            self.fields["fixed_repeat.ki"][0].set("0")
        if mode == "p":
            self.fields["fixed_repeat.kd"][0].set("0")
        self.fields["operation_mode"][0].set("固定参数复测")
        for key in ("duration_ms", "experiment_name", "feedback_unit", "multi_target_worst_weight"):
            if key in config:
                self.fields[key][0].set(str(config[key]))
        self.fields["target_values"][0].set(", ".join(str(value) for value in config.get("target_values", [config.get("target_rad_s", 20)])))
        for key, value in source_rules.items():
            if isinstance(value, bool):
                value = "true" if value else "false"
            choices = CHOICES.get(key, {})
            self.fields[key][0].set(next((label for label, code in choices.items() if code == value), str(value)))
        self.status.set(f"已导入{source}及原测试条件；点击保存或连接并准备后生效，不自动启动")

    def open_run(self):
        if self.current_run:
            if os.name == "nt":
                os.startfile(self.current_run)
            else:
                webbrowser.open(self.current_run.as_uri())

    def open_svg(self):
        if self.current_run:
            path = self.current_run / "speed_response.svg"
            if path.exists():
                webbrowser.open(path.as_uri())

    def browse_run(self):
        path = filedialog.askdirectory(parent=self.root, initialdir=ROOT / "artifacts", title="选择包含manifest.json的运行目录")
        if path:
            self.current_run = Path(path)
            self.load_results(force=True)

    def close(self):
        if self.busy:
            self.closing = True
            self.stop()
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    try:
        MotorTunerApp(root)
        root.mainloop()
    except Exception as exc:
        messagebox.showerror("界面启动失败", str(exc), parent=root)
        root.destroy()
        raise


if __name__ == "__main__":
    main()
