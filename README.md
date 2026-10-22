# PID 自动整定台 · Bayesian + LLM PID Tuner

基于贝叶斯优化的 PID 调参上位机，支持 PC 仿真、串口实机试验和固定参数复测。可选大模型按间隔分析曲线与指标；PID 候选由贝叶斯优化器生成，AI 不直接下发增益。

项目仓库：[Ledust-creat/bayes-llm-pid-tuner](https://github.com/Ledust-creat/bayes-llm-pid-tuner)。

本项目基于 [KINGSTON-115/llm-pid-tuner 的 dev 分支](https://github.com/KINGSTON-115/llm-pid-tuner/tree/dev)扩展，保留必要的上游大模型模块和 MIT 许可。新增功能围绕 STM32 FOC 调试形成，但串口层使用通用的目标与反馈字段。来源与依赖见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 功能

| 功能 | 行为 | 使用场景 |
|---|---|---|
| 纯贝叶斯 / 贝叶斯 + AI | 搜索候选；可按间隔咨询中文建议 | 自动试验与结果分析 |
| P / PI / PD / PID | 按所选维度搜索；P固定I/D=0，PD固定I=0 | 下位机必须实际支持对应参数 |
| 固定参数复测 | 同一参数重复运行，不搜索、不调用AI | 检查候选的重复性 |
| 多目标与历史 | 一组参数依次测试1～10个目标；支持跨批次续调 | 比较多个工作点，减少从头试起 |
| 本地图形界面 | 配置、曲线筛选、评分表、日志与中文建议 | Windows / Tkinter，无网页服务 |
| CSV / SVG记录 | 保留原始遥测、各目标指标和汇总曲线 | 复核试验与导入最佳参数复测 |

![历史速度环五轮多目标固定复测：20、50、80 rad/s](images/speed-validation.svg)

上图是已保存的电机速度环记录。串口目标的单位、实际控制环和允许范围由固件决定；在界面中改名称不会切换固件，也不会自动换算单位。

## 首次安装与启动

建议在 Windows 上使用包含 Tcl/Tk 的 Python。本项目在 Python 3.13.6 环境验证；源码使用 Python 3.10+ 语法。当前提供源码，不把上游 Release 中的 EXE 当成本定制版。

在本目录打开 PowerShell：

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-motor-bo.txt
.\venv\Scripts\python.exe motor_bo.py
```

最后一行只运行 PC 仿真，不打开串口、不请求真实模型。结果保存在 `artifacts/motor_bo/`，先检查这一流程再连接硬件。

首次准备实机配置时，从示例复制一份（已有配置不要覆盖）：

```powershell
Copy-Item motor_bo_hw_config.example.json motor_bo_hw_config.json
.\venv\Scripts\python.exe motor_bo_gui.py
```

之后可以双击 `start_motor_gui.bat`。它使用项目内 `venv` 启动同一界面，旧的 `motor_bo_*` 文件名保留兼容，不要求本地目录名称与 GitHub 仓库名称一致。界面启动不会打开串口；配置好后点击“连接并准备”进行 PING，再点击“确认开始”才执行试验。

### 要改哪些配置

| 文件 | 用途 | 是否上传 |
|---|---|---|
| `motor_bo_config.json` | PC 仿真模型、搜索轮数与模式 | 是 |
| `motor_bo_hw_config.example.json` | 通用串口速度跟踪结构示例 | 是 |
| `motor_bo_position_config.example.json` | 连续绝对位置结构示例，P默认0，需填写可运行基线 | 是 |
| `motor_bo_hw_config.json` | 复制示例后使用的本机实机配置 | 否 |
| `motor_bo_local.example.json` | 无密钥的 API 示例 | 是 |
| `motor_bo_local.json` | 本机 API 地址、模型和密钥 | 否 |

实机先填写 `serial.port`，把 `YOUR_COM_PORT` 改为实际端口；核对 `ranges`、`protocol_limits`、目标、时长和反馈截止值，关闭占用串口的其他工具。示例参数不作为任意电机的安全默认值；首次试验需限流供电、可操作的断电手段和固件本地保护。

位置环先根据真实固件准备本机配置。当前配套 FOC 固件使用连续绝对位置、度数传输，位置误差直接相减，因此 `cyclic_period=0`；目标允许范围约为±360°不表示720°周期。周期最短差对象才设置非零周期。改变单位、评分或试验条件后建立新历史。

### 三种运行方式

| 目标 | 设置 | 行为 |
|---|---|---|
| 纯贝叶斯搜索 | `operation_mode=bayes`，`run_mode=bayes_only` | 不调用AI |
| 贝叶斯 + AI | `operation_mode=bayes`，`run_mode=bayes_llm` | 初始试验后按`llm_interval_rounds`咨询；正常批末另有范围建议 |
| 固定复测 | `operation_mode=fixed_repeat` | 使用`fixed_repeat`参数，不调用AI或优化器 |

`rounds_per_batch`是搜索总轮数，`initial_random_trials`是首批随机参数组数，AI间隔不是总轮数。`history_mode=new`建立新系列，`continue`承接同条件历史；`--resume`只补完AI暂停的当前批次。三个目标、二十轮最多执行六十次目标试验，每次结束均请求STOP。

需要 AI 时，首次复制 `motor_bo_local.example.json` 为 `motor_bo_local.json`，填写自己的地址、模型和密钥；纯贝叶斯与固定复测不需要它。`MOTOR_BO_*` 环境变量优先于该文件。不要提交真实配置或把密钥贴到日志、截图和 issue。

## 串口与评分

当前协议为115200、8N1、ASCII、逗号分隔、CRLF结尾，无CRC、序号和NACK。命令为`PING`、`SET_PID`、`RUN_STEP`、`STOP`，成功回复`OK`，遥测为`DATA,elapsed_ms,target,feedback`。拒绝命令可能静默无应答，上位机等待超时，不自动重发。完整约定见 [串口协议](docs/PROTOCOL.md)。

实机单目标评分：70%归一化绝对误差积分、20%超调、10%调节时间占比；默认多目标综合分为70%平均分加30%最差目标分，越低越好。仿真另含电流限幅占用，分数不与实机直接比较。复测应维持同样的起点、负载和试验定义。

## 测试、目录与文档

配置文件准备好后，可运行离线测试；测试使用假串口和模拟建议，不连接真实设备：

```powershell
.\venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p 'test_motor_bo*.py'
```

| 入口或目录 | 内容 |
|---|---|
| `motor_bo_gui.py` / `start_motor_gui.bat` | 桌面界面 |
| `motor_bo_hw.py` / `motor_bo_link.py` | 实机试验 / 仅PING的链路检查 |
| `motor_bo.py` / `motor_bo_core/` | 仿真入口 / 优化、试验、评分与通信模块 |
| `llm/` / `core/` / `pid_safety.py` | 保留的上游客户端及其导入依赖 |
| `tests/` / `docs/` | 当前工具的离线测试与使用说明 |

- [桌面界面操作](docs/motor_bo_gui.md)
- [仿真、实机与历史说明](docs/zh-CN/MOTOR_BO.md)
- [独立仓库发布检查](docs/PUBLISHING.md)
- 配套 MCU 工程：[stm32f103-foc-mt6701](https://github.com/Ledust-creat/stm32f103-foc-mt6701)

本次选定发布的是定制上位机，不包含上游旧启动器、Arduino示例、MATLAB/PLC入口和字体资源。`.gitignore`只放行所需文件；这些旧文件如仍在开发者本机，不等于属于本版发布功能。保留上游版权和 [MIT License](LICENSE)，二次开发及调试过程使用了 AI 辅助。
