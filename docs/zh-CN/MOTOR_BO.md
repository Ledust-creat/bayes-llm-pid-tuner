# 电机速度环：大模型 + 贝叶斯优化（纯 PC 仿真）

这是独立于原项目 `tuner.py` 的实验入口。它**不会访问串口或 STM32**；现阶段得到的 `Kp/Ki/Kd` 只对这里的示例模型有效，不能直接用于实机。源码基于上游 `dev` 快照 `d885cd659e78234896ce0b851f479f8eab1ef218`。

## Windows 上运行

在项目根目录打开 PowerShell，使用 Python 3.10 或更新版本；本次已用 Python 3.13.6 验证，无需更新系统 Python：

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-motor-bo.txt
.\venv\Scripts\python.exe motor_bo.py
```

`motor_bo_config.json` 的 `run_mode` 选择 `bayes_only` 或 `bayes_llm`，默认纯贝叶斯；`llm_interval_rounds` 默认是 5。默认运行基线 + 4 个初始点 + 15 个贝叶斯点，只搜索 `Kp/Ki`，`Kd` 始终为 0；`--tune-d` 改为基线 + 6 + 24 点，同时搜索 `Kd`。

```powershell
.\venv\Scripts\python.exe motor_bo.py --tune-d
.\venv\Scripts\python.exe -m unittest tests.test_motor_bo
```

每轮都是从静止开始的 `0→10 rad/s`、2 秒仿真，速度 PI 以 930 Hz 更新，内层电流仅用固定一阶滞后近似，`Iq_ref` 被限制在 ±1.55 A。示例惯量、阻尼、电流响应及增益范围位于 `motor_bo_config.json`，不是实机辨识结果。D 项对测量速度求导并经一阶滤波；积分具备输出限幅时的抗累积处理。

贝叶斯优化负责提出增益；大模型只能从 `continue/pause/stop` 和 `explore/balanced/exploit` 中选择。`stop` 在初始试验结束前无效；模型返回错误、超时或试图给出增益时，程序采用 `continue + balanced`。固定评分越低越好：归一化绝对误差 55%、超调 20%、±5% 稳定时间 15%、电流限幅占用 10%。这些权重不会随大模型回复变化。

## 接大模型（可选）

纯贝叶斯模式不请求模型。混合模式在完成基线和初始样本后按间隔咨询 AI；默认 20 轮会在第 5、10、15 轮请求。AI 理由会要求简体中文，并显示在终端和运行目录的 `ai_advice.md`；机器记录仍保存在 `rounds.jsonl`。使用真实模型时，在项目根目录的 `motor_bo_local.json` 文本文件里填写地址、模型 ID 和密钥：

```powershell
$env:MOTOR_BO_API_KEY = '<your-key>'
$env:MOTOR_BO_BASE_URL = 'https://your-compatible-endpoint/v1'
$env:MOTOR_BO_MODEL = 'your-model'
.\venv\Scripts\python.exe motor_bo.py --llm live
```

中转站示例使用 OpenAI 兼容接口，所以 `provider` 填 `openai`；`model` 必须与密钥允许调用的模型 ID 完全一致。`motor_bo_local.json` 已加入 `.gitignore`，避免密钥被 Git 提交。运行 `motor_bo.py` 前确认 `run_mode` 是所需模式；也可用 `--llm live` 或 `--llm mock` 临时覆盖文本配置。

## 结果与恢复

每次运行输出到 `artifacts/motor_bo/<时间戳>/`：`manifest.json` 记录实验配置、模式和 AI 调用次数，`rounds.jsonl` 记录候选、指标和模型回复，`ai_advice.md` 汇总中文建议，`traces/round_*.csv` 保存完整曲线，`best.json` 保存最低评分，直接用浏览器打开 `response.svg` 可比较基线和最优速度曲线。模型选择 `pause` 时可继续同一次运行；重放历史模型回复则新建运行目录：

```powershell
.\venv\Scripts\python.exe motor_bo.py --resume --run-dir artifacts/motor_bo/<原目录名>
.\venv\Scripts\python.exe motor_bo.py --llm replay --replay-from artifacts/motor_bo/<原目录名>
```

同一随机种子、配置、依赖版本和大模型回复记录用于复现实验。仿真结果只适用于示例模型，不能直接作为实机 PID 参数。

## STM32 串口链路冒烟测试

`motor_bo_link.py` 是独立的串口冒烟工具，只发送 `PING\r\n` 并等待固件回复 `OK\r\n`，不会设置 PID、启动 PWM 或运行电机试验。当前协议使用 USART2、115200、8N1、ASCII 逗号分隔和 CRLF 行尾，不带 CRC 或序号。先在 `motor_bo_hw_config.json` 的 `serial.port` 填 COM 口，关闭占用该串口的 VOFA/终端，再运行：

```powershell
.\venv\Scripts\python.exe motor_bo_link.py
```

离线命令编码、遥测解析和串口假设备测试：

```powershell
.\venv\Scripts\python.exe -m unittest tests.test_motor_bo_serial
```

PC 与 STM32 的命令、应答、静默拒绝行为及通用 `DATA,elapsed_ms,target,feedback` 格式见[串口协议](../PROTOCOL.md)。收到 `RUN_STEP` 的 `OK` 后开始接收遥测；固件到时关闭 PWM，不发送额外完成消息。

## 实机 PID 贝叶斯调参

`motor_bo_hw.py` 是独立的实机入口，不影响仿真 `motor_bo.py`。它设置PID、执行阶跃、读取通用 `DATA` 并按误差/超调/稳定时间评分；支持P、PI、PD、PID搜索，实际控制环由固件决定。每次结束发送STOP，主机绝对反馈截止依赖串口遥测，不能替代下位机保护。

实机用户参数集中在 `motor_bo_hw_config.json`：`serial` 配串口，`operation_mode` 为 `bayes` 或 `fixed_repeat`；`target_values`、`duration_ms`、`feedback_abs_limit` 配目标、时长和绝对反馈截止。`protocol_limits` 配目标上下界及端点包含规则，`response` 配评分尺度/坐标周期，GUI全部可编辑。固定复测不调用AI或贝叶斯，评分与CSV、`summary.json`、曲线保存到 `artifacts/motor_bo_validation/<时间戳>/`，不修改调参历史。API密钥仍在 `motor_bo_local.json`。

`ranges` 是发送增益和AI推荐都不得越过的边界；协议范围必须与固件一致，修改JSON不会修改固件。目前位置固件以度传目标和反馈，目标范围 `[-360,360)`、时长上限5000ms；逻辑位置连续累计、误差直接相减，`cyclic_period=0`，目标范围不是720度周期。位置模式关闭 `trial_guard.enabled`，避免把负角和零位当作反转/停转；绝对反馈截止、通信和STOP检查仍开启。旧速度配置缺少新字段时保留原正目标行为。具体配置与评分说明见 [桌面使用说明](../motor_bo_gui.md#位置环配置与评分)。

`run_mode` 选择 `bayes_only` 或 `bayes_llm`；公开示例默认纯贝叶斯，混合模式复用 `motor_bo_local.json` 的模型/API 配置。`llm_interval_rounds` 控制批内咨询频率：新 PID 在基线与随机初始样本完成后首次咨询，后续按间隔咨询，不在末轮请求批内建议。每批**正常完成**后，混合模式还会额外请求一次下一批搜索范围；建议必须位于 `ranges` 内、每轴宽度至少为硬边界的 20%、包含历史最佳点，否则沿用本批范围。纯贝叶斯不调用 AI，下一批仍使用当前范围，但继续复用历史数据。AI 不直接给下一轮 PID 增益；候选始终由贝叶斯优化器生成。

跨批次继续时保持 `pid_id` 不变，`history_mode=continue`复用完成历史而不重复基线/随机初始点；`new`建立新系列且旧记录保留。历史位于 `artifacts/motor_bo_hardware/pid_profiles/<pid_id>/<系列ID>.json`，`active.json`指向当前系列。`pid_id`只是历史标签，不切换固件。续调必须保持目标、时长、单位、搜索模式、评分尺度/周期与异常规则一致；电源、负载、滤波、固件或起点条件变化也应新建历史。模型最多拟合最近100点并补入全局最佳；人工硬边界内的历史可参与拟合，即使不在下一批建议的子范围内。下一组候选仍由当前搜索范围约束。

在 `motor_bo_hw_config.json` 的 `serial.port` 配好 COM 口、关闭 VOFA/串口终端后运行：

```powershell
.\venv\Scripts\python.exe motor_bo_hw.py
```

程序先PING并显示参数，人工输入 `START` 才会运行。结果保存到 `artifacts/motor_bo_hardware/<时间戳>/`，含PID、评分、CSV、最优结果和配置快照；兼容文件名 `speed_response.svg` 实际绘制对应单位的反馈曲线。固件决定调哪个环路，不会自动搜索级联多环。首帧作为位置起点只是估计，并非阶跃前采样；起点已在目标时属于保持试验，不能据此证明阶跃性能。协议未实现回零，复测须保持起点一致。异常时尝试STOP，但仍需独立限流和可操作的断电手段。

若 AI 建议暂停，当前轮已完成 STOP 后，清单状态记为 `paused`；程序提示恢复命令。恢复时会重新 PING，并再次要求人工输入 `START`，从下一轮继续，不重复已记录轮次：

```powershell
.\venv\Scripts\python.exe motor_bo_hw.py --resume --run-dir "artifacts/motor_bo_hardware/<运行目录>"
```

AI 建议停止则状态为 `stopped`，不能恢复；超速、遥测无效或 STOP 未确认会标记为中止，不提供自动恢复，也不把该批提交为完成历史。`--resume` 是补完**当前暂停批次**，`history_mode=continue` 是开启**下一批**，两者不可混用。每批运行目录含 `manifest.json`（配置、系列 ID、历史数、下一批范围和状态）、`rounds.jsonl`、`ai_advice.md`、`optimizer.json`、`traces/round_*.csv` 与 `speed_response.svg`。历史最佳如果贴着硬边界，AI 缩范围也无法搜索边界之外；只有人工确认固件允许并修改硬边界后才能尝试更低/更高增益，且不保证一定优于原最佳。
