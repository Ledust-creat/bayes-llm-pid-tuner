# 来源与第三方组件

## 上游工程

本项目基于 [KINGSTON-115/llm-pid-tuner](https://github.com/KINGSTON-115/llm-pid-tuner/tree/dev) 的 dev 分支快照扩展。早期仿真清单记载的上游参考提交为 `d885cd659e78234896ce0b851f479f8eab1ef218`；该值是来源记录，不表示后续修改均属于该提交。

上游许可为 MIT，版权声明为 **Copyright (c) 2026 Charles Liang**，完整条款保留在根目录 [LICENSE](LICENSE)。发布时必须同时保留许可和来源说明。

| 文件或目录 | 来源 / 用途 | 本版处理 |
|---|---|---|
| `llm/` | 上游模型客户端、provider、提示词与响应解析 | 保留，以支持混合模式 |
| `core/__init__.py`、`config.py`、`console.py`、`buffer.py`、`history.py`、`i18n.py`、`compat.py` | 上游客户端的直接或包初始化依赖 | 保留，不因文件名通用就删掉 |
| `pid_safety.py` | 上游 `core.config` 的导入依赖 | 保留；本定制版实机增益边界仍以实机JSON为准 |
| `motor_bo_core/`、`motor_bo*.py`、桌面启动脚本 | 围绕本项目增加的优化、试验、通信和界面 | 随本版发布，开发过程使用AI辅助 |
| 旧启动器、MATLAB/PLC入口、Arduino示例与字体 | 上游其他功能 | 不纳入本版选定发布文件，也不删除本机原件 |

本版扩展包括：独立仿真入口、贝叶斯候选生成、受限AI建议、实机文本协议、固定复测、多目标评分、跨批次历史、CSV/SVG与Tkinter界面。实际固件控制环和电机参数由各自MCU工程实现，本仓库不把其控制代码归入上位机原创范围。

## 运行依赖

| 依赖 | 用途 | 分发方式 |
|---|---|---|
| `bayesian-optimization` | 高斯过程与Expected Improvement搜索 | 按requirements从Python包源安装 |
| `requests` | 上游模型客户端的HTTP回退传输 | 按requirements安装 |
| `pyserial` | 串口通信 | 按requirements安装 |
| Python / Tcl-Tk | 运行时与本地桌面界面 | 用户安装，不将虚拟环境打包入源码 |

运行依赖具有各自许可证；本项目的MIT许可不替代它们的许可。当前发布源码与依赖声明，不复制依赖的源码、虚拟环境、上游字体或EXE；将来若分发二进制，应另行核对捆绑组件的许可与通知。

真实API密钥、本机配置、完整试验数据和文章草稿不在本版发布范围内。README中的曲线是本项目历史实测记录的渲染素材，不是新生成的测量结果。
