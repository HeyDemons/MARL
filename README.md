# MARL 作业合集（Multi-Agent Reinforcement Learning)

本仓库整理了从单智能体到多智能体的一系列强化学习作业与项目，涵盖 DQN、Policy Gradient、REINFORCE、DDPG、A2C、MAPPO、QMIX、MADDPG/MATD3/MASAC 以及贪吃蛇等应用。每个子目录基本自洽，含训练、运行与绘图脚本。


## 环境与依赖
- Python 3.8–3.11（建议使用虚拟环境/conda）
- PyTorch、NumPy、Matplotlib、tqdm、TensorBoard
- Gym/Gymnasium 及相关环境（例如 CarRacing 需 box2d 依赖）
- PettingZoo + SuperSuit（多智能体 MPE 任务）
- Pygame（贪吃蛇项目）

按需安装：
- 贪吃蛇项目：`cd snake-ai-2 && pip install -r requirements.txt`
- 多智能体 MPE：`pip install "pettingzoo[mpe]" supersuit`
- 其他常见依赖：`pip install torch numpy matplotlib tqdm tensorboard`


## 目录概览
- `hw1`：入门/笔记本
- `hw2`：DQN 与 Policy Gradient（CartPole 等）
- `hw3`：CarRacing DQN（PyTorch）
- `hw4`：REINFORCE（CartPole）
- `hw5`：DDPG（CarRacing），含曲线与视频
- `hw6`：A2C
- `hw7`：MAPPO（PettingZoo MPE）
- `hw8`：QMIX（值分解）
- `hw9` / `hw10`：多智能体算法实现与实验（MADDPG/MATD3/MASAC 等）
- `hw910_and_pr1` / `pr1`：项目延伸与复现实验
- `pr2`：CNN/结构化模型与训练脚本
- `snake-ai-2`：贪吃蛇环境与智能体（含训练与测试）
- `models`：模型文件

每个子目录通常包含：
- `train.py`：训练入口
- `run.py`/`test.py`：评估或对局演示
- `plot.py`/`plots`：绘图脚本与结果
- `runs`/`logs`：TensorBoard 与 Checkpoint 日志
- `videos`/`gif`：可视化视频/动图


## 快速开始
1) 创建环境（示例）
   - conda: `conda create -n marl python=3.10 && conda activate marl`
   - 或 venv: `python -m venv .venv && source .venv/bin/activate`
2) 安装依赖（按需）
   - 贪吃蛇：`cd snake-ai-2 && pip install -r requirements.txt`
   - 多智能体：`pip install "pettingzoo[mpe]" supersuit`
   - 可视化：`pip install tensorboard`


## 运行示例
- 基础任务（以 `hw2` 为例）：
  - 训练：`python hw2/main.py`
  - 或单文件训练：`python hw2/dqn.py` / `python hw2/policy_gradient.py`
- CarRacing（`hw3`/`hw5`）：
  - `python hw3/dqn.py` 或 `python hw3/car_racing_dqn.py`
  - `python hw5/train_ddpg.py`
- 多智能体（`hw7`/`hw8`/`hw9`/`hw10`）：
  - `python hw7/train.py`
  - `python hw8/main.py`
  - `python hw9/train.py` 或 `python hw9/run.py`
  - `python hw10/train.py` 或 `python hw10/run.py`
- 贪吃蛇（`snake-ai-2`）：
  - 训练：`python snake-ai-2/train.py`
  - 演示：`python snake-ai-2/main.py`

提示：多数脚本支持命令行参数，使用 `-h/--help` 查看可选项。


## 日志与可视化
- 训练日志与模型：默认保存在各子目录的 `runs/`、`logs/`、`outputs/` 中。
- 曲线绘制：执行对应 `plot.py` 或在 `plots/` 查看生成图像。
- TensorBoard：`tensorboard --logdir hw*/runs`（或进入具体子目录查看）
- 视频/动图：在 `videos/`、`gif/` 下查看评估/演示结果。


## 复现实验与排错
- 固定随机种子（若脚本支持 `--seed`）。
- 环境版本不兼容：优先使用与代码匹配的 Gym/Gymnasium 与 PettingZoo 版本；CarRacing 可能需要安装 Box2D 依赖。
- MPE 任务无法创建：确认已安装 `pettingzoo[mpe]` 与 `supersuit`，并参考 `hw7/hw9/hw10` 的 `utils/env.py` 配置。
- 若显存不足：降低网络规模、批大小或关闭视频记录。


## 许可
仅用于课程作业与学习复现用途，数据与第三方依赖遵循其各自许可协议。
