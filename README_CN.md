# TCLA

[English](README.md)

面向 RoboTwin 和 Franka 的动作自编码器，以及 ACT 和 Baku 策略。

## 环境配置

```bash
conda env create -f environment.yaml
conda activate tcla
```

训练脚本需要 Bash。请安装与你的 GPU 相匹配的 PyTorch 版本。
RoboTwin 的仿真依赖需在独立的 RoboTwin 环境中安装。

## 路径配置

所有与本机环境相关的路径统一配置在 `configs/paths.yaml` 中，相对路径均以本项目根目录为基准解析。
本地设置可写入 `configs/paths.local.yaml`，该文件已被排除在版本控制之外，只需填写需要覆盖的配置项：

```yaml
paths:
  data:
    robotwin: data/robotwin
    franka: data/franka
  action_ae:
    robotwin: results/robotwin_ae
    franka: results/franka_ae
  policy_run: results/robotwin_policy
  robotwin: external/RoboTwin
  language_model: sentence-transformers/all-MiniLM-L6-v2
```

可通过环境变量 `TCLA_PATHS_CONFIG` 指定其他本地配置文件。
运行 `python -m tcla.configuration` 可查看解析后的配置。

训练完成后，将 `paths.action_ae.robotwin` 或 `paths.action_ae.franka` 设置为对应 AE 的训练输出目录。
潜动作策略训练会从该目录加载已保存的 AE 配置、检查点和数据集统计量。
`ACTION_AE_STEP` 用于选择检查点的训练步数，默认值为 `160000`。
策略的 token 维度、动作块长度和频域设置均从已保存的 AE 配置中读取。

## 训练

参考 AE 使用 48 个 DCT 频率分量、32 个动作时间步，以及 8 个维度为 16 的潜变量 token，
并配备 EMA 目标编码器和以语言为条件的 Perceiver 进度模型。
模型各模块和损失项沿用参考实现的参数设置，学习率与参考配置一致，保持恒定。

```bash
bash examples/Robotwin_ae/train_files/train_ae.sh 0
bash examples/Robotwin/train_files/run_latent_baku_robotwin_train.sh 0
bash examples/Robotwin/train_files/run_raw_baku_robotwin_train.sh 0
bash examples/Robotwin/train_files/run_latent_act_robotwin_train.sh 0
bash examples/Robotwin/train_files/run_raw_act_robotwin_train.sh 0
```

分布式训练时，使用逗号分隔多个 GPU 编号。并行运行多个训练任务时，请设置 `MAIN_PROCESS_PORT`。
可通过环境变量覆盖的参数包括 `BATCH_SIZE`、`MAX_TRAIN_STEPS`、`LOG_EVERY_STEPS`、
`SAVE_EVERY_STEPS`、`RUN_NAME` 和 `NUM_FREQS`，其中 `NUM_FREQS` 用于 AE 训练。
其他配置项可通过 `--key value` 的形式覆盖：

```bash
bash examples/Robotwin_ae/train_files/train_ae.sh 0 --trainer.max_train_steps 100
```

验证损失保存为 CSV 文件。
检查点包含模型、优化器和学习率调度器的状态。
使用 `--trainer.is_resume true --trainer.resume_checkpoint <checkpoint>` 可恢复训练。

Franka 的数据格式和训练入口详见 [Franka 说明](examples/franka/README.md)。

## 评估

在本地路径配置中设置 `paths.robotwin`、`paths.policy_run` 和 `paths.task_embedding_cache`。
任务嵌入缓存必须与训练时使用的任务嵌入一致。
如果训练数据使用的是 MiniLM 嵌入，也可设置 `ROBOTWIN_TASK_EMB_SOURCE=minilm`。

```bash
bash examples/Robotwin/eval_files/run_eval_latent_baku.sh adjust_bottle 0 steps_160000
```

项目同样提供 raw Baku、latent ACT 和 raw ACT 的评估脚本。
可通过 `ROBOTWIN_EVAL_PYTHON` 指定仿真环境使用的 Python 解释器。

## 验证

```bash
python -m unittest discover -s tests -v
```

这些检查使用合成数据，不会下载模型权重，也不会操作机器人硬件。
