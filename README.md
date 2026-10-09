# TCLA

Action autoencoder v3 with ACT and Baku policies for RoboTwin and Franka.

## Environment

```bash
conda env create -f environment.yaml
conda activate tcla
```

Training scripts require Bash. Install a PyTorch build appropriate for your GPU.
RoboTwin simulation dependencies belong to the separate RoboTwin installation.

## Paths

All machine-specific paths are configured in `configs/paths.yaml`. Relative paths
are resolved against this project directory. For local settings, create
`configs/paths.local.yaml`, which is excluded from version control. It only needs
the values you wish to override:

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

`TCLA_PATHS_CONFIG` selects an alternative local configuration file. Inspect
resolved settings with `python -m tcla.configuration`.

After training an AE, set `paths.action_ae.robotwin` or `paths.action_ae.franka`
to its generated run directory. Latent policy training loads the saved AE config,
checkpoint, and dataset statistics from that directory. `ACTION_AE_STEP` selects
the checkpoint step (default: `160000`). The policy token dimensions, chunk size,
and frequency settings are read from the saved AE configuration.

## Training

The reference AE uses 48 DCT bins, 32 action steps, 8 latent tokens of width 16,
an EMA target encoder, and a language-conditioned Perceiver progress model.
The encoder, decoder, dynamics, and progress losses retain the reference model's
parameterization. Learning rates are constant, as in the reference configuration.

```bash
bash examples/Robotwin_ae/train_files/train_version3.sh 0
bash examples/Robotwin/train_files/run_latent_baku_robotwin_train.sh 0
bash examples/Robotwin/train_files/run_raw_baku_robotwin_train.sh 0
bash examples/Robotwin/train_files/run_latent_act_robotwin_train.sh 0
bash examples/Robotwin/train_files/run_raw_act_robotwin_train.sh 0
```

Use comma-separated GPU IDs for distributed training. Set `MAIN_PROCESS_PORT`
when running concurrent jobs. Environment overrides include `BATCH_SIZE`,
`MAX_TRAIN_STEPS`, `LOG_EVERY_STEPS`, `SAVE_EVERY_STEPS`, `RUN_NAME`, and `NUM_FREQS`
(AE training). Additional configuration overrides use `--key value`:

```bash
bash examples/Robotwin_ae/train_files/train_version3.sh 0 --trainer.max_train_steps 100
```

Validation losses are recorded as CSV; latent trajectories are plotted with t-SNE.
Checkpoints include model, optimizer, and scheduler states. Resume with
`--trainer.is_resume true --trainer.resume_checkpoint <checkpoint>`.

See [Franka](examples/franka/README.md) for its data format and entry points.

## Evaluation

Set `paths.robotwin`, `paths.policy_run`, and `paths.task_embedding_cache` in the
local path configuration. The embedding cache must contain the same task
embeddings used during training. Alternatively, set
`ROBOTWIN_TASK_EMB_SOURCE=minilm` when the training data uses those embeddings.

```bash
bash examples/Robotwin/eval_files/run_eval_latent_baku.sh adjust_bottle 0 steps_160000
```

Equivalent scripts exist for raw Baku, latent ACT, and raw ACT.
`ROBOTWIN_EVAL_PYTHON` can select the simulator's Python interpreter.

## Verification

```bash
python -m unittest discover -s tests -v
```

The checks use synthetic data and do not download model weights or run robot hardware.
