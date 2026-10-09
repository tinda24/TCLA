#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_yaml="${1:?config_yaml required}"
module="${2:?training module required}"
shift 2
gpu_ids="${GPU_IDS:-0}"
if [[ $# -gt 0 && "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  gpu_ids="$1"
  shift
fi
IFS=',' read -r -a gpus <<< "$gpu_ids"
launch_args=(--num_processes "${#gpus[@]}" --main_process_port "${MAIN_PROCESS_PORT:-29500}")
if [[ ${#gpus[@]} -gt 1 ]]; then
  launch_args+=(--multi_gpu)
fi
overrides=()
[[ -z "${RUN_NAME:-}" ]] || overrides+=(--run_name "$RUN_NAME")
[[ -z "${BATCH_SIZE:-}" ]] || overrides+=(--datasets.action_data.per_device_batch_size "$BATCH_SIZE")
[[ -z "${MAX_TRAIN_STEPS:-}" ]] || overrides+=(--trainer.max_train_steps "$MAX_TRAIN_STEPS")
[[ -z "${LOG_EVERY_STEPS:-}" ]] || overrides+=(--trainer.log_every_steps "$LOG_EVERY_STEPS")
[[ -z "${SAVE_EVERY_STEPS:-}" ]] || overrides+=(--trainer.save_every_steps "$SAVE_EVERY_STEPS")
if [[ "$module" == tcla.training.train_action_space ]]; then
  [[ -z "${NUM_FREQS:-}" ]] || overrides+=(--num_freqs "$NUM_FREQS")
fi
cd "$root"
CUDA_VISIBLE_DEVICES="$gpu_ids" "${PYTHON:-python}" -m accelerate.commands.launch \
  "${launch_args[@]}" -m "$module" --config_yaml "$config_yaml" "${overrides[@]}" "$@"
