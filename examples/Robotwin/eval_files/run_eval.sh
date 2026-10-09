#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TCLA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$TCLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
path_exports="$("${ROBOTWIN_EVAL_PYTHON:-${PYTHON:-python}}" -m tcla.configuration --shell)"
eval "$path_exports"
: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to the installed RoboTwin directory}"
: "${RUN_DIR:?Set RUN_DIR to the policy training output directory}"
config="${1:?deployment config required}"
shift
task="${1:-adjust_bottle}"
gpu="${2:-0}"
round="${3:-steps_160000}"
export CUDA_VISIBLE_DEVICES="$gpu"
export ROBOTWIN_ROOT
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
ckpt="${CKPT_PATH:-$RUN_DIR/checkpoints/${round}_pytorch_model.pt}"
[[ -f "$ckpt" ]] || { echo "Checkpoint not found: $ckpt" >&2; exit 1; }
source "$SCRIPT_DIR/robotwin_pythonpath.sh"
source "$SCRIPT_DIR/resolve_robotwin_python.sh"
cd "$ROBOTWIN_ROOT"
"$ROBOTWIN_EVAL_PYTHON" "$SCRIPT_DIR/launch_eval.py" \
  --config "$SCRIPT_DIR/$config" --overrides \
  --task_name "$task" --task_config "${TASK_CONFIG:-demo_clean}" \
  --ckpt_setting "$round" --round "$round" --seed "${SEED:-0}" \
  --device cuda:0 --temporal_agg "${TEMPORAL_AGG:-true}" \
  --stage2_root "${STAGE2_ROOT:-$RUN_DIR/eval_results}" \
  --test_num "${TEST_NUM:-100}" --config_yaml "$RUN_DIR/config.yaml" \
  --ckpt_path "$ckpt" --run_dir "$RUN_DIR" --save_phase_plot "${SAVE_PHASE_PLOT:-true}"
