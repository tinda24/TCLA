#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$root/scripts/launch_train.sh" "examples/Robotwin_ae/train_files/train_version3.yaml" tcla.training.train_action_space_version3 "$@"
