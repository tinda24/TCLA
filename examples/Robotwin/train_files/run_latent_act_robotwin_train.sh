#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$root/scripts/launch_train.sh" "examples/Robotwin/train_files/train_latent_act_robotwin.yaml" tcla.training.train_policy "$@"
