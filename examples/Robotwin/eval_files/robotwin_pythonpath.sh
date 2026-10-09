#!/usr/bin/env bash
# Shared PYTHONPATH for RoboTwin sim eval (policy code + vendored cuRobo).
# Requires ROBOTWIN_ROOT. Optionally SCRIPT_DIR (TCLA eval_files) for deploy modules.
set -euo pipefail

: "${ROBOTWIN_ROOT:?ROBOTWIN_ROOT must be set before sourcing robotwin_pythonpath.sh}"

CUROBO_SRC="${ROBOTWIN_ROOT}/envs/curobo/src"
if [[ ! -f "${CUROBO_SRC}/curobo/__init__.py" ]]; then
  echo "ERROR: vendored cuRobo not found at ${CUROBO_SRC}" >&2
  echo "Expected RoboTwin layout: \${ROBOTWIN_ROOT}/envs/curobo/src/curobo/" >&2
  exit 1
fi

_parts=()
if [[ -n "${SCRIPT_DIR:-}" ]]; then
  _parts+=("${SCRIPT_DIR}")
fi
_parts+=("${ROBOTWIN_ROOT}/policy" "${ROBOTWIN_ROOT}" "${CUROBO_SRC}")
PYTHONPATH="$(IFS=:; echo "${_parts[*]}")${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH CUROBO_SRC

export ROBOTWIN_TASK_EMB_SOURCE="${ROBOTWIN_TASK_EMB_SOURCE:-cache}"
