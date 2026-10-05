#!/usr/bin/env bash
# Verify an existing Codex interpreter without installing or selecting Python.
set -euo pipefail

target_file=${1:-.python-version}
python_target=$(tr -d '\r\n' < "$target_file")
if [[ ! "$python_target" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  printf '::error::Invalid Python target in %s: %s\n' "$target_file" "$python_target"
  exit 1
fi

printf 'Project Python target: %s\n' "$python_target"
printf 'Default Python on PATH: '
python3 --version || true

if [[ -n ${CODEX_PRIMARY_RUNTIME_PYTHON:-} ]]; then
  target_python=$CODEX_PRIMARY_RUNTIME_PYTHON
  printf 'Checking supplied Codex primary runtime: %s\n' "$target_python"
else
  # Inspect the reference image's preinstalled versions directly. Do not run
  # its entrypoint, pyenv install/global, uv or actions/setup-python.
  versions_dir=${PYENV_ROOT:-/root/.pyenv}/versions
  target_python=$versions_dir/$python_target/bin/python
  if [[ ! -x "$target_python" ]]; then
    printf '::error::Python %s is not preinstalled in the Codex reference image.\n' "$python_target"
    printf 'Preinstalled versions:\n'
    if [[ -d "$versions_dir" ]]; then
      find "$versions_dir" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort
    fi
    exit 1
  fi
fi

actual_version=$("$target_python" -c 'import platform; print(platform.python_version())')
printf 'Observed preinstalled Python: %s\n' "$actual_version"
if [[ "$actual_version" != "$python_target" ]]; then
  printf '::error::Codex supplies Python %s; project target is %s.\n' "$actual_version" "$python_target"
  exit 1
fi
