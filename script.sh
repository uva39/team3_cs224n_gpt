#!/usr/bin/env bash

ENV_NAME="cs224n_dfp"
ENV_FILE="env.yml"
VENV_DIR=".venv"

is_sourced() {
  [[ "${BASH_SOURCE[0]}" != "${0}" ]]
}

say() {
  printf '\n==> %s\n' "$1"
}

warn() {
  printf 'warning: %s\n' "$1" >&2
}

fail() {
  printf 'error: %s\n' "$1" >&2
  return 1
}

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi

  local candidate
  for candidate in \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/anaconda3/bin/conda" \
    "/opt/conda/bin/conda"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

install_with_conda() {
  local conda_bin="$1"
  local conda_base

  conda_base="$("${conda_bin}" info --base)" || return 1
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh" || return 1

  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    say "Found existing conda environment: ${ENV_NAME}"
  else
    say "Creating conda environment: ${ENV_NAME}"
    conda env create -f "${ENV_FILE}" || return 1
  fi

  say "Activating conda environment: ${ENV_NAME}"
  conda activate "${ENV_NAME}" || return 1
}

create_venv() {
  local venv_log
  venv_log="$(mktemp)" || return 1

  python3 -m venv --clear "${VENV_DIR}" >"${venv_log}" 2>&1
  local venv_status=$?
  if [[ "${venv_status}" -ne 0 ]]; then
    cat "${venv_log}" >&2
    rm -f "${venv_log}"
    fail "Could not create ${VENV_DIR}. Install conda, or install python3-venv and run: source script.sh" || return 1
  fi

  rm -f "${venv_log}"
  [[ -f "${VENV_DIR}/bin/activate" ]] || fail "${VENV_DIR}/bin/activate was not created. Install python3-venv, then run: source script.sh" || return 1
}

venv_deps_installed() {
  python - <<'PY' >/dev/null 2>&1
import torch
import transformers
PY
}

install_with_venv() {
  command -v python3 >/dev/null 2>&1 || fail "Neither conda nor python3 was found. Install one of them, then run: source script.sh" || return 1

  warn "conda was not found; using Python venv fallback at ${VENV_DIR}. env.yml asks for Python 3.8, so conda is preferred."

  if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    say "Creating virtual environment: ${VENV_DIR}"
    create_venv || return 1
  fi

  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate" || return 1

  if venv_deps_installed; then
    say "Python dependencies already installed"
    return 0
  fi

  say "Installing pip dependencies from ${ENV_FILE}"
  python -m pip install --upgrade pip || return 1

  local req_file
  req_file="$(mktemp)" || return 1
  awk '/^[[:space:]]+- pip:/{in_pip=1; next} in_pip && /^[[:space:]]+- /{sub(/^[[:space:]]+- /, ""); print}' "${ENV_FILE}" > "${req_file}" || return 1
  python -m pip install -r "${req_file}"
  local pip_status=$?
  rm -f "${req_file}"
  return "${pip_status}"
}

print_checks() {
  say "Checking required commands"
  python --version || return 1
  python -m pip --version || return 1
  python - <<'PY'
import torch
import transformers
print(f"torch {torch.__version__}")
print(f"transformers {transformers.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
PY
}

print_project_commands() {
  say "Project commands"
  cat <<'CMDS'
Run these after setup:

  source script.sh
  python optimizer_test.py
  python sanity_check.py
  python classifier.py
  python paraphrase_detection.py --use_gpu
  python sonnet_generation.py --use_gpu

If you are on CPU, omit --use_gpu.
CMDS

  if ! is_sourced || [[ "$(basename "$0")" == "setup.sh" ]]; then
    cat <<'NOTE'

Note: this script was executed in a child shell.
To keep the environment active in your current terminal, run:

  source script.sh
NOTE
  fi
}

main() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1
  cd "${script_dir}" || return 1

  [[ -f "${ENV_FILE}" ]] || fail "${ENV_FILE} not found in ${script_dir}" || return 1

  local conda_bin
  conda_bin="$(find_conda || true)"
  if [[ -n "${conda_bin}" ]]; then
    install_with_conda "${conda_bin}" || return 1
  else
    install_with_venv || return 1
  fi

  print_checks || return 1
  print_project_commands
}

main "$@"
SCRIPT_STATUS=$?

if is_sourced; then
  return "${SCRIPT_STATUS}"
fi
exit "${SCRIPT_STATUS}"
