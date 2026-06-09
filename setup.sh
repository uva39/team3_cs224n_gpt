#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! python3 -m venv .venv; then
  echo "Failed to create .venv. On Ubuntu, install python3.10-venv first:"
  echo "  apt-get update && apt-get install -y python3.10-venv"
  exit 1
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Setup complete."
echo "Activate with:"
echo "  source .venv/bin/activate"
echo
echo "Quick checks:"
echo "  python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
echo "  python optimizer_test.py"
