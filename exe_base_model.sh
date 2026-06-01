#!/usr/bin/env bash
set -euo pipefail

mkdir -p checkpoints runs

LOG_FILE="log.txt"

exec > "$LOG_FILE"

#echo "//////////////////////////////////////////"
#echo "START LAST LAYER BASELINE"
#echo "//////////////////////////////////////////"

#PYTHONPATH=. python -u ./scripts/run_classifier.py \
#    --use_gpu \
#    --fine-tune-mode last-linear-layer \
#    --lr 1e-3 \
#    --epochs 10 \
#    --sst-batch-size 64 \
#    --cfimdb-batch-size 8 \
#    --hidden-dropout-prob 0.2 \
#    --sst-filepath checkpoints/base-linear-sst.pt \
#    --cfimdb-filepath checkpoints/base-linear-cfimdb.pt \
#    --predictions-prefix base-linear-

echo "//////////////////////////////////////////"
echo "START FULL MODEL BASELINE_1"
echo "//////////////////////////////////////////"

PYTHONPATH=. python -u ./scripts/run_classifier.py \
    --use_gpu \
    --fine-tune-mode full-model \
    --lr 1e-5 \
    --epochs 5 \
    --sst-batch-size 64 \
    --cfimdb-batch-size 8 \
    --hidden-dropout-prob 0.2 \
    --sst-filepath checkpoints/base-full-lr1e-5-ep5-sst.pt \
    --cfimdb-filepath checkpoints/base-full-lr1e-5-ep5-cfimdb.pt \
    --predictions-prefix base-full-lr1e-5-ep5-

echo "//////////////////////////////////////////"
echo "START FULL MODEL BASELINE_2"
echo "//////////////////////////////////////////"

PYTHONPATH=. python -u ./scripts/run_classifier.py \
    --use_gpu \
    --fine-tune-mode full-model \
    --lr 2e-5 \
    --epochs 5 \
    --sst-batch-size 64 \
    --cfimdb-batch-size 8 \
    --hidden-dropout-prob 0.2 \
    --sst-filepath checkpoints/base-full-lr2e-5-ep5-sst.pt \
    --cfimdb-filepath checkpoints/base-full-lr2e-5-ep5-cfimdb.pt \
    --predictions-prefix base-full-lr2e-5-ep5-

echo "//////////////////////////////////////////"
echo "ALL BASELINE EXPERIMENTS DONE"
echo "//////////////////////////////////////////"