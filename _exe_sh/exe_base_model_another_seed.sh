#!/usr/bin/env bash
set -euo pipefail

mkdir -p checkpoints runs

LOG_FILE="log_base.txt"

exec > "$LOG_FILE"

for seed in 0 1 2
do

echo "//////////////////////////////////////////"
echo "START LAST LAYER BASELINE"
echo "//////////////////////////////////////////"

PYTHONPATH=. python -u ./scripts/run_classifier.py \
    --use_gpu \
    --fine-tune-mode last-linear-layer \
    --lr 1e-3 \
    --epochs 10 \
    --sst-batch-size 64 \
    --cfimdb-batch-size 8 \
    --hidden-dropout-prob 0.2 \
    --sst-filepath checkpoints/sst.pt \
    --cfimdb-filepath checkpoints/cfimdb.pt \
    --predictions-prefix base-linear-seed$seed- \
    --unuse-schedule \
    --use-simple-classifier \
    --seed $seed

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
    --sst-filepath checkpoints/sst.pt \
    --cfimdb-filepath checkpoints/cfimdb.pt \
    --predictions-prefix base-full-lr1e-5-ep5-seed$seed- \
    --unuse-schedule \
    --use-simple-classifier \
    --seed $seed

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
    --sst-filepath checkpoints/sst.pt \
    --cfimdb-filepath checkpoints/cfimdb.pt \
    --predictions-prefix base-full-lr2e-5-ep5-seed$seed- \
    --unuse-schedule \
    --use-simple-classifier \
    --seed $seed

echo "//////////////////////////////////////////"
echo "ALL BASELINE EXPERIMENTS DONE"
echo "//////////////////////////////////////////"

done