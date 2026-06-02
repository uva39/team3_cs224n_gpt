#!/usr/bin/env bash
set -euo pipefail

mkdir -p checkpoints runs

LOG_FILE="log_no_Rdrop_mlp.txt"

exec > "$LOG_FILE"

echo "//////////////////////////////////////////"
echo "START LAST LAYER NO_RDROP"
echo "//////////////////////////////////////////"

PYTHONPATH=. python -u ./scripts/run_classifier.py \
    --use_gpu \
    --fine-tune-mode last-linear-layer \
    --lr 1e-3 \
    --epochs 10 \
    --sst-batch-size 64 \
    --cfimdb-batch-size 8 \
    --hidden-dropout-prob 0.2 \
    --sst-filepath checkpoints/no-Rdrop-linear-sst.pt \
    --cfimdb-filepath checkpoints/no-Rdrop-linear-cfimdb.pt \
    --predictions-prefix no-Rdrop-mlp-linear- \
    --max-grad-norm 1.0 \
    --weight-decay 0.01 
#    --unuse-schedule \
#    --use-simple-classifier

echo "//////////////////////////////////////////"
echo "START FULL MODEL NO_RDROP_1"
echo "//////////////////////////////////////////"

PYTHONPATH=. python -u ./scripts/run_classifier.py \
    --use_gpu \
    --fine-tune-mode full-model \
    --lr 1e-5 \
    --epochs 5 \
    --sst-batch-size 64 \
    --cfimdb-batch-size 8 \
    --hidden-dropout-prob 0.2 \
    --sst-filepath checkpoints/no-Rdrop-full-lr1e-5-ep5-sst.pt \
    --cfimdb-filepath checkpoints/no-Rdrop-full-lr1e-5-ep5-cfimdb.pt \
    --predictions-prefix no-Rdrop-mlp-full-lr1e-5-ep5- \
    --max-grad-norm 1.0 \
    --weight-decay 0.01 
#    --unuse-schedule \
#    --use-simple-classifier

echo "//////////////////////////////////////////"
echo "START FULL MODEL NO_RDROP_2"
echo "//////////////////////////////////////////"

PYTHONPATH=. python -u ./scripts/run_classifier.py \
    --use_gpu \
    --fine-tune-mode full-model \
    --lr 2e-5 \
    --epochs 5 \
    --sst-batch-size 64 \
    --cfimdb-batch-size 8 \
    --hidden-dropout-prob 0.2 \
    --sst-filepath checkpoints/no-Rdrop-full-lr2e-5-ep5-sst.pt \
    --cfimdb-filepath checkpoints/no-Rdrop-full-lr2e-5-ep5-cfimdb.pt \
    --predictions-prefix no-Rdrop-mlp-full-lr2e-5-ep5- \
    --max-grad-norm 1.0 \
    --weight-decay 0.01 
#    --unuse-schedule \
#    --use-simple-classifier

echo "//////////////////////////////////////////"
echo "ALL EXPERIMENTS DONE"
echo "//////////////////////////////////////////"