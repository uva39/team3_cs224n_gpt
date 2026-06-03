#!/usr/bin/env bash
set -euo pipefail

mkdir -p checkpoints runs

LOG_FILE="log_Rdrop.txt"

exec > "$LOG_FILE"

for alpha in 0.5 1.0 2.0
do

echo "//////////////////////////////////////////"
echo "START LAST LAYER (rdrop_alpha = $alpha)" 
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
    --predictions-prefix Rdrop-mean-mlp-linear-RdropAlpha$alpha- \
    --max-grad-norm 1.0 \
    --weight-decay 0.01 \
    --pooling-config 'mean' \
    --use-rdrop \
    --rdrop-alpha $alpha
    
#    --unuse-schedule \
#    --use-simple-classifier

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
    --predictions-prefix Rdrop-mean-simple-linear-RdropAlpha$alpha- \
    --max-grad-norm 1.0 \
    --weight-decay 0.01 \
    --pooling-config 'mean' \
    --use-simple-classifier \
    --use-rdrop \
    --rdrop-alpha $alpha

echo "//////////////////////////////////////////"
echo "START FULL MODEL RDROP_1 (rdrop_alpha = $alpha)"
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
    --predictions-prefix Rdrop-mean-mlp-full-lr1e-5-ep5-RdropAlpha$alpha- \
    --max-grad-norm 1.0 \
    --weight-decay 0.01 \
    --pooling-config 'mean' \
    --use-rdrop \
    --rdrop-alpha $alpha

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
    --predictions-prefix Rdrop-mean-simple-full-lr1e-5-ep5-RdropAlpha$alpha- \
    --max-grad-norm 1.0 \
    --weight-decay 0.01 \
    --pooling-config 'mean' \
    --use-simple-classifier \
    --use-rdrop \
    --rdrop-alpha $alpha

echo "//////////////////////////////////////////"
echo "START FULL MODEL RDROP_2 (rdrop_alpha = $alpha)"
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
    --predictions-prefix Rdrop-mean-mlp-full-lr2e-5-ep5-RdropAlpha$alpha- \
    --max-grad-norm 1.0 \
    --weight-decay 0.01 \
    --pooling-config 'mean' \
    --use-rdrop \
    --rdrop-alpha $alpha
#    --use-simple-classifier

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
    --predictions-prefix Rdrop-mean-simple-full-lr2e-5-ep5-RdropAlpha$alpha- \
    --max-grad-norm 1.0 \
    --weight-decay 0.01 \
    --pooling-config 'mean' \
    --use-simple-classifier \
    --use-rdrop \
    --rdrop-alpha $alpha

echo "//////////////////////////////////////////"
echo "ALL EXPERIMENTS DONE (rdrop_alpha = $alpha)"
echo "//////////////////////////////////////////"

done