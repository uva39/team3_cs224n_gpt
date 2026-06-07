#!/usr/bin/env bash
set -euo pipefail

mkdir -p checkpoints runs predictions logs

LOG_FILE="logs/log_targeted_sweep_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee "$LOG_FILE") 2>&1

# 필요하면 여기만 바꿔서 일부만 실행 가능
# 예:
#   RUN_SST=1 RUN_CFIMDB=0 bash exe_targeted_sweep.sh
#   RUN_SST=0 RUN_CFIMDB=1 bash exe_targeted_sweep.sh
RUN_SST="${RUN_SST:-1}"
RUN_CFIMDB="${RUN_CFIMDB:-1}"

SEEDS=(0 1 2)

# 네 환경에서 SST full-model batch 64가 OOM 나면 16 또는 8로 낮추면 됨
SST_BS=64
CFIMDB_BS=4

COMMON_ARGS=(
  --use_gpu
  --fine-tune-mode full-model
  --sst-batch-size "$SST_BS"
  --cfimdb-batch-size "$CFIMDB_BS"
)

sanitize_float() {
  # 파일명에 들어가기 좋게 2e-5, 0.75 등을 변환
  echo "$1" | sed 's/\./p/g' | sed 's/-/m/g'
}

run_sst_mlp_no_rdrop() {
  local seed="$1"
  local lr="$2"
  local dropout="$3"
  local exp_name="$4"

  local lr_tag
  local drop_tag
  lr_tag="$(sanitize_float "$lr")"
  drop_tag="$(sanitize_float "$dropout")"

  local prefix="sweep-SST-${exp_name}-seed${seed}-lr${lr_tag}-drop${drop_tag}-"

  echo "============================================================"
  echo "SST | MLP no-RDrop | seed=$seed | lr=$lr | dropout=$dropout | exp=$exp_name"
  echo "============================================================"

  PYTHONPATH=. python -u ./scripts/run_classifier.py \
    "${COMMON_ARGS[@]}" \
    --train-flag 1 \
    --lr "$lr" \
    --epochs 5 \
    --hidden-dropout-prob "$dropout" \
    --sst-filepath "checkpoints/${prefix}sst.pt" \
    --cfimdb-filepath "checkpoints/${prefix}cfimdb.pt" \
    --predictions-prefix "$prefix" \
    --max-grad-norm 1.0 \
    --weight-decay 0.01 \
    --warmup-ratio 0.06 \
    --pooling-config last \
    --seed "$seed"
}

run_cfimdb_rdrop_last_mean_simple() {
  local seed="$1"
  local lr="$2"
  local alpha="$3"
  local epochs="$4"
  local dropout="$5"
  local exp_name="$6"

  local lr_tag
  local alpha_tag
  local drop_tag
  lr_tag="$(sanitize_float "$lr")"
  alpha_tag="$(sanitize_float "$alpha")"
  drop_tag="$(sanitize_float "$dropout")"

  local prefix="sweep-CFIMDB-${exp_name}-seed${seed}-lr${lr_tag}-ep${epochs}-alpha${alpha_tag}-drop${drop_tag}-"

  echo "============================================================"
  echo "CFIMDB | RDrop last_mean simple | seed=$seed | lr=$lr | alpha=$alpha | epochs=$epochs | dropout=$dropout | exp=$exp_name"
  echo "============================================================"

  PYTHONPATH=. python -u ./scripts/run_classifier.py \
    "${COMMON_ARGS[@]}" \
    --train-flag 2 \
    --lr "$lr" \
    --epochs "$epochs" \
    --hidden-dropout-prob "$dropout" \
    --sst-filepath "checkpoints/${prefix}sst.pt" \
    --cfimdb-filepath "checkpoints/${prefix}cfimdb.pt" \
    --predictions-prefix "$prefix" \
    --max-grad-norm 1.0 \
    --weight-decay 0.01 \
    --warmup-ratio 0.06 \
    --pooling-config last_mean \
    --use-simple-classifier \
    --use-rdrop \
    --rdrop-alpha "$alpha" \
    --seed "$seed"
}


echo "============================================================"
echo "START TARGETED HYPERPARAMETER SWEEP"
echo "LOG_FILE=$LOG_FILE"
echo "RUN_SST=$RUN_SST"
echo "RUN_CFIMDB=$RUN_CFIMDB"
echo "============================================================"


if [[ "$RUN_SST" == "1" ]]; then
  echo "============================================================"
  echo "START SST SWEEP"
  echo "============================================================"

  # 1) SST 핵심 후보: no-RDrop + MLP + last pooling
  # 기존 결과상 SST에서는 R-Drop보다 이쪽이 더 유망했으므로 lr=2e-5 근처만 탐색
  for seed in "${SEEDS[@]}"; do
    for lr in 1.5e-5 2e-5 3e-5; do
      run_sst_mlp_no_rdrop "$seed" "$lr" 0.2 "mlp-lr-sweep"
    done
  done

  # 2) dropout sweep
  # lr=2e-5 고정, dropout만 0.1/0.3 추가 확인
  # dropout=0.2는 위 lr sweep에 이미 포함됨
  for seed in "${SEEDS[@]}"; do
    for dropout in 0.1 0.3; do
      run_sst_mlp_no_rdrop "$seed" 2e-5 "$dropout" "mlp-dropout-sweep"
    done
  done

  echo "============================================================"
  echo "DONE SST SWEEP"
  echo "============================================================"
fi


if [[ "$RUN_CFIMDB" == "1" ]]; then
  echo "============================================================"
  echo "START CFIMDB SWEEP"
  echo "============================================================"

  # 1) CFIMDB 핵심 후보: last_mean + simple + R-Drop
  # 기존 결과상 alpha=1.0 근처가 좋았으므로 0.75/1.0/1.5만 탐색
  for seed in "${SEEDS[@]}"; do
    for alpha in 0.75 1.0 1.5; do
      run_cfimdb_rdrop_last_mean_simple "$seed" 2e-5 "$alpha" 5 0.2 "alpha-sweep"
    done
  done

  # 2) epoch 확인
  # CFIMDB는 best_epoch가 낮게 나오는 경향이 있어서 epoch=3도 확인
  for seed in "${SEEDS[@]}"; do
    run_cfimdb_rdrop_last_mean_simple "$seed" 2e-5 1.0 3 0.2 "epoch3-check"
  done

  # 3) lr 확인
  # lr=2e-5가 유망하지만, 1e-5도 같은 최고 성능 후보가 있었으므로 alpha=1.0에서 확인
  for seed in "${SEEDS[@]}"; do
    run_cfimdb_rdrop_last_mean_simple "$seed" 1e-5 1.0 5 0.2 "lr1e-5-check"
  done

  echo "============================================================"
  echo "DONE CFIMDB SWEEP"
  echo "============================================================"
fi


echo "============================================================"
echo "ALL TARGETED SWEEP DONE"
echo "============================================================"