#!/usr/bin/env bash
# 학습 하이퍼파라미터 그리드.
# 한 번에 돌리지 말고, 시간이 허락하는 범위 안에서 줄여서 사용하세요.
# 각 run의 train_log.json은 LOG_DIR 아래에 config 태그로 저장됩니다.

set -euo pipefail

LOG_DIR=${LOG_DIR:-predictions/sweeps}
mkdir -p "$LOG_DIR"

# ============================================
# A. LoRA rank × lr × dropout × weight_decay
# ============================================
for MODEL in gpt2 gpt2-large; do
  for RANK in 4 8 16 32; do
    for LR in 1e-4 2e-4 5e-4; do
      for HD in 0.1 0.2; do
        for WD in 0.0 0.01 0.1; do
          TAG="${MODEL}-lora-r${RANK}-lr${LR}-hd${HD}-wd${WD}"
          LOGFILE="${LOG_DIR}/${TAG}.json"
          if [[ -f "$LOGFILE" ]]; then
            echo "[skip] $TAG already done"
            continue
          fi
          echo "[run] $TAG"
          python sonnet_generation.py --use_gpu \
            --model_size "$MODEL" \
            --use_lora --lora_rank "$RANK" --lora_alpha "$((RANK*2))" --lora_dropout 0.05 \
            --grad_checkpoint \
            --batch_size 2 --epochs 8 --lr "$LR" \
            --hidden_dropout "$HD" --attn_dropout "$HD" --weight_decay "$WD" \
            --patience 3 --temperature 1.1 --top_p 0.9 \
            --no_repeat_ngram_size 3 --max_lines 14 \
            --log_path "$LOGFILE"
        done
      done
    done
  done
done

# ============================================
# B. 풀 FT × lr × dropout × weight_decay (small만 권장)
# ============================================
for LR in 5e-6 1e-5 2e-5; do
  for HD in 0.1 0.2; do
    for WD in 0.0 0.01 0.1; do
      TAG="gpt2-full-lr${LR}-hd${HD}-wd${WD}"
      LOGFILE="${LOG_DIR}/${TAG}.json"
      if [[ -f "$LOGFILE" ]]; then echo "[skip] $TAG"; continue; fi
      echo "[run] $TAG"
      python sonnet_generation.py --use_gpu \
        --model_size gpt2 --batch_size 8 --epochs 10 --lr "$LR" \
        --hidden_dropout "$HD" --attn_dropout "$HD" --weight_decay "$WD" \
        --patience 3 --no_repeat_ngram_size 3 --max_lines 14 \
        --log_path "$LOGFILE"
    done
  done
done

echo "[done] all runs in $LOG_DIR"
