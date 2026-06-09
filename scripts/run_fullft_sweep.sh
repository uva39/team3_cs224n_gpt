#!/usr/bin/env bash
# Full Fine-tuning 상한선 측정 sweep.
#
# 목적
#   - gpt2(124M) / gpt2-medium(355M) / gpt2-large(774M) 의 풀 FT dev CHRF 비교.
#   - 디코딩은 고정(temp=1.0, top_p=0.9, rep_penalty=1.2, no_repeat_ngram=3, max_lines=14)
#     하여 학습 변인(model_size × lr × seed)만 비교 가능하게 한다.
#   - 각 run의 ckpt/log/생성텍스트는 tag(seed 포함) 기반으로 자동 분리 저장 →
#     덮어쓰기/혼동 없이 보고서에 그대로 가져다 쓸 수 있다.
#
# 실행
#   bash scripts/run_fullft_sweep.sh                # phase=A (5 runs, seed=11711)
#   PHASE=B bash scripts/run_fullft_sweep.sh        # 시드 분산용 2 runs (best lr 입력 필요)
#   PHASE=C bash scripts/run_fullft_sweep.sh        # gpt2-large (grad_ckpt) 1 run
#
# 환경 변수
#   PHASE              실행 단계 (A | B | C, 기본 A)
#   BEST_LR_SMALL      Phase B 시드 분산용 small 모델 lr (기본 3e-5)
#   BEST_LR_MEDIUM     Phase B 시드 분산용 medium 모델 lr (기본 3e-5)
#   SEEDS_VARIANCE     Phase B에서 추가로 돌릴 시드 (콤마구분, 기본 "42,2024")
#   DRY_RUN=1          명령만 출력하고 실행은 하지 않음
#
# resume
#   각 run은 predictions/train_<tag>.json 존재 여부로 skip 판정. 중단 후 재실행해도 안전.

set -euo pipefail

# 어디서 호출되든 repo root(스크립트 파일의 상위 디렉터리)에서 실행되도록 보정.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PHASE=${PHASE:-A}
DRY_RUN=${DRY_RUN:-0}

# 공통 디코딩 설정 (모든 run에서 동일하게 적용 → 모델 비교가 공정해진다)
COMMON_DECODE=(
  --temperature 1.0
  --top_p 0.9
  --repetition_penalty 1.2
  --no_repeat_ngram_size 3
  --max_lines 14
)

# 공통 학습 설정
COMMON_TRAIN=(
  --use_gpu
  --patience 4
)

run_one() {
  # run_one <tag> <python args...>
  local TAG=$1; shift
  local LOG="predictions/train_${TAG}.json"
  local STDOUT="predictions/run_${TAG}.log"
  if [[ -f "$LOG" ]]; then
    echo "[skip] $TAG (log exists: $LOG)"
    return 0
  fi
  echo "============================================================"
  echo "[run] $TAG  $(date '+%F %T')"
  echo "      python sonnet_generation.py $*"
  echo "      stdout → $STDOUT"
  echo "============================================================"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  # 한 run이 OOM/에러로 죽어도 다음 run으로 진행: 비-0 종료를 흡수.
  if python sonnet_generation.py "$@" > "$STDOUT" 2>&1; then
    echo "[done] $TAG  $(date '+%F %T')"
  else
    local rc=$?
    echo "[FAIL] $TAG  rc=$rc  $(date '+%F %T') — tail $STDOUT:"
    tail -n 25 "$STDOUT" || true
    # 실패 표식 파일 (다음 실행 시 skip 되지 않도록 LOG는 만들지 않음)
    echo "{\"failed\": true, \"rc\": $rc, \"at\": \"$(date -Iseconds)\"}" \
      > "predictions/failed_${TAG}.json"
  fi
}

case "$PHASE" in
  A)
    # ----------------------------------------------------------------
    # Phase A: seed=11711 고정, model_size × lr 격자
    # 5 runs. gpt2-small 3개 lr, gpt2-medium 3개 lr.
    # 실행 시간 추정 (1g.10gb MIG):
    #   gpt2-small:  ~3분/run  × 3 = 9분
    #   gpt2-medium: ~25분/run × 3 = 75분
    #   합계 ~1.5시간
    # ----------------------------------------------------------------
    SEED=11711

    # gpt2-small (124M), batch=8 ok in 10GB
    for LR in 1e-5 3e-5 5e-5; do
      TAG="gpt2-fullft-s${SEED}-15ep-${LR}"
      run_one "$TAG" \
        "${COMMON_TRAIN[@]}" "${COMMON_DECODE[@]}" \
        --model_size gpt2 --seed "$SEED" \
        --epochs 15 --batch_size 8 --lr "$LR"
    done

    # gpt2-medium (355M), batch=4 (메모리 안전 마진)
    for LR in 1e-5 3e-5 5e-5; do
      TAG="gpt2-medium-fullft-s${SEED}-15ep-${LR}"
      run_one "$TAG" \
        "${COMMON_TRAIN[@]}" "${COMMON_DECODE[@]}" \
        --model_size gpt2-medium --seed "$SEED" \
        --epochs 15 --batch_size 4 --lr "$LR"
    done
    ;;

  B)
    # ----------------------------------------------------------------
    # Phase B: 시드 분산 (Phase A 결과에서 best lr를 알아낸 뒤 실행)
    # 각 모델 크기의 best lr로 추가 시드 2개씩 → 분산 평가
    # ----------------------------------------------------------------
    BEST_LR_SMALL=${BEST_LR_SMALL:-3e-5}
    BEST_LR_MEDIUM=${BEST_LR_MEDIUM:-3e-5}
    SEEDS_VARIANCE=${SEEDS_VARIANCE:-42,2024}

    IFS=',' read -ra SEED_ARR <<< "$SEEDS_VARIANCE"
    for SEED in "${SEED_ARR[@]}"; do
      TAG="gpt2-fullft-s${SEED}-15ep-${BEST_LR_SMALL}"
      run_one "$TAG" \
        "${COMMON_TRAIN[@]}" "${COMMON_DECODE[@]}" \
        --model_size gpt2 --seed "$SEED" \
        --epochs 15 --batch_size 8 --lr "$BEST_LR_SMALL"

      TAG="gpt2-medium-fullft-s${SEED}-15ep-${BEST_LR_MEDIUM}"
      run_one "$TAG" \
        "${COMMON_TRAIN[@]}" "${COMMON_DECODE[@]}" \
        --model_size gpt2-medium --seed "$SEED" \
        --epochs 15 --batch_size 4 --lr "$BEST_LR_MEDIUM"
    done
    ;;

  C)
    # ----------------------------------------------------------------
    # Phase C: gpt2-large 풀 FT (stretch).
    #
    # gpt2-large(774M) 풀 FT는 1g.10gb(=10GB VRAM)에 들어가지 않을 가능성이 높다:
    #   - fp32 weights: ~3GB
    #   - AdamW state (m,v): ~6GB
    #   - activations (batch=1, grad_ckpt): 1~2GB
    #   합 ~10GB → OOM 위험.
    #
    # 만약 OOM이면 메시지를 기록만 하고 다음 phase로 진행한다.
    # 그 경우 gpt2-large는 LoRA 경로(단계 6에 이미 구현)로만 다룬다.
    # ----------------------------------------------------------------
    SEED=11711
    LR=1e-5
    TAG="gpt2-large-fullft-s${SEED}-10ep-${LR}"
    run_one "$TAG" \
      "${COMMON_TRAIN[@]}" "${COMMON_DECODE[@]}" \
      --model_size gpt2-large --seed "$SEED" \
      --epochs 10 --batch_size 1 --lr "$LR" \
      --grad_checkpoint
    ;;

  *)
    echo "unknown PHASE: $PHASE (use A, B, or C)" >&2
    exit 2
    ;;
esac

echo
echo "[sweep] PHASE=$PHASE finished."
echo "[sweep] 요약은: python scripts/summarize_experiments.py"
