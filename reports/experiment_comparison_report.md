# Quora Paraphrase Experiment Comparison Report

## 1. 분석 기준

이 리포트는 `logs/*.csv`에 기록된 Quora paraphrase detection 실험만 비교한다. 기본 비교 원칙은 대조군과 비교군 사이에서 한 가지 설정만 다르게 두는 것이다. 주요 지표는 `dev_acc`, 보조 지표는 `dev_f1`로 본다.

일부 3epoch 실험의 대조군은 별도 3epoch baseline 파일이 없으므로, 동일한 5epoch baseline CSV에서 epoch 2 행을 잘라 사용했다. 이때 CSV의 `best_dev_acc` 컬럼은 이전 best를 기록하는 구조라서, 같은 epoch의 실제 `dev_acc/dev_f1` 값을 사용했다.

## 2. 전체 결론

- Full fine-tuning 기준 최고 단일 성능은 `lr=2e-5, bs=4`이며, epoch 2에서 `dev_acc=0.8956`, `dev_f1=0.8890`을 기록했다.
- 가장 안정적인 full fine-tuning 설정은 `lr=1e-5, bs=8`이다. 최종 epoch까지 상승하며 `dev_acc=0.8953`, `dev_f1=0.8887`을 기록했고 train/dev gap도 상대적으로 작다.
- Full fine-tuning에서 `lr=1e-4`는 확실히 과하다. 같은 `bs=4`에서 `lr=1e-5` 대비 `dev_acc -0.0248`, `dev_f1 -0.0273`이다.
- Full fine-tuning의 dropout, weight decay는 현재 범위에서는 성능 개선을 만들지 못했다. baseline과 비교하면 둘 다 아주 작게 낮거나 거의 동일하다.
- LoRA는 학습 가능한 파라미터가 매우 적어 gap은 작지만, 현재 `query,value` target만으로는 full fine-tuning 성능에 못 미친다. 다만 5epoch에서 10epoch로 이어 학습하면 `dev_acc +0.0141`이 올라, 아직 학습 여지가 남아 있다.
- LoRA에서 rank 16은 rank 8보다 명확히 낫다. 동일한 `wd=0.01`, 3epoch 조건에서 `dev_acc +0.0042`, `dev_f1 +0.0027`이다.

## 3. 한 변수 비교표

| 비교 항목 | 대조군 | 비교군 | dev_acc 변화 | dev_f1 변화 | 해석 |
|---|---|---|---:|---:|---|
| LR: `1e-5 -> 2e-5`, `bs=4` | `lr1e-5-ep5-bs4` | `lr2e-5-ep5-bs4` | +0.0013 | +0.0009 | 작은 개선. `bs=4`에서는 `2e-5`가 약간 유리하다. |
| LR: `1e-5 -> 2e-5`, `bs=8` | `lr1e-5-ep5-bs8` | `lr2e-5-ep5-bs8` | -0.0008 | -0.0008 | `bs=8`에서는 `1e-5`가 약간 더 안정적이다. |
| LR: `1e-5 -> 1e-4`, `bs=4` | `lr1e-5-ep5-bs4` | `lr1e-4-ep5-bs4` | -0.0248 | -0.0273 | 큰 하락. full fine-tuning에서 `1e-4`는 너무 크다. |
| Batch: `4 -> 8`, `lr=1e-5` | `lr1e-5-ep5-bs4` | `lr1e-5-ep5-bs8` | +0.0010 | +0.0005 | `lr=1e-5`에서는 `bs=8`이 근소하게 우세하다. |
| Batch: `4 -> 8`, `lr=2e-5` | `lr2e-5-ep5-bs4` | `lr2e-5-ep5-bs8` | -0.0011 | -0.0012 | `lr=2e-5`에서는 `bs=4`가 더 좋다. lr과 batch size 상호작용이 있다. |
| Dropout: `0 -> 0.1`, full FT 3epoch | `lr1e-5-bs8` epoch 2 baseline | `wd0-drop0.1` 3epoch | -0.0013 | -0.0021 | dropout 단독은 개선 없음. 약한 성능 하락. |
| Weight decay: `0 -> 0.01`, full FT 3epoch | `lr1e-5-bs8` epoch 2 baseline | `wd0.01-drop0` 3epoch | -0.0004 | -0.0006 | weight decay 단독도 거의 동일하거나 소폭 하락. |
| WD+Dropout combo, full FT 5epoch | `lr1e-5-ep5-bs8` | `wd0.01-drop0.1` | -0.0008 | -0.0007 | 조합도 baseline을 넘지는 못했다. gap만 아주 약간 감소. |
| LoRA duration: 5epoch -> 10epoch | `r8/a16` 5epoch | `r8/a16` resume to 10epoch | +0.0141 | +0.0134 | LoRA는 5epoch에서 아직 덜 학습된 상태였다. 이어 학습 효과가 큼. |
| LoRA weight decay: `0 -> 0.01`, r8 3epoch | `r8/a16` no-wd epoch 2 | `r8/a16 wd0.01` 3epoch | -0.0001 | +0.0001 | 사실상 차이 없음. LoRA r8에서는 wd 효과가 거의 없다. |
| LoRA rank: `8 -> 16`, wd=0.01 3epoch | `r8/a16 wd0.01` | `r16/a16 wd0.01` | +0.0042 | +0.0027 | rank를 늘리면 명확히 좋아진다. 현재 LoRA는 capacity가 병목일 가능성이 있다. |

## 4. Full Fine-Tuning 분석

### Learning rate

`lr=2e-5`는 `bs=4`에서 가장 높은 peak를 만든다. 하지만 best epoch가 2로 이르고, 이후 train 성능은 계속 오르지만 dev 성능은 정체하거나 떨어진다. 즉 빠르게 잘 맞추지만 과적합도 빠르게 온다.

`lr=1e-5, bs=8`은 최고점은 아주 근소하게 낮지만 epoch 4까지 안정적으로 상승한다. 제출 또는 재현 목적이라면 이 설정이 가장 다루기 쉽다.

`lr=1e-4`는 full fine-tuning에서는 부적합하다. train acc는 높아지지만 dev 성능이 크게 낮아져서, pretrained representation을 과하게 흔든 것으로 보인다.

### Batch size

Batch size 효과는 learning rate에 종속적이다. `lr=1e-5`에서는 `bs=8`이 좋고, `lr=2e-5`에서는 `bs=4`가 좋다. 즉 batch size 자체보다 effective update noise와 lr 조합이 중요해 보인다.

### Regularization

Full fine-tuning에서 `dropout=0.1`, `weight_decay=0.01`은 baseline 대비 성능 개선을 만들지 못했다. 다만 `wd+dropout` 조합은 final gap이 `0.0462 -> 0.0459`로 아주 약하게 줄었다. 성능 향상보다는 과적합을 미세하게 누르는 정도다.

## 5. LoRA 분석

현재 LoRA는 GPT-2 본체를 freeze하고 `query,value` projection에만 adapter를 붙인 설정이다. r8 기준 trainable parameter는 약 296K로, 전체 125M 대비 약 0.24%다.

LoRA r8/a16 5epoch는 `dev_acc=0.8642`에 그쳤지만, 같은 checkpoint에서 10epoch까지 이어 학습하자 `dev_acc=0.8784`까지 상승했다. train/dev gap도 매우 작아서 overfitting보다는 underfitting 또는 capacity 부족 쪽에 가깝다.

Weight decay는 LoRA r8에서는 거의 효과가 없다. 반면 rank를 8에서 16으로 늘리면 3epoch 기준으로 `dev_acc +0.0042`, `dev_f1 +0.0027`이 개선된다. 따라서 LoRA 쪽 다음 실험은 regularization보다 capacity 확장이 더 우선이다.

## 6. 추천 다음 실험

1. Full fine-tuning 제출 후보는 `lr=1e-5, bs=8, ep5` 또는 `lr=2e-5, bs=4`의 best checkpoint다. 최고점만 보면 `lr=2e-5, bs=4`가 우세하고, 안정성은 `lr=1e-5, bs=8`이 우세하다.
2. LoRA는 `rank=16` 이상을 더 봐야 한다. 현재 결과는 rank 증가가 weight decay보다 훨씬 의미 있다.
3. LoRA target을 `query,value`에서 `query,value,attention_dense` 또는 `query,value,attention_dense,out_dense`로 넓히는 실험이 유망하다.
4. LoRA는 10epoch에서도 full fine-tuning보다 낮으므로, LoRA만으로 따라잡으려면 rank/target 확장 또는 더 긴 epoch가 필요하다.
5. Full fine-tuning에서 `lr=1e-4`는 제외하는 것이 좋다.
