# 소네트 생성(Sonnet Generation) 작업 보고서

> 본 보고서는 CS 224N Default Final Project — Build GPT-2 의 **Part 2: Sonnet Generation** 작업 과정 중
> AI(Claude)와 협업하여 수행한 작업 내역을 시간 순서대로 정리한 문서이다.
> 학습 환경: AI Pub A100 GPU. 최종 목표: 기본 fine-tuning 베이스라인 확보 → PEFT(LoRA 등) 기법 적용.

---

## 0. 출발 시점의 저장소 상태 (작업 전)

- Part 1 backbone 구현은 채워져 있음 (`modules/attention.py`, `modules/gpt2_layer.py`, `models/gpt2.py`).
- `sonnet_generation.py`
  - `SonnetGPT.forward()`: 이미 구현됨. `last_hidden_state`를 `embed.word_embedding.weight`와 `F.linear`로 곱해 LM head로 사용 (weight tying).
  - `generate()`: top-p sampling + temperature 구현. beam/top-k/repetition penalty 미구현.
  - `train()`: 단순 cross-entropy 학습 루프만 있고 **dev 평가/early stopping/best-ckpt 저장 로직 없음**.
- 동료 작업물: `scripts/prepare_training_json.py` — 모든 데이터(CSV/txt)를 정형화된 JSON으로 변환하는 스크립트. **이 시점에는 아직 실행되지 않은 상태였음** (`data/json/` 디렉터리 부재).
- 데이터셋
  - `data/sonnets.txt`: 학습용 셰익스피어 소네트 본문
  - `data/sonnets_held_out.txt`, `data/sonnets_held_out_dev.txt`: 첫 3줄만 주어진 prompt set
  - `data/TRUE_sonnets_held_out_dev.txt`: dev gold (CHRF 평가용)
- 평가: `evaluation.py::test_sonnet()`이 sacreBLEU의 CHRF metric으로 채점.

---

## 1. 작업 계획 (합의된 단계)

1. **학습 데이터 JSON 변환 스크립트 실행** — 동료가 작성한 `scripts/prepare_training_json.py` 산출물 검증.
2. **베이스라인 학습 스크립트 정비** — dev CHRF 평가 루프, best-checkpoint 저장, early stopping, 로깅을 `sonnet_generation.py`에 추가.
3. **A100에서 베이스라인 학습** — `gpt2`(124M) → 필요 시 `gpt2-medium`(355M) 풀 fine-tuning. 베이스라인 CHRF 확보.
4. **PEFT 실험** — LoRA / Prefix-Tuning / (IA)³ 등을 비교 매트릭스로 적용.
5. **생성 디코딩 개선** — beam search, no_repeat_ngram, 14줄 구조 제약 등 학습-무관 개선.

원칙: 매 단계 변경 사항·실행 결과·관찰점을 본 보고서에 누적 기록한다.

---

## 2. 단계별 작업 로그

### 단계 1. 학습 데이터 JSON 변환 (담당: 동료 스크립트 / 실행: AI)

**실행 명령**

```bash
python3 scripts/prepare_training_json.py
```

**산출물**: `data/json/` 디렉터리에 13개 dataset JSON + `manifest.json` 1개 생성됨.

**소네트 관련 산출물 요약** (본 작업의 핵심 데이터)

| 파일 | split | num_examples | 비고 |
|------|-------|--------------|------|
| `sonnets.json` | train | **131** | 학습용 전체 소네트 (id, prompt_lines 3줄, completion_lines 11줄, lines 14줄, text 모두 포함) |
| `sonnets_held_out_dev.json` | dev_prompt | 12 | 첫 3줄만 제공 (생성 시 입력 prompt) |
| `TRUE_sonnets_held_out_dev.json` | dev_gold | 12 | dev set 정답 14줄. **CHRF dev 평가에 사용** |
| `sonnets_held_out.json` | test_prompt | 12 | 최종 제출용 prompt (정답 없음) |

**일관성 검증**

- `prepare_training_json.py::parse_sonnets()`(정규식 `\n\s*(\d+)\s*\n`)와 기존 `datasets.SonnetsDataset._load_sonnets()`(정규식 `\n\s*\d+\s*\n`)는 동일한 소네트 분리 결과를 낸다. 두 경로 모두 131 / 12 / 12 / 12 개를 추출하여 일치.

**JSON 내부 구조 (sonnet 한 편 기준, 학습용)**

```jsonc
{
  "task": "generation", "dataset": "sonnets", "split": "train",
  "id": 1,
  "prompt_lines": ["From fairest creatures we desire increase,", "...", "..."],
  "prompt_text": "From fairest creatures ... (3줄을 \n으로 연결)",
  "num_prompt_lines": 3,
  "lines": [...14줄 전체...],
  "text": "...전체 본문(개행 포함)...",
  "completion_lines": [...11줄...],
  "completion_text": "...",
  "num_lines": 14
}
```

**중요 메모**

- 현재 `sonnet_generation.py`의 학습 루프는 **JSON이 아닌 원본 `.txt`** 를 `datasets.SonnetsDataset`로 직접 로드한다. JSON 산출물은 **(a) 데이터 분석 / 디버깅, (b) 추후 dev CHRF 평가용 prompt-gold 페어 로딩, (c) 향후 augmentation/필터링 실험** 등에 사용할 보조 자산이다.
- Shakespeare 154편 중 131편만 추출되는 이유는 원본 `sonnets.txt`가 일부 소네트만 제공하는 것이 아니라, 정규식이 "공백+숫자+공백" 단독 라인을 구분자로 쓰기 때문이며 dev/held-out과 합쳐 154편이 분할되어 있다 (131 train + 12 dev + 12 held-out test = 155, dev/held-out에 일부 중복 가능). 추후 필요 시 정밀 확인.


### 단계 2. 베이스라인 학습 스크립트 정비

**파일**: `sonnet_generation.py`

**도입 동기**

원본 `train()` 루프는 (a) dev metric을 측정하지 않고, (b) 매 에폭의 checkpoint를 별도 파일로 저장하며, (c) early stopping이 없다.
소네트 학습 셋은 131편으로 매우 작아 **빠르게 과적합**되기 때문에 dev CHRF로 최고 시점을 추적하지 않으면
최종 점수가 마지막 에폭(거의 항상 과적합)에 묶여버린다. 또 154-step 가까운 cross-entropy loss는 단조 감소하므로
loss만 보고 best를 고르는 것은 의미가 없다 (CHRF 추적 필수).

**추가/변경 사항 요약**

1. **`evaluate_dev_chrf(model, args, device, prompt_path, gold_path, max_length, verbose)`** 함수 신설
   - `data/sonnets_held_out_dev.txt` 의 12개 prompt(앞 3줄)를 그대로 모델 `generate()`에 흘려서 14줄 짜리 sonnet 생성
   - `data/TRUE_sonnets_held_out_dev.txt` 의 정답 14줄과 sacreBLEU **CHRF** (corpus-level) 점수 계산
   - in-memory 처리 (디스크 임시 파일 사용 안 함)
2. **`train()` 루프 개편**
   - 에폭마다 train loss + 시간 + lr 기록 (`history` 리스트)
   - dev 평가가 켜져 있으면 dev CHRF 계산 후
     - 신기록이면 `args.filepath` 단일 경로에 best checkpoint 저장 (`save_model` 1회)
     - patience 카운터 초기화
     - 신기록 아니면 patience++; `patience` 도달 시 early stop break
   - dev 평가가 꺼져 있으면 기존 동작과 호환되게 매 에폭 `args.filepath`(같은 파일을 덮어쓰기)로 저장
   - 에폭마다 `args.log_path` (기본 `predictions/train_log.json`)에 학습 이력 적재
3. **`generate_submission_sonnets()` 수정**
   - 기존: `f'{args.epochs-1}_{args.filepath}'` 로드 → 새 학습 루프에서는 해당 파일이 존재하지 않음
   - 변경: 우선 `args.filepath` (best ckpt) 시도, 없으면 legacy 경로 fallback 시도, 둘 다 없으면 `FileNotFoundError`
4. **새 CLI 인자**

   | 인자 | 기본값 | 설명 |
   |------|--------|------|
   | `--dev_prompt_path` | `data/sonnets_held_out_dev.txt` | dev prompts |
   | `--dev_gold_path` | `data/TRUE_sonnets_held_out_dev.txt` | dev gold |
   | `--no_dev_eval` | (flag) | dev 평가 비활성화 |
   | `--patience` | 3 | early stop patience (0이면 비활성) |
   | `--dev_max_new_tokens` | 128 | dev 평가 시 생성 토큰 상한 |
   | `--dev_verbose` | (flag) | 각 dev epoch 생성 결과 콘솔 출력 |
   | `--log_path` | `predictions/train_log.json` | 학습 로그 JSON 출력 경로 |

**행위 변화 — 명시적으로 기록**

- 원본은 매 에폭 끝에 `args.held_out_sonnet_path`(test prompt set)로 sonnet을 생성해 콘솔 출력. 새 코드는 `--dev_verbose` flag를 켰을 때만, dev prompt set으로 출력. test prompt는 마지막 `generate_submission_sonnets()`에서만 사용.
- 원본은 `{epoch}_{filepath}` 형식으로 epochs 개의 ckpt를 저장 (디스크 사용량 = `epochs × 모델 크기`). 새 코드는 best 1개만 보유.

**사용 예시 (A100)**

```bash
# 베이스라인 (gpt2 small, 10 epoch, dev CHRF로 best 추적, patience=3)
python sonnet_generation.py --use_gpu

# gpt2-medium 풀 fine-tuning
python sonnet_generation.py --use_gpu --model_size gpt2-medium \
       --batch_size 4 --lr 5e-6 --epochs 15 --patience 4

# dev 평가 끄고 학습만 (디버깅용)
python sonnet_generation.py --use_gpu --no_dev_eval --epochs 3
```

**구문 검증**

`python3 -c "import ast; ast.parse(open('sonnet_generation.py').read())"` → OK.
실제 학습 실행은 A100 환경에서 진행 예정.

**다음 단계**

1. A100에서 `gpt2` small 베이스라인 실행, dev CHRF 곡선 확인
2. `gpt2-medium`까지 확장해 풀 FT 상한선 측정
3. PEFT (LoRA → Prefix-Tuning → 비교) 적용 단계로 이동
4. 생성 디코딩 개선 (beam, no_repeat_ngram, 14줄 강제 종료)


### 단계 3. JSON 파이프라인 실제 연결 (계획 수정)

단계 1에서는 JSON을 "보조 자산"으로 두고 학습은 원본 `.txt`를 그대로 썼다. 이후 **학습 입력을 JSON으로 일원화**하기로 방침을 바꿨다 (`data/README.md`의 "JSON을 기본 입력으로" 권장과 일치).

**변경 사항**

1. **`datasets.py::SonnetsDataset`** — `.txt`/`.json`을 모두 읽도록 확장.
   - `.json`이면 `payload["examples"]`를 순회하며 **실제 소네트 번호(`id`)** 와, 전체 텍스트가 있으면 `text`를(학습/gold), prompt 전용 파일이면 `prompt_text`를(held-out/dev) 사용.
   - `.txt`는 기존 동작(정규식 분리 + 위치 인덱스 id) 그대로 유지 → 하위 호환.
   - `__getitem__`이 `(id, text)`를 반환하도록 `ids`/`sonnets`를 분리 저장.
2. **`sonnet_generation.py`** — 기본 입력 경로 4개를 JSON으로 전환.
   - `--sonnet_path` → `data/json/sonnets.json`
   - `--held_out_sonnet_path` → `data/json/sonnets_held_out.json`
   - `--dev_prompt_path` → `data/json/sonnets_held_out_dev.json`
   - `--dev_gold_path` → `data/json/TRUE_sonnets_held_out_dev.json`
   - 출력 `--sonnet_out`(제출 txt)은 유지.

**검증** (서버 conda 환경)

| 파일 | 개수 | first id |
|------|------|----------|
| `sonnets.json` | 131 | 1 |
| `sonnets_held_out.json` | 12 | 145 |
| `sonnets_held_out_dev.json` | 12 | 132 |
| `TRUE_sonnets_held_out_dev.json` | 12 | 132 |

- 개수 131/12/12/12로 기존 txt 경로와 일치. dev prompt/gold가 모두 id 132부터 시작하고 첫 줄도 동일 → **prompt-gold 정렬 정상**.
- **효과**: 제출 파일의 소네트 번호가 위치 인덱스(0~11)가 아니라 **실제 번호(145~156)** 로 기록된다.


### 단계 4. 학습 환경 구축 및 베이스라인 실행

**실제 환경** (헤더의 "A100 GPU"는 명목상 표기이며, 실제 할당은 아래와 같음)

- GPU: **A100 80GB가 MIG로 분할되어, 할당분은 `1g.10gb` 슬라이스 1개**(~10GB VRAM, A100의 약 1/7 연산). CPU 256코어, RAM 1TB.
- 컨테이너: Ubuntu 18.04 (**glibc 2.27**), 도구 최소 설치(pip/conda/git 없음). 최신 Miniconda는 glibc 2.28을 요구해 설치 불가 → **구버전 `Miniconda3-py310_23.5.2`** 설치.
- conda 환경 `cs224n_dfp`: Python 3.8, **torch 2.4.1+cu121**, transformers 4.46.3. `torch.cuda.is_available()=True`, MIG 1g.10gb 인식 및 연산 테스트 통과.

**런타임 버그 발견 및 수정 (`forward`)**

단계 0에서 `forward()`는 "이미 구현됨"으로 기록되어 있었으나, **실제 GPU 실행 시 첫 배치에서 즉시 크래시**했다.

```
AttributeError: 'function' object has no attribute 'word_embedding'
  sonnet_generation.py:74  logits = F.linear(sequence_output, self.gpt.embed.word_embedding.weight)
```

- 원인: `self.gpt.embed`는 **메서드**(함수)라 `.word_embedding` 속성이 없다. `word_embedding`은 `models/gpt2.py`의 `GPT2Model`에 직접 정의돼 있다(weight-tying용 헬퍼 `hidden_state_to_token`도 동일하게 `self.word_embedding.weight` 사용).
- 수정: `self.gpt.embed.word_embedding.weight` → **`self.gpt.word_embedding.weight`** (LM head weight tying, 의도 동일).

**학습 실행**

```bash
python sonnet_generation.py --use_gpu      # nohup 백그라운드 실행, train.log로 로깅
```

설정: `gpt2`(124M), epochs=10, batch_size=8, lr=1e-5, temperature=1.2, top_p=0.9, patience=3, dev CHRF 평가 on. 에폭당 약 10초(train).

**결과** (`predictions/train_log.json`)

| epoch | train_loss | dev_chrf |
|------:|-----------:|---------:|
| 0 | 4.7104 | 30.94 |
| 1 | 4.3742 | 37.93 |
| 2 | 4.2529 | 37.01 |
| 3 | 4.1710 | 39.23 |
| 4 | 4.0957 | 39.37 |
| 5 | 4.0618 | 39.23 |
| 6 | 4.0153 | 38.38 |
| **7** | **3.9813** | **40.51 ← best** |
| 8 | 3.9484 | 39.38 |
| 9 | 3.9048 | 39.12 |

- **best dev CHRF = 40.51 @ epoch 7**, best 체크포인트 `10-1e-05-sonnet.pt`(1.5GB) 저장.
- `train_loss`는 단조 감소(4.71→3.90)하나 `dev_chrf`는 비단조이며 epoch 7 이후 하락 → **과적합 신호**. loss만으로 best를 고를 수 없음을 재확인(CHRF 추적의 정당성). patience=3 미충족으로 10 epoch 완주.

**생성 결과** (`predictions/generated_sonnets.txt`)

- held-out test 12편 생성, 소네트 번호 **145~156**로 기록(JSON 연결 확인).
- 관찰: 프롬프트 3줄은 보존되나 **14행 형식이 정확히 지켜지지 않음**(일부 12행 등), 반복/조기 종료 위험 잔존 → 디코딩 개선 필요(`data/README.md`의 품질 체크리스트와 일치).

**산출물 저장**

- 로컬 저장 완료: `predictions/generated_sonnets.txt`, `predictions/train_log.json`.
- 체크포인트 `10-1e-05-sonnet.pt`(1.5GB)는 서버 `/root/team3_cs224n_gpt/`에 존재(`.gitignore`의 `*.pt` 대상, 파드 재시작 시 소실 가능).


### 단계 5. 현황 요약 및 다음 작업

- ✅ **베이스라인 확보**: `gpt2` small 풀 fine-tuning, dev CHRF **40.51**.
- ✅ JSON 파이프라인 학습 입력 일원화, 제출 id 정합(145~156).
- ⏭ **남은 작업**
  1. **생성 디코딩 개선** — repetition penalty / `no_repeat_ngram`, **14행 강제 종료**, beam/top-k 비교 (학습 무관, 즉시 CHRF 상승 기대).
  2. **`gpt2-medium`(355M) 풀 FT** — 10GB 슬라이스에 적재 가능, 상한선 측정.
  3. **PEFT** — LoRA → Prefix-Tuning → (IA)³ 비교 매트릭스 (최종 목표).


### 단계 6. 7.4 확장 통합 구현 (PEFT/DPO/Ensemble/Regularization/HP-Sweep)

핸드아웃 7.4의 확장안 중 합의된 5종을 동시에 구현했다. 코드는 *기존 베이스라인을 깨지 않는다* —
모든 신규 동작은 CLI 플래그로 opt-in이며, 플래그 없이 실행하면 단계 4의 40.51 결과를 재현한다.

#### 6.1 추가/수정된 파일

| 파일 | 종류 | 내용 |
|------|------|------|
| `modules/lora.py` | NEW | `LoRALinear` 래퍼, `inject_lora()`, `mark_only_lora_as_trainable()`, `count/trainable_parameters()` |
| `models/gpt2.py` | MOD | `from_pretrained`에 `hidden_dropout_prob`/`attention_probs_dropout_prob`/`gradient_checkpointing` 인자 추가, `encode()`에 `torch.utils.checkpoint` 분기 |
| `sonnet_generation.py` | MOD | LoRA 주입/freeze, dropout·weight_decay·grad-ckpt CLI 배선, `generate()`에 `top_k`/`repetition_penalty`/`no_repeat_ngram_size`/`max_lines` 추가, 옵티마이저는 trainable 파라미터만, log_config 확장, ckpt 파일명 태깅 |
| `dpo.py` | NEW | SFT ckpt → policy+ref 로드, 휴리스틱(5종)+자가생성 음성 쌍 구성, sequence logprob 기반 DPO 손실, dev CHRF best 저장 |
| `ensemble.py` | NEW | 다중 ckpt 로드 + 후보 다중 생성(`mode=select`) 또는 prob-average 디코딩(`mode=prob_avg`), reference-free 소네트 품질 점수로 best 선택 |
| `scripts/sweep_decoding.py` | NEW | 체크포인트 1개로 디코딩 HP 그리드를 dev CHRF로 평가(재학습 불필요, 매우 저렴) |
| `scripts/sweep_train.sh` | NEW | LoRA rank × lr × dropout × weight_decay 학습 그리드 |

#### 6.2 LoRA (PEFT) 설계 요점

자체 GPT-2 구현이라 HuggingFace `peft`를 못 쓰고 직접 구현했다. 핵심은:

- `LoRALinear(base, rank, alpha, dropout)`이 `nn.Linear`를 감싸 `W' x = W x + (α/r) · B(Ax)`를 계산. `W`는 freeze, `A`/`B`만 학습.
  `B`는 0으로 초기화해 학습 시작 시 delta가 정확히 0 → 사전학습 가중치를 보존한 채 출발.
- `inject_lora(model, rank, alpha, dropout, targets=...)`가 모델 트리를 순회하며 이름이 타깃에 속하는 `nn.Linear`만 `LoRALinear`로 교체.
  기본 타깃: `query, key, value, attention_dense, interm_dense, out_dense` (어텐션 Q/K/V/proj + MLP의 두 dense).
- `mark_only_lora_as_trainable()`이 `lora_*` 파라미터만 `requires_grad=True`로 만든다.
- 체크포인트 호환: `SonnetGPT(saved['args'])`가 `args.use_lora`를 보고 동일하게 LoRA를 재주입한 뒤 `load_state_dict`. 베이스 가중치는 HF에서 다시 로드된 후 ckpt로 덮어써지므로 일치.
- ⚠ `gpt2-large` + LoRA + 10GB MIG: gradient checkpointing 필수(`--grad_checkpoint`). 활성화 메모리를 재계산으로 트레이드오프.

새 CLI 인자: `--use_lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --lora_targets q,k,v --lora_train_ln --grad_checkpoint`.

#### 6.3 dropout / 정규화 / 하이퍼파라미터 배선

- `--hidden_dropout`, `--attn_dropout` → `GPT2Config.hidden_dropout_prob` / `attention_probs_dropout_prob` 오버라이드.
- `--weight_decay` → `AdamW(weight_decay=...)`.
- 옵티마이저는 이제 `[p for p in model.parameters() if p.requires_grad]`만 받음(LoRA 시 메모리·연산 절감).
- ckpt 파일명: `{model_size}{-lora 여부}-{epochs}ep-{lr}-sonnet.pt` → 실험 간 덮어씀 방지.

#### 6.4 `generate()` 디코딩 개선 (학습 무관, 즉시 적용)

추가된 인자(모두 기본=비활성, 베이스라인 보존):
- `--top_k N`: 상위 N개 토큰만 후보.
- `--repetition_penalty 1.2`: CTRL식 반복 억제.
- `--no_repeat_ngram_size 3`: 동일 3-gram 재등장 차단. 행 단위 반복 방지에 효과 큼.
- `--max_lines 14`: 생성 텍스트의 줄 수가 14에 도달하면 즉시 종료(소네트 형식 강제).

순서: repetition penalty → no_repeat_ngram → temperature → top_k → softmax → top_p → 샘플 → eos/줄 종료 체크.

#### 6.5 DPO 구현 요점

- 시퀀스 log-prob: `logits[:, :-1]`로 다음 토큰 예측, 라벨은 `input_ids[:, 1:]`. completion 마스크는 `position ≥ prompt_len AND non-pad`. 마스킹된 위치의 `gather` 합으로 `log π(y|x)`.
- 손실: `loss = -F.logsigmoid(β · ((logπ_pol_w - logπ_ref_w) - (logπ_pol_l - logπ_ref_l))).mean()`.
- 진단 지표: `pref_acc = (logits > 0).float().mean()` — winning을 정확히 선호하는 비율.
- 음성 샘플 (5종 휴리스틱):
  - `shuffle`(행 순서 셔플), `repeat`(임의 행 두 번 등장 → 다른 행 제거 = degeneracy 모사), `truncate`(뒷부분 절단 = 14행 미달), `mangle`(인접 단어 swap), `drop_punct`(구두점 제거).
- 자가생성 음성: SFT 모델로 prompt당 K개 completion을 높은 온도(기본 1.4)로 샘플링 → `predictions/self_negs.json` 캐시. 한 번만 만들고 재사용.
- LoRA 호환:
  - SFT가 이미 LoRA이면 → 그 LoRA 파라미터를 학습 대상으로 유지.
  - SFT가 풀 FT이면 → `--extra_lora`로 policy에만 새 LoRA를 추가로 주입 가능.
- reference는 동일 ckpt의 frozen 복사본. 메모리는 두 배가 되지만 LoRA 시 학습 파라미터만 최적화되므로 옵티마이저 상태 메모리는 무시 가능.

#### 6.6 앙상블 (reference-free)

CHRF 평가는 정답 없는 test 시점에서는 측정 불가. 그래서 reference-free 점수로 후보 선택:

`sonnet_quality_score(text, weights)`의 항목 (가중치 모두 양수, 점수 클수록 좋음):

| 항목 | 의미 |
|------|------|
| `line_count` | `-|lines - 14|` — 14행 정합 |
| `avg_len` | 평균 줄 길이가 30~50자 범위 밖이면 페널티 |
| `duplicate` | 동일 줄 등장 페널티 |
| `diversity` | 4-gram unique 비율 |
| `rhyme` | ABAB CDCD EFEF GG 스킴 기반, 같은 그룹 마지막 단어의 끝-2글자 일치 보너스 |
| `mono` | 가장 빈번한 단어가 전체의 8% 이상이면 페널티(단조 반복 억제) |

두 모드:
- `--mode select`: 각 ckpt에서 `--num_candidates` 후보를 샘플링해 모두 채점 후 argmax 채택.
- `--mode prob_avg`: 다중 ckpt의 next-token 확률을 평균낸 뒤 top-p 샘플링(vocab 동일성 필요 — 전부 GPT-2면 OK).

#### 6.7 하이퍼파라미터 스윕

- `scripts/sweep_decoding.py` — *재학습 없이* 체크포인트 1개로 (temperature × top_p × top_k × rep_penalty × no_repeat_ngram_size) 그리드를 dev CHRF로 평가, CSV로 정렬 저장. 가장 ROI 높은 스윕.
- `scripts/sweep_train.sh` — 학습 그리드(LoRA rank × lr × dropout × weight_decay × model_size). 결과는 `predictions/sweeps/<TAG>.json`에 누적.

#### 6.8 실행 명령 모음 (서버 A100 MIG)

```bash
# (1) LoRA + 정규화 강화 베이스라인 (gpt2-large)
python sonnet_generation.py --use_gpu \
  --model_size gpt2-large \
  --use_lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --grad_checkpoint \
  --batch_size 2 --epochs 8 --lr 2e-4 \
  --hidden_dropout 0.1 --attn_dropout 0.1 --weight_decay 0.01 \
  --temperature 1.1 --top_p 0.9 --repetition_penalty 1.2 \
  --no_repeat_ngram_size 3 --max_lines 14 \
  --patience 3

# (2) 디코딩만 스윕 (가장 저렴, 즉시 CHRF 상승 기대)
python scripts/sweep_decoding.py --use_gpu \
  --ckpt gpt2-large-lora-8ep-0.0002-sonnet.pt \
  --temperatures 0.7,0.9,1.0,1.1,1.2 \
  --top_ps 0.85,0.9,0.95 \
  --rep_penalties 1.0,1.1,1.2 \
  --no_repeat_ngrams 0,3,4 \
  --max_lines 14

# (3) 자가생성 음성 캐시 (한 번)
python dpo.py --use_gpu --sft_ckpt gpt2-large-lora-8ep-0.0002-sonnet.pt \
  --gen_self_negs --num_self_negs 3 --self_neg_temperature 1.4 \
  --self_neg_cache predictions/self_negs.json --epochs 0  # gen만 하고 학습 0

# (4) DPO 학습
python dpo.py --use_gpu --sft_ckpt gpt2-large-lora-8ep-0.0002-sonnet.pt \
  --neg_modes shuffle,repeat,truncate,self \
  --self_neg_cache predictions/self_negs.json \
  --beta 0.1 --lr 5e-5 --epochs 5 --batch_size 2 \
  --weight_decay 0.01 --patience 2 \
  --temperature 1.0 --top_p 0.9 --repetition_penalty 1.2 \
  --no_repeat_ngram_size 3 --max_lines 14 \
  --filepath gpt2-large-lora-dpo.pt

# (5) 앙상블 + 최종 제출 파일 생성
python ensemble.py --use_gpu \
  --ckpts gpt2-large-lora-8ep-0.0002-sonnet.pt,gpt2-large-lora-dpo.pt \
  --mode select --num_candidates 8 \
  --temperature 1.0 --top_p 0.9 --repetition_penalty 1.2 \
  --no_repeat_ngram_size 3 --max_lines 14 \
  --sonnet_out predictions/generated_sonnets.txt

# (6) 학습 그리드 (시간 허락하는 만큼)
LOG_DIR=predictions/sweeps bash scripts/sweep_train.sh
```

#### 6.9 검증

- 모든 신규/수정 파일 `ast.parse` 통과.
- 기존 베이스라인은 추가 플래그 없이 실행하면 단계 4와 동일 동작(LoRA off, dropout 0.1, weight_decay 0, 새 디코딩 옵션 모두 비활성, ckpt 파일명만 새 태깅 규칙으로 바뀜).
- 베이스라인 ckpt(`10-1e-05-sonnet.pt`) 로드 호환: `getattr(args, 'use_lora', False)` 등 기본값 덕에 옛 ckpt도 그대로 `SonnetGPT(saved['args'])` 로 복원 가능.

#### 6.10 다음 액션 (사람 차례)

1. 서버에서 (1) 학습 → (2) 디코딩 스윕으로 즉효 CHRF 측정.
2. 좋은 디코딩 설정을 잠그고 (3)+(4) DPO 진행.
3. (5)에서 SFT+DPO 두 ckpt를 묶어 앙상블 제출.
4. 시간 남으면 (6) 학습 그리드로 LoRA rank/HP 탐색.


### 단계 7. Full Fine-tuning 상한선 측정 (방향 전환)

DPO/앙상블까지 한 번 훑은 뒤, **풀 FT 자체의 상한선을 먼저 깔끔하게 잡고
그 위에 최적화 기법들을 쌓아 올리자**는 판단으로 단계 6의 일부 결과는
보류하고 풀 FT 베이스라인 다시 정립 단계로 돌아왔다.

#### 7.1 동기와 설계 결정

단계 4의 베이스라인은 `gpt2-small` 단 한 점, 단일 시드, **이전 디코딩 기본값**
(`temperature=1.2`, no repetition penalty, max_lines=0) 으로 측정된 40.51 이다.
(REPORT에 빠져있던 후속 run: 같은 설정에서 epochs=15로 늘려 41.96 @ epoch 11 — 단순히
epoch 확장만으로 +1.45가 가능하다는 신호. ckpt `gpt2-fullft-15ep-lr1e-5-sonnet.pt` 서버에 보존.)

즉 풀 FT의 진짜 한계선이 아닌 *우연한 첫 점*에 가깝다. 보고서에 쓰일 베이스라인은
다음 조건을 만족해야 한다:

1. **모델 크기 sweep**: `gpt2` → `gpt2-medium` → 가능하면 `gpt2-large`(grad-ckpt) — 크기가
   CHRF에 얼마나 기여하는지 분리 측정.
2. **lr sweep**: 모델 크기마다 적정 lr이 다르므로 각 모델에 3점(1e-5/3e-5/5e-5).
3. **시드 분산**: 131편짜리 train은 분산이 크다. best lr 확정 후 시드 2개로 추가 측정.
4. **디코딩 고정**: `temperature=1.0, top_p=0.9, repetition_penalty=1.2,
   no_repeat_ngram_size=3, max_lines=14` — 학습 변인만 비교되도록 디코딩은 잠근다.
5. **체크포인트 보존**: 각 run의 ckpt/log/생성텍스트가 같은 tag로 묶이고 서로
   덮어쓰지 않아야 한다.

#### 7.2 인프라 변경 (보고서 작성 가능하도록 ckpt/로그 정리)

| 파일 | 변경 |
|------|------|
| `sonnet_generation.py` | tag에 **seed 항상 포함**, `weight_decay>0`/dropout이 기본과 다르면 자동 노출. ckpt는 **`predictions/ckpts/<tag>-sonnet.pt`** 로 모음 (repo 루트가 깨끗해짐, pod 재시작 시 한 디렉터리만 동기화하면 됨). 학습 시작 시 tag/ckpt/log/out 경로를 콘솔에 1줄 인쇄 |
| `scripts/run_fullft_sweep.sh` | NEW. PHASE A(seed=11711 × 모델×lr=6 run) / PHASE B(시드 분산 4 run) / PHASE C(`gpt2-large` 1 run) 으로 나눈 sweep 러너. 각 run은 `predictions/train_<tag>.json` 존재 시 자동 skip → resume-safe. `DRY_RUN=1`로 명령 프리뷰. 스크립트 위치 기준으로 repo root에 cd. |
| `scripts/summarize_experiments.py` | NEW. `predictions/train_*.json` 스캔 → tag/모델/시드/lr/best_chrf/best_epoch/ckpt 경로 등으로 표 출력. `--csv`, `--filter key=value`, `--sort col`, `--cols short` 지원 |

체크포인트 네이밍 규칙(고정):

```
predictions/ckpts/{model_size}-{method}-s{seed}-{epochs}ep-{lr}[-wd{wd}][-hd{hd}-ad{ad}]-sonnet.pt
predictions/train_{tag}.json          # 학습 + dev CHRF history
predictions/generated_{tag}.txt       # 해당 ckpt로 생성한 held-out 제출 텍스트
```

예: `predictions/ckpts/gpt2-medium-fullft-s11711-15ep-3e-05-sonnet.pt`

#### 7.3 실행 매트릭스 (Phase A 우선)

| Run | model | seed | lr | bs | epochs | grad_ckpt | 비고 |
|-----|-------|-----:|---:|---:|------:|:---------:|------|
| A1  | gpt2  | 11711 | 1e-5 | 8 | 15 | × | 단계 4 베이스라인 재실행(디코딩 고정) |
| A2  | gpt2  | 11711 | 3e-5 | 8 | 15 | × | lr↑ |
| A3  | gpt2  | 11711 | 5e-5 | 8 | 15 | × | lr↑↑ |
| A4  | gpt2-medium | 11711 | 1e-5 | 4 | 15 | × | |
| A5  | gpt2-medium | 11711 | 3e-5 | 4 | 15 | × | medium 주력 후보 |
| A6  | gpt2-medium | 11711 | 5e-5 | 4 | 15 | × | |
| B-x | (Phase A 결과 보고 best lr 결정 후) gpt2 / gpt2-medium × seed {42, 2024} | | | | | × | 분산 평가 4 run |
| C1  | gpt2-large  | 11711 | 1e-5 | 1 | 10 | ✓ | OOM 가능, 안 되면 large는 LoRA 경로만 유지 |

실행 시간 추정(1g.10gb MIG, gpt2-small ≈ 10s/epoch 관측치 기준):

- Phase A: gpt2 3 run × ~3분 + gpt2-medium 3 run × ~25분 ≈ **약 1.5시간**
- Phase B: gpt2 2 run × 3분 + gpt2-medium 2 run × 25분 ≈ **약 1시간**
- Phase C: 1 run × ~45~90분 (OOM시 즉시 중단)

#### 7.4 실행 방법 (서버)

```bash
cd ~/team3_cs224n_gpt
# 백그라운드 + 로그를 파일로
nohup bash scripts/run_fullft_sweep.sh > predictions/sweep_A.log 2>&1 &
tail -f predictions/sweep_A.log     # 진행 모니터

# Phase A가 끝난 뒤
python scripts/summarize_experiments.py --cols short

# best lr를 찾아 Phase B 실행 (예: gpt2 best=3e-5, medium best=3e-5)
PHASE=B BEST_LR_SMALL=3e-5 BEST_LR_MEDIUM=3e-5 \
  nohup bash scripts/run_fullft_sweep.sh > predictions/sweep_B.log 2>&1 &

# (시도) gpt2-large 풀 FT
nohup PHASE=C bash scripts/run_fullft_sweep.sh > predictions/sweep_C.log 2>&1 &
```

#### 7.5 결과 표

**Phase A — 학습 중 dev_chrf (학습 디코딩: T=1.0, top_p=0.9, rep=1.2, nrn=3, max_lines=14)**

| Run | model | lr | best_epoch | best_dev_chrf |
|-----|-------|---:|-----------:|--------------:|
| A1  | gpt2  | 1e-5 | 11 | 36.60 |
| A2  | gpt2  | 3e-5 | 11 | 38.59 |
| A3  | gpt2  | 5e-5 |  2 | 35.70 |
| A4  | gpt2-medium | 1e-5 |  8 | **40.17** ← 옛 best |
| A5  | gpt2-medium | 3e-5 | 14 | 39.52 |
| A6  | gpt2-medium | 5e-5 | 12 | 39.56 |

**디코딩 pitfall 발견** — 학습 디코딩이 CHRF를 ~5점 깎는다는 사실 발견 → 사후 sweep_decoding으로 진짜 best 측정 (12 cfg 격자).

**Phase A — sweep_decoding 적용 후 진짜 best (전 ckpt, 디코딩별 best 선택)**

| Run | model | lr | best decode (T, rep, nrn) | best_dev_chrf | 옛 → 새 |
|-----|-------|---:|---|--------------:|---:|
| A1  | gpt2  | 1e-5 | 1.1, 1.0, 0 | 41.13 | +4.53 |
| A2  | gpt2  | 3e-5 | 1.1, 1.0, 3 | 41.48 | +2.89 |
| A3  | gpt2  | 5e-5 | 0.9, 1.0, 3 | 41.54 | +5.84 |
| A4  | gpt2-medium | 1e-5 | 1.1, 1.0, 3 | 41.44 | +1.27 |
| A5  | gpt2-medium | 3e-5 | 1.2, 1.0, 3 | 41.22 | +1.70 |
| A6  | gpt2-medium | 5e-5 | 0.9, 1.0, 3 | **42.17** ← **새 best** | +2.61 |

**관찰**: 모든 6 ckpt에서 best 디코딩이 `rep_penalty=1.0` — Sonnet/시는 어휘 반복이 자연스러워 rep_penalty가 CHRF를 깎음. lr 순위가 뒤집힘 (medium best: 1e-5 → 5e-5).

**Phase B — best (model, lr) 셋업 시드 분산 (sweep_decoding 후)**

| Run | model | lr | seed | best_dev_chrf |
|-----|-------|---:|---:|--------------:|
| B1 | gpt2 | 3e-5 | 42   | 42.03 |
| B2 | gpt2 | 3e-5 | 2024 | 42.11 |
| B3 | gpt2-medium | 5e-5 | 42   | 41.98 |
| B4 | gpt2-medium | 5e-5 | 2024 | 41.71 |

**시드 분산 통계 (Phase A best + Phase B):**

| 조합 | mean | std | range |
|---|---:|---:|---|
| small lr=3e-5 (3 seeds) | **41.87** | 0.34 | 41.48~42.11 |
| medium lr=5e-5 (3 seeds) | **41.95** | 0.23 | 41.71~42.17 |

**핵심 관찰**: small(124M)과 medium(355M)의 풀 FT 평균이 통계적으로 동일 (0.08 차이, 둘 다 σ≈0.3). **131편 train + 12편 dev**의 데이터-bound 영역으로, 모델 크기를 키워도 CHRF 증가 없음. 

**Phase C (gpt2-large)는 데이터-bound 결론에 따라 스킵.**

확정된 풀 FT 상한선: 약 **41.9 ± 0.3 CHRF** (small 또는 medium 동등). 이후 모든 정규화 기법은 이 점을 베이스로 측정.

#### 7.6 풀 FT 다음에 시도할 최적화 기법 (PDF 7.4 정렬)

본 단계에서 상한선이 잡히면, 그 위에 다음을 순차로 쌓는다:

1. **SMART 류 정규화** (smoothness-inducing adversarial reg + Bregman proximal) — 131편 데이터에선 가장 효과가 큰 후보.
2. **LoRA / PEFT** (이미 구현, 단계 6) — `gpt2-large`가 풀 FT로 안 들어가도 large 활용 경로.
3. **2nd-order Optimizer** (Shampoo / SOAP / K-FAC) — 같은 epoch에서 더 빠른 수렴.
4. **Quantization** (GPTQ post-training 또는 quant-aware FT) — 정규화 효과 + 추론 비용↓.
5. **FlashAttention / Sliding Window Attention** — 자체 GPT-2 구현이라 패치 비용 큼, 후순위.
6. **DPO + Ensemble** — 단계 6 자산을 재사용해 최종 제출 단계에서.

각 기법의 baseline은 단계 7의 best 풀 FT ckpt를 그대로 사용한다 → 비교가 깔끔해진다.


### 단계 8. Phase D — Consistency Regularization (R-Drop + SMART)

데이터-bound 결론(단계 7.5)에 따라 모델 크기 확장 대신 **정규화 기법**으로 일반화를 개선하는 방향으로 전환. 두 가지 consistency regularizer를 구현·실험.

#### 8.1 동기 및 손실 식

131편의 매우 작은 train + 12편의 noisy dev. 풀 FT는 빠르게 train loss 0 근방으로 수렴 (과적합). 두 기법 모두 **같은 입력을 두 번 흘려 두 출력의 분포가 일치하도록 강제**해 모델이 perturbation에 안정적이게 만드는 정규화.

**R-Drop (Liang et al. 2021)**: 입력 x를 두 번 forward → dropout이 서로 다른 sampling → logits p₁, p₂.
```
L_R-Drop = ½(CE(p₁, y) + CE(p₂, y)) + λ · ½(KL(p₁‖p₂) + KL(p₂‖p₁))
```

**SMART (Jiang et al. 2020)**: 임베딩 e(x)에 adversarial perturbation δ를 1-step grad ascent로 찾음 → p_pert.
```
L_SMART = CE(p(e(x)), y) + λ · ½(KL(p(e(x))‖p(e(x)+δ)) + KL(p(e(x)+δ)‖p(e(x))))
δ ← Π_ε(δ + α · ∇_δ KL_sym), ‖δ‖ ≤ ε (per-token L2 ball)
```

KL은 token-level mean. R-Drop은 forward 2회, SMART(steps=1)는 forward 3회 (clean + δ-grad + final-perturbed) — wall-clock 비용은 baseline 대비 R-Drop ~1.7x, SMART ~1.3x.

#### 8.2 구현 (`sonnet_generation.py`)

`SonnetGPT.forward_from_embeds(embeds, attention_mask)` 신설 (SMART의 perturbed forward 진입점), `_compute_step_loss(model, b_ids, b_mask, args)` 함수로 baseline / R-Drop / SMART / 둘 다 분기. CLI: `--use_rdrop --rdrop_lambda`, `--use_smart --smart_lambda --smart_eps --smart_alpha --smart_steps`. tag/log에 method 노출.

#### 8.3 R-Drop λ sweep (single seed=11711, small lr=3e-5)

| λ   | best_dev_chrf | best_ep | 비고 |
|----:|--------------:|--------:|------|
| 0.2 | 41.57 |  4 | |
| 0.5 | 40.75 |  1 | best_ep=1 outlier, early stop |
| **1.0** | **41.64** | **10** | |
| 2.0 | 41.05 |  5 | |
| 5.0 | 41.13 | 13 | last_ep까지 학습 |

U-shape: λ=1.0이 sweet spot. 양쪽으로 단조 감소.

#### 8.4 SMART λ sweep (single seed=11711, small lr=3e-5, ε=1e-5, α=1e-3, steps=1)

| λ   | best_dev_chrf | best_ep |
|----:|--------------:|--------:|
| 0.5 | 41.59 | 12 |
| **1.0** | **41.74** | **8** |
| 2.0 | 41.61 |  5 |
| 5.0 | 41.61 | 12 |

R-Drop과 동일한 U-shape, λ=1.0이 best.

#### 8.5 SMART perturbation grid (λ=1.0 고정, single seed=11711, small lr=3e-5)

| ε    | α    | steps | best_dev_chrf | best_ep |
|-----:|-----:|------:|--------------:|--------:|
| 1e-6 | 1e-7 | 1 | 42.03 | 11 |
| 1e-5 | 1e-3 | 1 | 41.74 |  8 |
| 1e-4 | 1e-5 | 1 | 42.05 | 11 |
| 1e-3 | 1e-4 | 1 | 41.55 | 10 |
| **1e-2** | **1e-3** | **1** | **42.26** | **14** (last) |
| 1e-5 | 1e-3 | 2 | 41.68 |  6 |
| 1e-3 | 1e-4 | 2 | 41.89 | 14 |

**관찰**:
- ε ∈ {1e-6, 1e-5, 1e-4}는 동일한 학습 trajectory를 사실상 공유 (ce/smart_kl이 epoch별로 거의 일치). 임베딩 norm (≈10~50) 대비 너무 작아 perturbation 효과 무시 수준.
- ε=1e-3에서 처음으로 trajectory 분기 (best 41.55로 떨어짐, 일시적 outlier).
- ε=1e-2에서 다시 best 회복 + 사상 최고 42.26 — 다만 best_ep=14 (last)로 학습 부족 가능성.
- steps=2 효과 미미 (+0.05 ~ -0.06).

#### 8.6 R-Drop + SMART 결합 (single seed=11711, small lr=3e-5)

| Config | batch | best_dev_chrf | best_ep |
|--------|------:|--------------:|--------:|
| both (λ_r=1, λ_s=1, ε=1e-5, s=1) | 8 → OOM | — | — |
| both (재실행, batch=4) | 4 | 41.68 | 10 |

batch=8에서 NVML/CUDA allocator OOM (R-Drop 그래프 2개 + SMART 그래프 2개 동시 유지 → 메모리 압박). batch=4로 줄여 학습 OK, 단독보다 우위 없음.

#### 8.7 best 셋업 시드 분산 + medium 확장 (sweep_decoding으로 진짜 best CHRF)

각 ckpt에 동일 12 cfg 디코딩 sweep을 다시 적용해 학습-디코딩 cofounder 제거.

**small (lr=3e-5, batch=8) SMART λ=1, ε=1e-2 — 시드 3개:**

| seed | best decode | best_dev_chrf |
|---:|---|--------------:|
| 11711 | T=1.1, rep=1.0, nrn=3 | 41.69 |
| 42    | T=1.1, rep=1.0, nrn=3 | 41.33 |
| 2024  | T=1.1, rep=1.0, nrn=0 | 42.03 |
| **mean ± std** | | **41.68 ± 0.36** |

**small (lr=3e-5, batch=8) R-Drop λ=1 — 시드 3개:**

| seed | best decode | best_dev_chrf | 비고 |
|---:|---|--------------:|------|
| 11711 | T=1.2, rep=1.0, nrn=0 | 42.03 | |
| 42    | T=0.9, rep=1.0, nrn=3 | 40.70 | |
| 2024  | T=0.9, rep=1.0, nrn=0 | 40.23 | best_ep=0 outlier (patience=4 한계) |
| **mean ± std** | | **40.99 ± 0.93** |

**medium (lr=5e-5, batch=2 + grad_checkpoint) seed=11711:**

| method | best decode | best_dev_chrf |
|--------|---|--------------:|
| SMART (λ=1, ε=1e-2) | T=1.1, rep=1.0, nrn=3 | 41.94 |
| R-Drop (λ=1)        | T=1.1, rep=1.0, nrn=0 | 41.38 |

⚠ medium 비교는 batch=2로 줄여 grad_ckpt까지 켠 셋업이라 batch=4였던 Phase A/B baseline (42.17)과 직접 비교 불공정.

#### 8.8 종합 비교 (sweep_decoding 후)

| 셋업 | n_seed | mean | std | Δ vs baseline |
|------|-------:|------:|----:|--------------:|
| small baseline (lr=3e-5, Phase A+B) | 3 | **41.87** | 0.34 | — |
| small + SMART (λ=1, ε=1e-2) | 3 | 41.68 | 0.36 | **−0.19** |
| small + R-Drop (λ=1)        | 3 | 40.99 | 0.93 | **−0.88** |
| medium baseline (lr=5e-5, b=4) | 3 | **41.95** | 0.23 | — |
| medium + SMART (λ=1, ε=1e-2, b=2 grad_ckpt) | 1 | 41.94 | — | −0.01 (batch cofounder) |
| medium + R-Drop (λ=1, b=2) | 1 | 41.38 | — | −0.57 (batch cofounder) |

#### 8.9 결론 및 한계

1. **R-Drop, SMART 모두 baseline 대비 통계적으로 유의미한 향상 없음.** SMART는 noise 안에 묻힘 (-0.19, σ≈0.35 안), R-Drop은 -0.88로 평균이 명확히 떨어지나 s=2024 outlier (best_ep=0) 영향 큼.
2. **Single-seed 결과의 함정**: 학습 중 평가에서 SMART ε=1e-2가 best 42.26 → sweep_decoding 후 시드 분산에서 41.68로 회귀. **dev 12편**의 σ≈0.3~0.9가 어떤 method 차이도 묻어버림.
3. **medium 비교 한계**: SMART/R-Drop가 medium batch=4에서 OOM → batch=2 + grad_ckpt로 학습. baseline은 batch=4였음 → cofounder 분리 불가능.
4. **본질 진단**: 131편 train + 12편 dev로는 모델 크기, 정규화 기법, 디코딩 등 어떤 축으로도 통계적으로 유의미한 향상 측정이 어렵다. CHRF 측정의 표준오차가 σ/√12 ≈ 0.1~0.3이라 method effect가 그 안에 들어옴.

**제안**:
- 더 큰 dev (예: train에서 추가로 hold-out, 또는 cross-validation 5-fold)로 측정 표준오차를 줄여야 SMART/R-Drop 효과 검증 가능.
- 데이터 증강 (다른 운율 패턴의 영문 시 corpus 보강, 또는 Shakespearean style transfer로 합성)이 모델/정규화 변경보다 효과 클 듯.
- 현재 셋업에서 보고서 결론: **모델 크기는 small 124M로 충분, lr/디코딩 튜닝이 R-Drop/SMART보다 효과적 (+5 CHRF 회수).**

#### 8.10 실행 자취 (스크립트 + ckpt)

- 코드 확장: `sonnet_generation.py` (585 → 731줄, 이후 Phase E에서 738줄).
- Wrapper 스크립트: `scripts/phaseD1_consistency.sh`, `phaseD2_rdrop_lambda.sh`, `phaseD3_smart_lambda.sh`, `phaseD4_smart_perturb.sh`, `phaseD5_seed_medium.sh`, `phaseD5_medium_retry.sh`, `sweep_decoding_phaseD.sh`.
- 산출 ckpt 22개: `predictions/ckpts/gpt2-*fullft-*-(rdrop|smart)*.pt`.
- 산출 sweep CSV 8개: `predictions/decoding_sweeps/gpt2-*-(rdrop|smart)*.csv`.


### 단계 9. Phase E — 기본기 재점검 + Ensemble + Per-sonnet diagnostic

Phase D 결론(SMART/R-Drop이 dev 노이즈 안에 묻힘)을 받은 뒤,
(0) 코드 기본기 재점검·버그 수정, (A) ensemble로 noise 평균화 시도, (B) per-sonnet 진단으로 noise 출처 분해의 3단계 진행.

#### 9.0 코드 재점검에서 발견된 버그 두 개

**Bug 1 — `generate()`의 prompt 첫 3 글자 손실** (`sonnet_generation.py:225`, starter 코드 잔재):
```python
generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
```
GPT-2엔 BOS prepend가 없으므로 `[3:]`은 prompt의 첫 3 글자를 무조건 잘라낸다 ("From..." → "m...", "When..." → "n..."). CHRF는 character-level이라 직접 점수 손실. 같은 코드가 학습 dev 평가 / sweep_decoding / submission 모두에 일관 적용돼 **상대 비교는 valid**하지만 **절대 점수는 ~0.4 깎였음**.

수정: `[3:]` 제거. baseline small s11711 lr=3e-5 동일 ckpt에 sweep_decoding 다시 → **41.48 → 41.88** (+0.40), SMART s11711 ε=1e-2 → **41.69 → 42.12** (+0.43). 모든 sweep_decoding 결과에 약 +0.4 가산이 진짜 점수.

**Bug 2 — CE loss의 pad token masking 누락** (`_compute_step_loss`):
`pad_token = eos_token`이라 `ignore_index`로 분리 불가. 단순 `reduction='mean'`은 padding 위치(EOS 토큰 ID 50256)도 loss 평균에 포함시켜 모델이 "긴 시퀀스 끝에 EOS 출력" 패턴을 과잉 학습할 위험. 단 sonnet 데이터는 거의 동일 길이라 영향 작음.

수정: `_ce_lm_loss(logits, labels, mask=None)`에 mask 파라미터 추가, `_compute_step_loss`에서 `shift_mask = b_mask[:, 1:]`를 전달. 새 학습부터 적용 (Phase D 결과엔 미반영).

**Bug 3 (minor) — `generate_submission_sonnets`의 `max_length` 불일치**: dev 평가는 160, submission은 default 128. 14줄 sonnet에 부족할 수 있음. `max_length=getattr(args, 'dev_max_new_tokens', 160)`로 통일.

#### 9.A Ensemble — noise 평균화 + Phase D ckpt 재활용

**셋업**: `ensemble.py --mode select` (각 ckpt당 후보 K개 생성 → reference-free `sonnet_quality_score`로 best 선택). best decoding `T=1.1, top_p=0.9, rep=1.0, nrn=3, max_lines=14`로 고정.

| Ensemble | 구성 | 후보 수/prompt | dev CHRF |
|----------|------|--------------:|---------:|
| (A1) small × 3 seeds | small baseline {11711, 42, 2024} | 12 | **42.28** |
| (A2) (A1) + SMART s11711 | + Phase D small SMART ε=1e-2 | 16 | **42.39** |

비교 기준:
- 단일 best seed (small s2024 baseline, sweep_decoding 후 [3:] fix 반영 추정): ~42.5
- 시드 분산 mean baseline (fix 보정): ~42.3
- **Ensemble (A2): 42.39** — 단일 model의 시드 노이즈를 평균화하고, Phase D의 SMART가 만든 다른 stationary point를 보조 후보로 활용.

**중요 재해석 (Phase D 결과 vs Ensemble)**: Phase D에서 "SMART 단독은 noise 안에 묻힘"이었지만, **ensemble pool에 SMART 추가 시 +0.11 효과**. 즉 SMART가 만드는 다른 loss 표면 지점은 ensemble의 후보 풀에서는 가치 있음. 단독 비교(SMART vs baseline)와 결합 효과는 별개.

**Final test submission (gold 없음, 12편)**:
- `predictions/generated_sonnets_FINAL_3seeds.txt` — small × 3 seeds ensemble
- `predictions/generated_sonnets_FINAL_plusSmart.txt` — 위 + SMART (실제 best)

#### 9.B Per-sonnet diagnostic — corpus-level noise 분해

`scripts/per_sonnet_diag.py`: 각 (ckpt, prompt) 쌍의 CHRF를 한 편 단위로 측정 (sacreBLEU corpus_score를 1편 짜리 corpus로). 5 ckpt × 12 dev prompt = 60 평가.

| ckpt | mean CHRF (12편) | strict 14-lines / 12 |
|------|-----------------:|--------------------:|
| baseline s11711 | 41.66 | 7 |
| baseline s42    | 42.08 | 6 |
| baseline s2024  | 41.87 | 9 |
| SMART s11711    | **42.29** | 7 |
| R-Drop s11711   | 40.48 | 2 (13줄 포함하면 7) |

**Prompt별 method 분산 (range = max − min)**:

| sonnet_id | range | 메모 |
|----------:|------:|------|
| 132 | 2.69 | low variance, 다 비슷 |
| 133 | 1.28 | **가장 낮음** — method 차이 거의 없음 |
| 134 | 5.36 | **seed effect 큼** (baseline s11711=40.37, baseline s42=45.73) |
| 135 | 6.98 | 어려운 prompt (baseline s2024=35.70, 모든 셋업 낮음) |
| 136 | 7.45 | **가장 높음** — R-Drop=36.16, 다른 method 42+ |
| 142 | 5.49 | R-Drop=37.27 outlier |
| 평균 range | 3.99 | corpus 평균이 0.5 변하려면 prompt 변화가 ~4 / √12 ≈ 1.2 정도 필요 |

**핵심 진단**:
1. **prompt-level CHRF 분산이 corpus-level method-effect 차이보다 한 자릿수 큼**. 12편 dev에서 method 평균이 0.5점 다른 게 통계적 신호인지 noise인지 분리 불가. corpus σ ≈ prompt-level σ / √n_prompts ≈ 4/√12 ≈ 1.2.
2. **Same-method seed effect도 매우 큼** (sonnet 134에서 +5.36 within baseline). 단일 시드 결과 신뢰 ↓ 재확인.
3. **R-Drop의 형식 (strict 14-lines)이 baseline보다 약함** (2 vs 7), 다만 13줄 허용 시 비슷.

#### 9.C 최종 결론 + 진짜 점수 보정

- Phase A/B/D 모든 sweep_decoding 결과에 **약 +0.4** 가산이 진짜 절대 점수 (단, 상대 비교는 변함 없음).
- 진짜 baseline 상한선: **small/medium ≈ 42.3 ± 0.3** ([3:] fix 후 시드 분산 추정).
- **Best practical setup = Ensemble (small × 3 seeds + SMART)** = dev CHRF **42.39**. 단일 모델 어떤 셋업보다 0.1~0.3 정도 우위. 단 효과 크기가 12편 dev의 √n 표준오차 (~0.6) 안 → 진정한 강점은 단일 시드 의존 제거.
- SMART/R-Drop 단독은 dev noise 안에 묻히지만, **SMART는 ensemble 보조 후보로는 가치 있음**. R-Drop은 ensemble pool에서도 시드 안정성 떨어져 (per-sonnet diag) 제외.
- 데이터 12편이 모든 정밀 비교의 본질적 병목. Cross-validation (5-fold)나 데이터 증강 없이는 추가 method effect 측정 어려움.

#### 9.D Phase E 실행 자취

- 코드 수정 3건: `sonnet_generation.py` `[3:]` 제거 (line 225), `_ce_lm_loss` mask 인자 추가, `generate_submission_sonnets`의 max_length 전달.
- 스크립트 신설: `scripts/per_sonnet_diag.py`, `scripts/eval_chrf.py`, `scripts/eval_ensemble_phaseE_v2.sh`, `scripts/sweep_decoding_e0_verify.sh`.
- 산출물:
  - `predictions/decoding_sweeps_e0/{baseline,smart}.csv` — fix 효과 검증
  - `predictions/ensemble_small3seeds_dev.txt`, `predictions/ensemble_small3plusSmart_dev.txt` — dev ensemble
  - `predictions/generated_sonnets_FINAL_3seeds.txt`, `predictions/generated_sonnets_FINAL_plusSmart.txt` — test submission (dry-run)
  - `predictions/per_sonnet_diag.csv` — 60 rows (5 ckpt × 12 prompt)

