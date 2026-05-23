# 데이터 학습 전략

## 목적

이 폴더에는 과제에서 제공된 원본 데이터와, 학습에 바로 사용할 수 있도록 변환한 JSON 데이터가 함께 들어 있습니다.
이 문서의 목적은 다음 네 가지입니다.

- 각 task에서 어떤 파일을 사용해야 하는지 정리하기
- 원본 데이터를 어떤 규칙으로 JSON으로 바꿨는지 설명하기
- 학습을 어떤 순서로 진행하는 것이 좋은지 제안하기
- 전처리와 평가에서 주의할 점을 정리하기


## 폴더 구성

원본 데이터 파일:

- `ids-sst-train.csv`
- `ids-sst-dev.csv`
- `ids-sst-test-student.csv`
- `ids-cfimdb-train.csv`
- `ids-cfimdb-dev.csv`
- `ids-cfimdb-test-student.csv`
- `quora-train.csv`
- `quora-dev.csv`
- `quora-test-student.csv`
- `sonnets.txt`
- `sonnets_held_out.txt`
- `sonnets_held_out_dev.txt`
- `TRUE_sonnets_held_out_dev.txt`

전처리된 JSON 파일:

- `json/ids-sst-train.json`
- `json/ids-sst-dev.json`
- `json/ids-sst-test-student.json`
- `json/ids-cfimdb-train.json`
- `json/ids-cfimdb-dev.json`
- `json/ids-cfimdb-test-student.json`
- `json/quora-train.json`
- `json/quora-dev.json`
- `json/quora-test-student.json`
- `json/sonnets.json`
- `json/sonnets_held_out.json`
- `json/sonnets_held_out_dev.json`
- `json/TRUE_sonnets_held_out_dev.json`
- `json/manifest.json`


## JSON 생성 규칙

JSON 파일은 아래 스크립트로 생성합니다.

```bash
python scripts/prepare_training_json.py
```

변환 원칙:

- sentiment 데이터는 `text_raw`와 정규화된 `text`를 함께 저장합니다.
- paraphrase 데이터는 원문 문장과 정규화된 문장을 함께 저장합니다.
- sonnet 데이터는 생성 실험에 바로 쓸 수 있도록 `prompt_*`와 `completion_*` 필드로 나눠 둡니다.
- 모든 JSON 파일은 줄바꿈과 들여쓰기가 포함된 pretty JSON 형식으로 저장합니다.

주요 필드 규칙:

- sentiment label은 원본 데이터의 정수 라벨을 그대로 사용합니다.
- paraphrase label은 `0` 또는 `1`의 이진 정수입니다.
- sonnet prompt는 각 시의 앞 3줄을 기준으로 구성합니다.
- sonnet train 데이터는 전체 시와, 첫 3줄 이후의 continuation을 함께 저장합니다.


## Task 1: 감성 분류

사용 데이터:

- `json/ids-sst-train.json`
- `json/ids-sst-dev.json`
- `json/ids-sst-test-student.json`
- `json/ids-cfimdb-train.json`
- `json/ids-cfimdb-dev.json`
- `json/ids-cfimdb-test-student.json`

권장 사용 방식:

- 먼저 SST로 분류기와 optimizer가 정상적으로 동작하는지 확인합니다.
- 첫 실험에서는 이미 정규화된 `text`를 기본 입력으로 사용합니다.
- punctuation이나 casing 효과를 따로 보고 싶을 때만 `text_raw`를 사용합니다.
- 가장 먼저 볼 평가지표는 dev accuracy입니다.

권장 학습 순서:

1. SST에서 GPT backbone을 freeze한 상태로 baseline을 확인합니다.
2. SST에서 full-model fine-tuning을 진행합니다.
3. 같은 방식으로 CFIMDB에서도 실험합니다.
4. 예측 파일이 안정적으로 생성되면 다음 task로 넘어갑니다.

중점적으로 볼 항목:

- dev accuracy
- train loss가 안정적으로 내려가는지
- 데이터가 더 작은 CFIMDB에서 과적합이 심해지지 않는지


## Task 2: Paraphrase Detection

사용 데이터:

- `json/quora-train.json`
- `json/quora-dev.json`
- `json/quora-test-student.json`

권장 사용 방식:

- 기본 입력은 정규화된 `sentence1`, `sentence2`를 사용합니다.
- 에러 분석을 위해 `sentence1_raw`, `sentence2_raw`도 함께 보관합니다.
- train, dev, test에서 prompt template을 최대한 동일하게 맞춥니다.
- 예측이 full vocabulary 분포에서 나오는지, 아니면 `yes/no` 비교로 결정되는지 반드시 기록합니다.

권장 학습 순서:

1. `gpt2`와 작은 batch size, 보수적인 learning rate로 baseline을 먼저 만듭니다.
2. train/dev/test의 prompt 형식을 통일합니다.
3. `(s1, s2)`와 `(s2, s1)`를 모두 쓰는 sentence-order augmentation을 실험합니다.
4. 일반 fine-tuning과 `yes/no` token restricted decision 방식을 비교합니다.
5. dev 성능으로만 튜닝한 뒤 마지막에만 student test를 사용합니다.

중점적으로 볼 항목:

- dev accuracy
- 클래스 불균형 영향
- lexical overlap, negation, entity swap에서의 실패 사례


## Task 3: Sonnet Generation

사용 데이터:

- `json/sonnets.json`
- `json/sonnets_held_out_dev.json`
- `json/TRUE_sonnets_held_out_dev.json`
- `json/sonnets_held_out.json`

권장 사용 방식:

- `sonnets.json`은 학습용으로 사용합니다.
- `sonnets_held_out_dev.json`은 생성 품질을 튜닝할 때 prompt로 사용합니다.
- 생성 결과는 `TRUE_sonnets_held_out_dev.json`과 비교합니다.
- 최종 제출용 생성은 `sonnets_held_out.json`으로 진행합니다.

권장 학습 순서:

1. 전체 sonnet text를 이용해 language modeling baseline을 먼저 학습합니다.
2. held-out dev prompt로 생성 품질을 평가합니다.
3. temperature, top-p 같은 decoding 설정을 조정합니다.
4. 생성된 continuation을 gold dev sonnet과 비교합니다.
5. dev에서 설정이 안정화된 뒤에만 held-out test prompt로 최종 생성합니다.

중점적으로 볼 항목:

- 14행 형식이 유지되는지
- 반복이 심하지 않은지
- 너무 일찍 generation이 끝나지 않는지
- held-out dev 기준 chrF 점수


## 추천 작업 흐름

1. 학습에는 `data/json/` 아래 JSON 파일을 기본 입력으로 사용합니다.
2. 원본 CSV와 TXT 파일은 추적 가능성을 위해 그대로 유지합니다.
3. 실험 기록에는 model size, learning rate, batch size, epochs, seed, dev score를 반드시 남깁니다.
4. 모든 하이퍼파라미터 조정은 dev 기준으로만 진행합니다.
5. 전처리 규칙이 바뀌면 JSON 파일을 다시 생성합니다.


## 간단한 로딩 예시

```python
import json
from pathlib import Path

path = Path("data/json/quora-train.json")
payload = json.loads(path.read_text(encoding="utf-8"))
examples = payload["examples"]

first_example = examples[0]
print(first_example["sentence1"])
print(first_example["sentence2"])
print(first_example["label"])
```


## 참고 메모

- `json/manifest.json`을 보면 데이터셋 크기를 가장 빠르게 확인할 수 있습니다.
- 전처리 규칙을 바꾸면 `scripts/prepare_training_json.py`를 다시 실행해야 합니다.
- 같은 실험 안에서 raw text와 normalized text를 섞어 쓰지 않는 것이 좋고, 섞어 쓸 경우에는 반드시 명시적인 ablation으로 기록해야 합니다.
