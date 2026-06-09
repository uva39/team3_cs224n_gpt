"""
LoRA (Low-Rank Adaptation) 구현.

본 프로젝트의 GPT-2는 HuggingFace 모델이 아니라 자체 구현(models/gpt2.py)이므로
HuggingFace `peft` 라이브러리를 그대로 붙일 수 없다. 그래서 nn.Linear를 감싸는
경량 래퍼(LoRALinear)와, 모델 트리를 순회하며 타깃 Linear를 LoRALinear로 교체하는
주입 함수(inject_lora)를 직접 구현한다.

LoRA 핵심 (Hu et al., 2021, https://arxiv.org/abs/2106.09685):
  W' x = W x + (alpha / r) * B (A x)
  - W 는 freeze (그래디언트 저장 X) → 학습 파라미터 수/메모리 급감
  - A: [r, in], B: [out, r] 만 학습. B는 0으로 초기화 → 학습 시작 시 delta=0 이라
    사전학습 가중치를 그대로 보존한 상태에서 출발한다.

이렇게 하면 gpt2-large(774M) 같은 큰 모델도 10GB MIG 슬라이스에서 finetuning 가능.
"""

import math
from typing import Iterable, Sequence

import torch
from torch import nn


# 기본 주입 대상: attention 의 q/k/v 와 출력 projection, 그리고 MLP 두 dense.
# 이름은 modules/attention.py, modules/gpt2_layer.py 의 속성명과 일치해야 한다.
DEFAULT_LORA_TARGETS = ("query", "key", "value", "attention_dense", "interm_dense", "out_dense")


class LoRALinear(nn.Module):
  """기존 nn.Linear 를 감싸 저랭크 보정항을 더하는 래퍼.

  base 의 가중치/bias 는 freeze 되고, lora_A/lora_B 만 학습된다.
  forward 시그니처(x -> y)는 nn.Linear 와 동일하므로 상위 모듈 수정 없이 교체 가능.
  """

  def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
    super().__init__()
    assert rank > 0, "LoRA rank must be > 0"
    self.base = base
    # 베이스 가중치는 freeze (그래디언트 미저장).
    for p in self.base.parameters():
      p.requires_grad = False

    self.in_features = base.in_features
    self.out_features = base.out_features
    self.rank = rank
    self.scaling = alpha / rank
    self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # A: [r, in], B: [out, r]. 베이스와 같은 dtype/device 에 생성.
    weight = base.weight
    self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features, dtype=weight.dtype, device=weight.device))
    self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=weight.dtype, device=weight.device))
    # A 는 kaiming, B 는 0 → 초기 delta = 0.
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result = self.base(x)
    # delta = (x @ A^T) @ B^T * scaling
    delta = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
    return result + self.scaling * delta

  def extra_repr(self) -> str:
    return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, scaling={self.scaling:.3f}"


def inject_lora(model: nn.Module,
                rank: int = 8,
                alpha: int = 16,
                dropout: float = 0.0,
                targets: Sequence[str] = DEFAULT_LORA_TARGETS) -> int:
  """model 의 하위 모듈을 순회하며, 이름이 `targets` 에 속하는 nn.Linear 를
  LoRALinear 로 교체한다. 교체한 레이어 수를 반환한다.

  순회 중 트리를 수정하면 위험하므로, 먼저 (parent, child_name, child) 를 모아 두고
  사후에 setattr 로 교체한다.
  """
  targets = set(targets)
  to_replace = []
  for parent_name, parent in model.named_modules():
    for child_name, child in parent.named_children():
      if child_name in targets and isinstance(child, nn.Linear):
        to_replace.append((parent, child_name, child))

  for parent, child_name, child in to_replace:
    setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))

  return len(to_replace)


def mark_only_lora_as_trainable(model: nn.Module, train_layernorm: bool = False) -> None:
  """LoRA 파라미터(lora_A/lora_B)만 학습 가능하게 두고 나머지는 모두 freeze.

  train_layernorm=True 면 LayerNorm 파라미터도 함께 학습한다(가끔 PEFT 안정화에 도움).
  """
  for name, param in model.named_parameters():
    if "lora_" in name:
      param.requires_grad = True
    elif train_layernorm and ("layer_norm" in name or "ln" in name):
      param.requires_grad = True
    else:
      param.requires_grad = False


def lora_state_dict(model: nn.Module) -> dict:
  """LoRA 파라미터만 담은 state_dict (체크포인트를 작게 저장하고 싶을 때)."""
  return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def count_trainable_parameters(model: nn.Module) -> tuple:
  """(trainable, total) 파라미터 수를 반환."""
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  total = sum(p.numel() for p in model.parameters())
  return trainable, total


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
  """옵티마이저에 넘길 학습 가능 파라미터만 yield."""
  return (p for p in model.parameters() if p.requires_grad)
