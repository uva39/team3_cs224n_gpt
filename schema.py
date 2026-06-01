import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union


@dataclass
class SaveInfo:
    dataset: str = ""
    seed: Optional[int] = None
    fine_tune_mode: str = ""
    lr: Optional[float] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    hidden_dropout_prob: Optional[float] = None

    checkpoint_path: str = ""
    prediction_prefix: str = ""

    best_epoch: Optional[int] = None
    best_dev_acc: Optional[float] = None
    best_dev_f1: Optional[float] = None

    final_train_acc: Optional[float] = None
    final_train_f1: Optional[float] = None
    final_dev_acc: Optional[float] = None
    final_dev_f1: Optional[float] = None

    dev_out: Union[str, Path] = ""
    test_out: Union[str, Path] = ""
    summary_out: Union[str, Path] = ""
    metrics_out: Union[str, Path] = ""
    
    
    weight_decay: Optional[float] = None,
    max_grad_norm: Optional[float] = None
    unuse_schedule: Optional[bool] = None
    warmup_ratio: Optional[float] = None
    use_simple_classifier: Optional[bool] = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise KeyError(f"Unknown SaveInfo field: {key}")
            setattr(self, key, value)

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        clean_dict = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(self).items()
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean_dict, f, indent=4, ensure_ascii=False)