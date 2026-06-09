#!/usr/bin/env python3

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets import load_paraphrase_data


TWO_SHOT_EXAMPLES = [
    (
        "how do i start learning about artificial intelligence ?",
        "how do you learn artificial intelligence ?",
        "yes",
    ),
    (
        "how can someone learn biochemistry using first principles thinking ?",
        "how can someone learn neuroscience using first principles thinking ?",
        "no",
    ),
]

SIX_SHOT_EXAMPLES = [
    (
        "how do i start learning about artificial intelligence ?",
        "how do you learn artificial intelligence ?",
        "yes",
    ),
    (
        "how can someone learn biochemistry using first principles thinking ?",
        "how can someone learn neuroscience using first principles thinking ?",
        "no",
    ),
    (
        "what should i do to get selected in gsoc 2018 ?",
        "i 'm a rookie . how should i start preparing for getting selected in gsoc 17 ?",
        "yes",
    ),
    (
        "how do i view a private broadcast on periscope ?",
        "who is alexander khan on periscope ?",
        "no",
    ),
    (
        "what are some examples of genotypes and phenotypes ?",
        "what are genotypes and phenotypes ? what are examples of this ?",
        "yes",
    ),
    (
        "what are the best companies for android developer in chennai ?",
        "is there any best android app development company in hyderabad ?",
        "no",
    ),
]


MODEL_BATCH_SIZES = {
    "gpt2": 32,
    "gpt2-medium": 16,
    "gpt2-large": 8,
}


def format_question(sent1, sent2, answer=None):
    prompt = f'Is "{sent1}" a paraphrase of "{sent2}"? Answer "yes" or "no": '
    if answer is not None:
        prompt += answer
    return prompt


def build_few_shot_prefix(examples):
    return "\n".join(format_question(sent1, sent2, answer) for sent1, sent2, answer in examples) + "\n"


def build_prompt(prefix, sent1, sent2):
    return prefix + format_question(sent1, sent2)


def collate_prompts(batch, prefix):
    prompts = [build_prompt(prefix, sent1, sent2) for sent1, sent2, _, _ in batch]
    labels = torch.tensor([label for _, _, label, _ in batch], dtype=torch.long)
    return prompts, labels


def compute_metrics(labels, preds):
    labels = torch.tensor(labels, dtype=torch.long)
    preds = torch.tensor(preds, dtype=torch.long)

    tn = int(((labels == 0) & (preds == 0)).sum().item())
    fp = int(((labels == 0) & (preds == 1)).sum().item())
    fn = int(((labels == 1) & (preds == 0)).sum().item())
    tp = int(((labels == 1) & (preds == 1)).sum().item())

    def safe_div(num, den):
        return num / den if den else 0.0

    precision_0 = safe_div(tn, tn + fn)
    recall_0 = safe_div(tn, tn + fp)
    f1_0 = safe_div(2 * precision_0 * recall_0, precision_0 + recall_0)

    precision_1 = safe_div(tp, tp + fp)
    recall_1 = safe_div(tp, tp + fn)
    f1_1 = safe_div(2 * precision_1 * recall_1, precision_1 + recall_1)

    acc = safe_div(tp + tn, len(labels))
    macro_f1 = (f1_0 + f1_1) / 2
    pred_pos_rate = safe_div(int((preds == 1).sum().item()), len(preds))
    true_pos_rate = safe_div(int((labels == 1).sum().item()), len(labels))

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "pred_pos_rate": pred_pos_rate,
        "true_pos_rate": true_pos_rate,
        "precision_0": precision_0,
        "recall_0": recall_0,
        "f1_0": f1_0,
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_model(model_name, dataset, prefix, device, dtype, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()

    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_prompts(batch, prefix),
    )

    all_labels = []
    all_preds = []
    started = time.time()
    max_length = getattr(model.config, "n_positions", tokenizer.model_max_length)

    with torch.no_grad():
        for prompts, labels in tqdm(loader, desc=model_name):
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_positions = attention_mask.sum(dim=1) - 1
            batch_positions = torch.arange(input_ids.size(0), device=device)
            next_token_logits = outputs.logits[batch_positions, last_positions]
            yes_logits = next_token_logits[:, yes_id]
            no_logits = next_token_logits[:, no_id]
            preds = (yes_logits > no_logits).long().cpu()

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["seconds"] = time.time() - started
    metrics["batch_size"] = batch_size

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="data/quora-dev.csv")
    parser.add_argument("--out", default=None)
    parser.add_argument("--prompt-out", default=None)
    parser.add_argument("--models", nargs="+", default=["gpt2", "gpt2-medium", "gpt2-large"])
    parser.add_argument("--shots", choices=["2", "6"], default="2")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    examples = TWO_SHOT_EXAMPLES if args.shots == "2" else SIX_SHOT_EXAMPLES
    prefix = build_few_shot_prefix(examples)
    if args.out is None:
        args.out = f"logs/few_shot_{args.shots}_gpt2_paraphrase_dev.csv"
    if args.prompt_out is None:
        args.prompt_out = f"logs/few_shot_{args.shots}_gpt2_paraphrase_prompt.txt"

    prompt_preview = build_prompt(prefix, "[sentence1]", "[sentence2]")
    Path(args.prompt_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.prompt_out).write_text(prompt_preview, encoding="utf-8")

    print("Few-shot prompt template:")
    print(prompt_preview)
    print()
    print(f"device={device}, dtype={dtype}")

    dataset = load_paraphrase_data(args.dev, split="dev")
    if args.limit is not None:
        dataset = dataset[: args.limit]

    rows = []
    for model_name in args.models:
        batch_size = MODEL_BATCH_SIZES.get(model_name, 8)
        print(f"\n== evaluating {model_name} batch_size={batch_size} ==")
        metrics = evaluate_model(model_name, dataset, prefix, device, dtype, batch_size)
        row = {
            "model": model_name,
            "n": len(dataset),
            "shots": len(examples),
            **metrics,
        }
        rows.append(row)
        print(
            f'{model_name}: acc={metrics["acc"]:.4f}, macro_f1={metrics["macro_f1"]:.4f}, '
            f'pred_pos_rate={metrics["pred_pos_rate"]:.4f}, '
            f'cm=[[{metrics["tn"]},{metrics["fp"]}],[{metrics["fn"]},{metrics["tp"]}]], '
            f'seconds={metrics["seconds"]:.1f}'
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "n",
        "shots",
        "batch_size",
        "acc",
        "macro_f1",
        "pred_pos_rate",
        "true_pos_rate",
        "precision_0",
        "recall_0",
        "f1_0",
        "precision_1",
        "recall_1",
        "f1_1",
        "tn",
        "fp",
        "fn",
        "tp",
        "seconds",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {args.out}")
    print(f"wrote {args.prompt_out}")


if __name__ == "__main__":
    main()
