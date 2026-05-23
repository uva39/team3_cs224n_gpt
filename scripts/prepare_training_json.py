#!/usr/bin/env python3

import csv
import json
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "json"


def normalize_sentiment_text(text):
    return text.lower().strip()


def normalize_paraphrase_text(text):
    return " ".join(
        text.lower()
        .replace(".", " .")
        .replace("?", " ?")
        .replace(",", " ,")
        .replace("'", " '")
        .split()
    )


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def read_row_index(row, fallback_index):
    for key in ("", "0"):
        value = row.get(key)
        if value is None:
            continue
        value = value.strip()
        if value.isdigit():
            return int(value)
    return fallback_index


def convert_sentiment_csv(path, dataset_name, split, has_label):
    examples = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            text_raw = row["sentence"].strip()
            example = {
                "task": "sentiment",
                "dataset": dataset_name,
                "split": split,
                "row_index": read_row_index(row, len(examples)),
                "id": row["id"].strip(),
                "text_raw": text_raw,
                "text": normalize_sentiment_text(text_raw),
            }
            if has_label:
                example["label"] = int(float(row["sentiment"]))
            examples.append(example)

    return {
        "task": "sentiment",
        "dataset": dataset_name,
        "split": split,
        "source_file": path.name,
        "num_examples": len(examples),
        "examples": examples,
    }


def convert_paraphrase_csv(path, split, has_label):
    examples = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            sentence1_raw = row["sentence1"].strip()
            sentence2_raw = row["sentence2"].strip()
            example = {
                "task": "paraphrase",
                "dataset": "quora",
                "split": split,
                "id": row["id"].strip(),
                "sentence1_raw": sentence1_raw,
                "sentence2_raw": sentence2_raw,
                "sentence1": normalize_paraphrase_text(sentence1_raw),
                "sentence2": normalize_paraphrase_text(sentence2_raw),
            }
            if has_label:
                example["label"] = int(float(row["is_duplicate"]))
            examples.append(example)

    return {
        "task": "paraphrase",
        "dataset": "quora",
        "split": split,
        "source_file": path.name,
        "num_examples": len(examples),
        "examples": examples,
    }


def parse_sonnets(text):
    parts = re.split(r"\n\s*(\d+)\s*\n", text)
    sonnets = []
    for i in range(1, len(parts), 2):
        sonnet_id = int(parts[i])
        body = parts[i + 1].strip()
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        sonnets.append({"id": sonnet_id, "lines": lines})
    return sonnets


def convert_sonnets_txt(path, split, has_full_text):
    text = path.read_text(encoding="utf-8")
    entries = []
    for sonnet in parse_sonnets(text):
        lines = sonnet["lines"]
        prompt_lines = lines[:3]
        example = {
            "task": "generation",
            "dataset": "sonnets",
            "split": split,
            "id": sonnet["id"],
            "prompt_lines": prompt_lines,
            "prompt_text": "\n".join(prompt_lines),
            "num_prompt_lines": len(prompt_lines),
        }
        if has_full_text:
            completion_lines = lines[3:]
            example.update(
                {
                    "lines": lines,
                    "text": "\n".join(lines),
                    "completion_lines": completion_lines,
                    "completion_text": "\n".join(completion_lines),
                    "num_lines": len(lines),
                }
            )
        entries.append(example)

    return {
        "task": "generation",
        "dataset": "sonnets",
        "split": split,
        "source_file": path.name,
        "num_examples": len(entries),
        "examples": entries,
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    datasets = {
        "ids-sst-train.json": convert_sentiment_csv(DATA_DIR / "ids-sst-train.csv", "sst", "train", True),
        "ids-sst-dev.json": convert_sentiment_csv(DATA_DIR / "ids-sst-dev.csv", "sst", "dev", True),
        "ids-sst-test-student.json": convert_sentiment_csv(
            DATA_DIR / "ids-sst-test-student.csv", "sst", "test", False
        ),
        "ids-cfimdb-train.json": convert_sentiment_csv(DATA_DIR / "ids-cfimdb-train.csv", "cfimdb", "train", True),
        "ids-cfimdb-dev.json": convert_sentiment_csv(DATA_DIR / "ids-cfimdb-dev.csv", "cfimdb", "dev", True),
        "ids-cfimdb-test-student.json": convert_sentiment_csv(
            DATA_DIR / "ids-cfimdb-test-student.csv", "cfimdb", "test", False
        ),
        "quora-train.json": convert_paraphrase_csv(DATA_DIR / "quora-train.csv", "train", True),
        "quora-dev.json": convert_paraphrase_csv(DATA_DIR / "quora-dev.csv", "dev", True),
        "quora-test-student.json": convert_paraphrase_csv(DATA_DIR / "quora-test-student.csv", "test", False),
        "sonnets.json": convert_sonnets_txt(DATA_DIR / "sonnets.txt", "train", True),
        "sonnets_held_out.json": convert_sonnets_txt(DATA_DIR / "sonnets_held_out.txt", "test_prompt", False),
        "sonnets_held_out_dev.json": convert_sonnets_txt(DATA_DIR / "sonnets_held_out_dev.txt", "dev_prompt", False),
        "TRUE_sonnets_held_out_dev.json": convert_sonnets_txt(
            DATA_DIR / "TRUE_sonnets_held_out_dev.txt", "dev_gold", True
        ),
    }

    manifest = {
        "output_directory": str(OUTPUT_DIR.relative_to(ROOT_DIR)),
        "datasets": [],
    }

    for file_name, payload in datasets.items():
        write_json(OUTPUT_DIR / file_name, payload)
        manifest["datasets"].append(
            {
                "file_name": file_name,
                "task": payload["task"],
                "dataset": payload["dataset"],
                "split": payload["split"],
                "source_file": payload["source_file"],
                "num_examples": payload["num_examples"],
            }
        )

    write_json(OUTPUT_DIR / "manifest.json", manifest)


if __name__ == "__main__":
    main()
