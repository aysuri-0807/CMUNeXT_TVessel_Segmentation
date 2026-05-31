'''
Script to read the results file, and create a simplified CSV. 
'''

from __future__ import annotations

import ast
import csv
import re
import sys
from pathlib import Path


def extract_metric_dicts(log_text: str) -> list[dict]:
    """
    Extract Python-style metric dictionaries from a training log.

    Example dictionary:
    {'epoch': 1, 'train_loss': 0.767, 'val_loss': 0.808, ...}
    """

    # Finds anything that looks like {...}
    raw_dicts = re.findall(r"\{[^{}]*\}", log_text)

    parsed_rows = []

    for raw in raw_dicts:
        try:
            data = ast.literal_eval(raw)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        # Only keep dictionaries that look like metric logs
        if "epoch" in data and "train_loss" in data and "val_loss" in data:
            parsed_rows.append(data)

    return parsed_rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    """
    Write extracted metric rows to a CSV file.
    """

    if not rows:
        raise ValueError("No valid training metric dictionaries were found.")

    # Preferred graphing column order
    preferred_columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "dice",
        "iou",
        "sensitivity",
        "specificity",
        "accuracy",
    ]

    # Include any extra columns that may appear later
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    columns = preferred_columns + sorted(all_keys - set(preferred_columns))

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def parse_training_log(input_file: str, output_file: str) -> None:
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    log_text = input_path.read_text(encoding="utf-8", errors="ignore")

    rows = extract_metric_dicts(log_text)

    # Sort by epoch just in case the log is out of order
    rows.sort(key=lambda row: row.get("epoch", 0))

    write_csv(rows, output_path)

    print(f"Parsed {len(rows)} epochs.")
    print(f"Saved CSV to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parse_training_log.py <input_log.txt> <output_metrics.csv>")
        sys.exit(1)

    parse_training_log(sys.argv[1], sys.argv[2])