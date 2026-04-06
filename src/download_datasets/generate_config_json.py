from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import DATASET_CONFIGS
from src.dataloader_functions.notebook_common import classify_columns


# Upstream notebooks mostly use explicit_nunique_threshold=50,
# while a few datasets (notably calif_houses) are run with ratio-based threshold.
UPSTREAM_TYPE_DETECTION_RULES: dict[str, dict[str, float | int | None]] = {
    "calif_houses": {"unique_ratio_threshold": 0.05, "explicit_nunique_threshold": None},
}


def _get_classification_kwargs(dataset_name: str, default_explicit_nunique_threshold: int) -> dict[str, float | int | None]:
    rule = UPSTREAM_TYPE_DETECTION_RULES.get(dataset_name)
    if rule is not None:
        return {
            "unique_ratio_threshold": rule.get("unique_ratio_threshold"),
            "explicit_nunique_threshold": rule.get("explicit_nunique_threshold"),
        }
    return {
        "unique_ratio_threshold": None,
        "explicit_nunique_threshold": default_explicit_nunique_threshold,
    }


def _task_to_folder(task: str) -> str:
    if task == "clf":
        return "classification"
    if task == "reg":
        return "regression"
    raise ValueError(f"Unsupported task '{task}'. Use 'clf' or 'reg'.")


def _task_target_lookup(task: str) -> dict[str, str | None]:
    lookup: dict[str, str | None] = {}
    for dataset_name, cfg in DATASET_CONFIGS.items():
        if cfg.get("task") == task:
            lookup[dataset_name] = cfg.get("target")
    return lookup


def _ordered_subset(columns: list[str], allowed: set[str]) -> list[str]:
    return [col for col in columns if col in allowed]


def _build_dataset_entry(
    csv_path: Path,
    task: str,
    target_lookup: dict[str, str | None],
    explicit_nunique_threshold: int,
) -> tuple[str, dict]:
    dataset_name = csv_path.stem
    target_column = target_lookup.get(dataset_name)

    df = pd.read_csv(csv_path, low_memory=False)

    if target_column and target_column not in df.columns:
        # Keep the dataset in config but mark as auxiliary if target is unavailable.
        target_column = None

    feature_columns = [col for col in df.columns if col != target_column]
    feature_df = df[feature_columns].copy()

    classify_kwargs = _get_classification_kwargs(
        dataset_name=dataset_name,
        default_explicit_nunique_threshold=explicit_nunique_threshold,
    )
    numerical_columns, categorical_columns, text_columns = classify_columns(feature_df, **classify_kwargs)

    categorical_set = set(categorical_columns)
    numerical_set = set(numerical_columns)
    text_set = set(text_columns)

    # Preserve original feature order while keeping classification mutually exclusive.
    ordered_numerical = _ordered_subset(feature_columns, numerical_set)
    ordered_text = _ordered_subset(feature_columns, text_set)
    ordered_categorical = _ordered_subset(
        feature_columns,
        categorical_set - numerical_set - text_set,
    )

    entry = {
        "file_name": csv_path.name,
        "task_type": task,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "categorical_columns": ordered_categorical,
        "numerical_columns": ordered_numerical,
        "text_columns": ordered_text,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "has_processed_pickle": (csv_path.with_name(f"{dataset_name}_processed.pkl")).exists(),
        "is_auxiliary_file": target_column is None,
    }
    return dataset_name, entry


def generate_config_json(task: str, dataset_root: Path, explicit_nunique_threshold: int = 50) -> Path:
    folder = _task_to_folder(task)
    dataset_dir = dataset_root / folder
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    target_lookup = _task_target_lookup(task)

    datasets: dict[str, dict] = {}
    for csv_path in sorted(dataset_dir.glob("*.csv")):
        dataset_name, entry = _build_dataset_entry(
            csv_path=csv_path,
            task=task,
            target_lookup=target_lookup,
            explicit_nunique_threshold=explicit_nunique_threshold,
        )
        datasets[dataset_name] = entry

    config = {
        "group": folder,
        "generated_on": str(date.today()),
        "schema_version": 2,
        "datasets": datasets,
    }

    output_path = dataset_dir / "config.json"
    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset/<group>/config.json from downloaded CSV files.")
    parser.add_argument(
        "--task",
        choices=["clf", "reg", "all"],
        default="all",
        help="Which dataset group config to generate.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Path to dataset root folder (default: dataset).",
    )
    parser.add_argument(
        "--explicit-nunique-threshold",
        type=int,
        default=50,
        help="Columns with unique values <= threshold are treated as categorical (default: 50).",
    )
    args = parser.parse_args()

    tasks = ["clf", "reg"] if args.task == "all" else [args.task]

    for task in tasks:
        output_path = generate_config_json(
            task=task,
            dataset_root=args.dataset_root,
            explicit_nunique_threshold=args.explicit_nunique_threshold,
        )
        print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
