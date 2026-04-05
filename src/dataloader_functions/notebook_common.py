from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from src.dataloader_functions.download_data import download_raw_data
from src.dataloader_functions.load_and_pp_raw_data import (
    _drop_empty_columns,
    _drop_single_value_columns,
)
from src.dataloader_functions.utils.data_2_df import read_any_to_df


def find_project_root(start_dir: str | Path | None = None) -> Path:
    """Find the project root by searching for known project markers."""
    current = Path(start_dir or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError(f"Could not find project root from: {current}")


def ensure_project_root_on_path(start_dir: str | Path | None = None) -> Path:
    """Ensure the repository root is available on sys.path for notebook imports."""
    project_root = find_project_root(start_dir=start_dir)
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def get_dataset_subfolder(dataset_config: dict) -> str:
    task = dataset_config["task"]
    if task == "clf":
        return os.path.join("classification")
    if task == "reg":
        return os.path.join("regression")
    raise ValueError(f"Unknown task: {task}")


def get_download_path(dataset_config: dict, start_dir: str | Path | None = None) -> tuple[str, str]:
    project_root = find_project_root(start_dir=start_dir)
    dataset_subfolder = get_dataset_subfolder(dataset_config)
    download_path = project_root / "dataset" / dataset_subfolder
    return str(download_path), dataset_subfolder


def download_dataset_data(
    dataset_config: dict,
    force_download: bool = False,
    remove_unlisted: bool = True,
    start_dir: str | Path | None = None,
) -> tuple[str, str]:
    """Download dataset files and return (download_path, dataset_subfolder)."""
    download_path, dataset_subfolder = get_download_path(dataset_config, start_dir=start_dir)
    result = download_raw_data(
        dataset_config=dataset_config,
        download_path=download_path,
        force_download=force_download,
        remove_unlisted=remove_unlisted,
    )
    if result is not None:
        print(f"Downloaded {dataset_config['dataset_name']} dataset to {download_path}")
    return download_path, dataset_subfolder


def load_dataset_frames(dataset_config: dict, download_path: str) -> list[pd.DataFrame]:
    """Load dataset files listed in config from download path as DataFrames."""
    file_names = dataset_config.get("rename_files") or dataset_config.get("files") or []
    frames: list[pd.DataFrame] = []
    for file_name in file_names:
        file_location = os.path.join(download_path, file_name)
        print(f"Loading {file_location}")
        df = read_any_to_df(file_location)
        if df is None:
            raise ValueError(f"Failed to load dataframe from file: {file_location}")
        frames.append(df)
    return frames


def run_basic_cleaning(
    dataset_files_df: list[pd.DataFrame],
    target_column: str,
    missing_ratio_threshold: float = 0.5,
) -> list[pd.DataFrame]:
    """Apply the generic cleaning pipeline used in notebooks."""
    cleaned_frames: list[pd.DataFrame] = []
    for df_file in dataset_files_df:
        df_size = df_file.shape
        # Normalize embedded newlines in string-like cells to keep each record single-line.
        df_file = strip_newlines_in_cells(df_file)
        df_file = _drop_empty_columns(df_file, threshold=missing_ratio_threshold)
        df_file = _drop_single_value_columns(df_file)
        df_file = df_file.drop_duplicates()
        if target_column in df_file.columns:
            df_file = df_file[df_file[target_column].notna()]
        df_file = df_file.loc[:, ~df_file.columns.str.contains("^Unnamed")]
        cleaned_frames.append(df_file)
        print(f"Dataframe shape before/afrer cleaning: {df_size} / {df_file.shape}")
    return cleaned_frames


def strip_newlines_in_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Replace CR/LF sequences inside string-like cells with a single space."""

    def _clean_value(value):
        if isinstance(value, str):
            return value.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        return value

    cleaned = df.copy()
    string_like_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in string_like_cols:
        cleaned[col] = cleaned[col].map(_clean_value)
    return cleaned


def drop_selected_columns(dataset_files_df: list[pd.DataFrame], cols_to_drop: list[str]) -> list[pd.DataFrame]:
    """Drop user-selected columns from each dataframe, if present."""
    cleaned_frames: list[pd.DataFrame] = []
    for df_file in dataset_files_df:
        df_size = df_file.shape
        for col in cols_to_drop:
            if col in df_file.columns:
                df_file = df_file.drop(col, axis=1)
            else:
                print(f"Column {col} not found in dataframe")
        cleaned_frames.append(df_file)
        print(f"Dataframe shape before/afrer by-hand cleaning: {df_size} / {df_file.shape}")
    return cleaned_frames


def is_mostly_numeric(series: pd.Series, length_threshold: float = 0.5, unique_threshold: float = 0.8) -> bool:
    """Check whether a string/object series behaves like numeric data."""
    stripped = series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
    original_len = series.astype(str).str.len().replace(0, 1)
    length_ratio = (stripped.str.len() / original_len).mean()
    unique_ratio = stripped.nunique(dropna=False) / max(series.nunique(dropna=False), 1)
    return bool(length_ratio > length_threshold and unique_ratio > unique_threshold)


def classify_columns(
    df: pd.DataFrame,
    unique_ratio_threshold: float | None = None,
    explicit_nunique_threshold: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Classify dataframe columns into numerical, categorical, and textual."""
    n_rows = len(df)
    if explicit_nunique_threshold is not None:
        nunique_threshold = explicit_nunique_threshold
    elif unique_ratio_threshold is not None:
        nunique_threshold = int(unique_ratio_threshold * n_rows)
    else:
        nunique_threshold = int(0.05 * n_rows)

    nunique_threshold = max(10, nunique_threshold)
    print(f"Threshold for categorical vs textual: {nunique_threshold}")

    numerical_cols: list[str] = []
    categorical_cols: list[str] = []
    textual_cols: list[str] = []

    for col in df.columns:
        series = df[col]
        nunique = series.nunique(dropna=False)

        if pd.api.types.is_numeric_dtype(series):
            if nunique <= nunique_threshold:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            if is_mostly_numeric(series):
                numerical_cols.append(col)
            elif nunique <= nunique_threshold:
                categorical_cols.append(col)
            else:
                textual_cols.append(col)
        else:
            print(f"Unhandled column type: '{col}' (dtype={series.dtype})")

    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Textual columns ({len(textual_cols)}): {textual_cols}")

    return numerical_cols, categorical_cols, textual_cols


def build_column_summary(
    df: pd.DataFrame,
    numerical_cols: list[str],
    categorical_cols: list[str],
    textual_cols: list[str],
) -> pd.DataFrame:
    """Build a standard summary table for column type annotations."""
    summary = []
    for col in df.columns:
        if col in categorical_cols:
            col_type = "categorical"
            num_categories = df[col].nunique(dropna=True)
            num_categories_display = int(num_categories)
        elif col in textual_cols:
            col_type = "textual"
            num_categories = df[col].nunique(dropna=True)
            num_categories_display = int(num_categories)
        elif col in numerical_cols:
            col_type = "numerical"
            num_categories = df[col].nunique(dropna=True)
            num_categories_display = f"~ {num_categories} ~"
        else:
            col_type = "unknown"
            num_categories_display = "--"

        example = df[col].dropna().iloc[0] if df[col].dropna().size > 0 else None
        summary.append(
            {
                "Column Name": col,
                "Example Value": str(example),
                "Type": col_type,
                "# Categories": num_categories_display,
            }
        )

    return pd.DataFrame(summary)