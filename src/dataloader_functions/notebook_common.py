from __future__ import annotations

import os
import sys
from pathlib import Path
import re

import pandas as pd
import numpy as np

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


UPSTREAM_DROP_COLUMNS: dict[str, list[str]] = {
    "customer_complaints": ["Complaint ID", "Date sent to company", "Timely response?", "Consumer disputed?"],
    "job_frauds": ["job_id"],
    "hs_cards": ["id"],
    "kickstarter": ["usd_pledged", "id"],
    "osha_accidents": [
        "summary_nr",
        "proj_cost",
        "proj_type",
        "nature_of_inj",
        "part_of_body",
        "event_type",
        "evn_factor",
        "hum_factor",
        "task_assigned",
    ],
    "spotify": ["track_id"],
    "airbnb": [
        "id",
        "listing_url",
        "thumbnail_url",
        "host_url",
        "medium_url",
        "picture_url",
        "xl_picture_url",
        "host_id",
        "host_thumbnail_url",
        "host_picture_url",
        "name",
        "host_name",
        "host_location",
        "street",
        "host_neighbourhood",
        "host_listings_count",
        "host_total_listings_count",
        "weekly_price",
        "cleaning_fee",
        "maximum_nights",
        "first_review",
        "last_review",
    ],
    "beer": ["review_aroma", "review_appearance", "review_palate", "review_taste"],
    "calif_houses": [
        "Id",
        "Address",
        "High School",
        "Middle School",
        "Elementary School",
        "State",
        "Region",
        "Sold Price",
        "Listed Price",
        "Last Sold On",
        "Listed On",
    ],
    "laptops": ["link", "Part Number", "Model Number", "Model Name"],
    "mercari": ["train_id", "test_id"],
    "sf_permits": [
        "Permit Number",
        "Record ID",
        "Permit Expiration Date",
        "Completed Date",
        "First Construction Document Date",
        "Current Status Date",
        "Permit Creation Date",
    ],
}


def _clean_zip_code(series: pd.Series) -> pd.Series:
    """Keep the first ZIP 3-digit prefix as nullable integer."""
    cleaned = series.astype(str).str.extract(r"(\d{3})")
    return cleaned[0].astype("Int64")


def _apply_dataset_custom_transforms(
    dataset_name: str,
    dataset_files_df: list[pd.DataFrame],
    target_column: str | None,
) -> list[pd.DataFrame]:
    """Apply custom per-dataset cleaning to mirror upstream notebooks."""
    out_frames: list[pd.DataFrame] = []

    for df_file in dataset_files_df:
        df = df_file.copy()

        if dataset_name == "customer_complaints":
            if target_column in df.columns:
                to_drop = ["Closed", "In progress", "Untimely response", "Closed with relief"]
                df = df[~df[target_column].isin(to_drop)].copy()
            if "Date received" in df.columns:
                dt = pd.to_datetime(df["Date received"], format="%m/%d/%Y", errors="coerce")
                df["Date received"] = dt.apply(lambda x: x.timestamp() if pd.notnull(x) else float("nan"))
            if "ZIP code" in df.columns:
                df["ZIP code"] = _clean_zip_code(df["ZIP code"])

        elif dataset_name == "hs_cards":
            if "player_class" in df.columns:
                df = df[df["player_class"] != "DREAM"]
                df = df[df["player_class"] != "DEATHKNIGHT"]

        elif dataset_name == "kickstarter":
            if "launched_at" in df.columns:
                launched = pd.to_datetime(df["launched_at"], errors="coerce")
                df["launched_at"] = launched.astype("int64") / 1e9
            if "deadline" in df.columns:
                deadline = pd.to_datetime(df["deadline"], errors="coerce")
                df["deadline"] = deadline.astype("int64") / 1e9

        elif dataset_name == "osha_accidents":
            if "Event Date" in df.columns:
                dt = pd.to_datetime(df["Event Date"], format="%m/%d/%Y", errors="coerce")
                df["Event Date"] = dt.apply(lambda x: x.timestamp() if pd.notnull(x) else float("nan"))

        elif dataset_name == "spotify":
            if "track_genre" in df.columns:
                target_genres = [
                    "pop",
                    "rock",
                    "hip-hop",
                    "jazz",
                    "classical",
                    "metal",
                    "electronic",
                    "indie",
                    "r-n-b",
                    "country",
                ]
                df = df[df["track_genre"].isin(target_genres)].copy()

        elif dataset_name == "airbnb":
            if "host_since" in df.columns:
                dt = pd.to_datetime(df["host_since"], format="%Y-%m-%d", errors="coerce")
                df["host_since"] = dt.astype("int64") / 1e9

        elif dataset_name == "beer":
            if "number_of_reviews" in df.columns:
                df = df[df["number_of_reviews"] >= 5].copy()

        elif dataset_name == "sf_permits":
            if "Filed Date" in df.columns and "Issued Date" in df.columns and target_column:
                filed_date = pd.to_datetime(df["Filed Date"], errors="coerce")
                issued_date = pd.to_datetime(df["Issued Date"], errors="coerce")
                df[target_column] = (issued_date - filed_date).dt.days
                if "Issued Date" in df.columns:
                    df = df.drop(["Issued Date"], axis=1)
                df["Filed Date"] = filed_date.apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)

            if "Location" in df.columns:
                location_split = df["Location"].astype(str).str.split(",", expand=True)
                if location_split.shape[1] >= 2:
                    location_split[0] = (
                        location_split[0].str.replace("(", "", regex=False).apply(pd.to_numeric, errors="coerce")
                    )
                    location_split[1] = (
                        location_split[1].str.replace(")", "", regex=False).apply(pd.to_numeric, errors="coerce")
                    )
                    location_split = location_split.iloc[:, :2]
                    location_split.columns = ["Location_Latitude", "Location_Longitude"]
                    df = pd.concat([df, location_split], axis=1)
                df = df.drop(["Location"], axis=1)

            if target_column and target_column in df.columns:
                df = df[df[target_column] < 1000]
                df = df[df[target_column] >= 0]

        out_frames.append(df)

    return out_frames


def run_upstream_dataset_cleaning(
    dataset_name: str,
    dataset_files_df: list[pd.DataFrame],
    target_column: str | None,
) -> list[pd.DataFrame]:
    """Run upstream-equivalent dataset-specific cleaning pipeline."""
    cols_to_drop = UPSTREAM_DROP_COLUMNS.get(dataset_name, [])
    dataset_files_cleaned = drop_selected_columns(dataset_files_df, cols_to_drop) if cols_to_drop else dataset_files_df
    return _apply_dataset_custom_transforms(dataset_name, dataset_files_cleaned, target_column)


def is_mostly_numeric(series: pd.Series, length_threshold: float = 0.5, unique_threshold: float = 0.8) -> bool:
    """Check whether a string/object series behaves like numeric data.

    A column is considered numeric-like when either:
    1) Most characters are already numeric after removing only a small formatting shell.
    2) The non-numeric shell is repetitive across rows (e.g., "ABV 12%", "ABV 15%").
    """
    s = series.dropna().astype(str)
    if s.empty:
        return False

    stripped = s.str.replace(r"[^\d\.\-]", "", regex=True)
    original_len = s.str.len().replace(0, 1)
    length_ratio = float((stripped.str.len() / original_len).mean())

    numeric_cast = pd.to_numeric(stripped.replace("", pd.NA), errors="coerce")
    numeric_parse_ratio = float(numeric_cast.notna().mean())

    # Keep alphabetic/punctuation shell and collapse whitespace to compare wrappers.
    non_numeric_shell = s.str.replace(r"[\d\.\-]", "", regex=True)
    non_numeric_shell = non_numeric_shell.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    repetitive_shell_ratio = float(non_numeric_shell.value_counts(normalize=True, dropna=False).iloc[0])

    mostly_numeric_chars = length_ratio >= length_threshold and numeric_parse_ratio >= 0.7
    repetitive_wrapper_numeric = repetitive_shell_ratio >= 0.8 and numeric_parse_ratio >= 0.7

    return bool(mostly_numeric_chars or repetitive_wrapper_numeric)


def clean_numeric_like_value(value):
    """Convert numeric-like values (e.g., 'ABV 12%', '15s') to float when possible."""
    if pd.isna(value):
        return value

    value_str = str(value)
    value_cleaned = re.sub(r"[^0-9\.\-]", "", value_str)

    if "-" in value_cleaned:
        value_cleaned = "-" + value_cleaned.replace("-", "")

    if value_cleaned.count(".") > 1:
        parts = value_cleaned.split(".")
        value_cleaned = "".join(parts[:-1]) + "." + parts[-1]

    if value_cleaned in {"", "-", ".", "-."}:
        return pd.NA

    try:
        return float(value_cleaned)
    except ValueError:
        return pd.NA


def clean_numeric_like_columns(df: pd.DataFrame, numeric_like_columns: list[str]) -> pd.DataFrame:
    """Clean numeric artefacts from columns identified as numerical by heuristics."""
    out = df.copy()
    for col in numeric_like_columns:
        if col not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        out[col] = out[col].map(clean_numeric_like_value)
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


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
        # Default heuristic: 5% for smaller datasets, capped at 50 for larger ones.
        nunique_threshold = min(50, int(0.05 * n_rows))

    nunique_threshold = max(2, nunique_threshold)
    print(f"Threshold for categorical vs textual: {nunique_threshold}")

    numerical_cols: list[str] = []
    categorical_cols: list[str] = []
    textual_cols: list[str] = []

    for col in df.columns:
        series = df[col]
        nunique = series.nunique(dropna=False)

        if pd.api.types.is_numeric_dtype(series):
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