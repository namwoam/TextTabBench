import os, sys
import argparse
from typing import Union
from copy import deepcopy


current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import get_dataset_names, get_dataset_config
from src.dataloader_functions.download_data import download_raw_data
from src.dataloader_functions.notebook_common import (
    get_download_path,
    load_dataset_frames,
    run_basic_cleaning,
    run_upstream_dataset_cleaning,
)
from src.dataloader_functions.utils.log_msgs import info_msg, warn_msg, error_msg


def _save_processed_frames(dataset_config: dict, download_path: str, dataset_files_cleaned: list) -> None:
    """Persist cleaned outputs as refreshed CSV and Pickle files."""
    rename_files = dataset_config.get("rename_files") or dataset_config.get("files") or []

    if len(rename_files) != len(dataset_files_cleaned):
        warn_msg(
            f"Expected {len(rename_files)} output files but got {len(dataset_files_cleaned)} dataframes. "
            "Skipping processed save."
        )
        return

    for file_name, df_file in zip(rename_files, dataset_files_cleaned):
        csv_output_path = os.path.join(download_path, file_name)
        df_file.to_csv(csv_output_path, index=False)
        info_msg(f"Saved cleaned CSV file: {csv_output_path}", color="green")

        file_base = os.path.splitext(file_name)[0]
        output_filename = f"{file_base}_processed.pkl"
        output_path = os.path.join(download_path, output_filename)
        df_file.to_pickle(output_path)
        info_msg(f"Saved processed file: {output_path}", color="green")


def _flat_output_name(dataset_name: str) -> str:
    return f"{dataset_name}.csv"


def _to_flat_dataset_config(dataset_config: dict) -> dict:
    """Map any dataset config to a flat single-file output naming scheme."""
    cfg = deepcopy(dataset_config)
    source_files = cfg.get("files") or []
    if len(source_files) != 1:
        raise ValueError(
            f"Dataset '{cfg.get('dataset_name')}' has {len(source_files)} source files. "
            "Flat path mode requires exactly 1 source file."
        )
    cfg["rename_files"] = [_flat_output_name(cfg["dataset_name"])]
    return cfg

def download_datasets(
    datasets_selection: Union[str, list] = 'default',
    task: str = 'all',
    force_download: bool = False,
) -> bool:
    """
    Run download and default processing of selected datasets using Python only.

    Input:
        datasets_selection (list or str): List of dataset names or selection criteria.
        - allowed list items for bulk selection: 'default'(default), 'extra', 'other'
        - also accepts specific dataset names as strings, e.g. 'hs_cards'
        task (str): Task type, either 'clf' for classification or 'reg' for regression.
        - options: 'clf', 'reg', 'all'(default)
        force_download (bool): Force redownload even when files already exist.

    Returns:
        bool: True if successful, False otherwise.

    Example:
        download_datasets(['default', 'okcupid'], task='clf')
    """

    dataset_name_list = get_dataset_names(datasets_selection, task=task)

    print(f"Downloading datasets: {dataset_name_list}")

    for dataset_name in dataset_name_list:
        try:
            dataset_config = _to_flat_dataset_config(get_dataset_config(dataset_name))
            download_path, _ = get_download_path(dataset_config=dataset_config, start_dir=project_root)

            info_msg(f"Downloading dataset '{dataset_name}' to '{download_path}'...")
            download_result = download_raw_data(
                dataset_config=dataset_config,
                download_path=download_path,
                force_download=force_download,
                remove_unlisted=False,
            )
            if not download_result:
                error_msg(f"Download failed for dataset '{dataset_name}'.")
                return False

            dataset_files_df = load_dataset_frames(dataset_config=dataset_config, download_path=download_path)
            dataset_files_cleaned = run_basic_cleaning(
                dataset_files_df=dataset_files_df,
                target_column=dataset_config["target"],
                missing_ratio_threshold=0.5,
            )
            dataset_files_cleaned = run_upstream_dataset_cleaning(
                dataset_name=dataset_name,
                dataset_files_df=dataset_files_cleaned,
                target_column=dataset_config.get("target"),
            )
            _save_processed_frames(dataset_config, download_path, dataset_files_cleaned)

            info_msg(f"Successfully processed dataset '{dataset_name}'.", color='green')
        except Exception as e:
            error_msg(f"Error processing dataset '{dataset_name}': {e}")
            return False

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess selected datasets via Python scripts.")

    parser.add_argument(
        "--selection",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Dataset selection(s). Accepts one or more of: 'default', 'extra', 'other', or specific dataset names. "
            "Example: --selection default hs_cards"
        )
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["clf", "reg", "all"],
        default="all",
        help="Task type: 'clf' for classification, 'reg' for regression, or 'all' (default)"
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force redownload even if dataset files already exist."
    )

    args = parser.parse_args()

    success = download_datasets(
        datasets_selection=args.selection,
        task=args.task,
        force_download=args.force_download,
    )
    if success:
        print("✅ All selected datasets processed successfully.")
    else:
        print("❌ Some datasets failed to process.")