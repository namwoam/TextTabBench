from __future__ import annotations

from copy import deepcopy

from configs.dataset_configs import all_configs, get_dataset_list as _get_dataset_list


# Explicit source-of-truth entry requested for Python-only dataset orchestration.
IT_SALARY_DATASET_CONFIG = {
    "dataset_name": "it_salary",
    "source": "kaggle",  # ['kaggle', 'local', 'openml', 'hf']
    "remote_path": "parulpandey/2020-it-salary-survey-for-eu-region",
    "files": ["IT%20Salary%20Survey%20EU%20%202020.csv"],
    "rename_files": ["IT_salary_eu_data.csv"],
    "task": "reg",  # ['reg', 'clf']
    "target": "Yearly brutto salary (without bonus and stocks) in EUR",
}


DATASET_CONFIGS = deepcopy(all_configs)
DATASET_CONFIGS["it_salary"] = {
    **DATASET_CONFIGS.get("it_salary", {}),
    **IT_SALARY_DATASET_CONFIG,
}


def get_dataset_config(dataset_name: str) -> dict:
    """Return a defensive copy of one dataset config."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset_name}' not found in configurations.")
    return deepcopy(DATASET_CONFIGS[dataset_name])


def get_dataset_names(datasets_selection: str | list, task: str = "all") -> list[str]:
    """Resolve dataset selections using the existing selection semantics."""
    return _get_dataset_list(datasets_selection, task=task)
