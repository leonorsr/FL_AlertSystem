from __future__ import annotations

import pandas as pd


# Must remain aligned with fl/fedavg-experiments/data_utils.py.
CLIENT_SPLIT_ASSIGNMENTS = {
    "KFall": {
        "train": [
            "SA06", "SA07", "SA08", "SA09", "SA10", "SA12", "SA13", "SA14", "SA15", "SA18",
            "SA19", "SA20", "SA22", "SA23", "SA24", "SA25", "SA26", "SA27", "SA28", "SA29",
            "SA30", "SA33", "SA35", "SA36", "SA37", "SA38",
        ],
        "dev": ["SA11", "SA16", "SA32"],
        "test": ["SA17", "SA21", "SA31"],
    },
    "SisFall": {
        "train": [
            "SA01", "SA02", "SA03", "SA04", "SA05", "SA06", "SA07", "SA08", "SA09", "SA10",
            "SA11", "SA13", "SA14", "SA17", "SA18", "SA19", "SA20", "SA21", "SA22", "SA23",
            "SE01", "SE02", "SE04", "SE07", "SE09", "SE11", "SE12", "SE13", "SE14", "SE15",
        ],
        "dev": ["SA15", "SE05", "SE06", "SE10"],
        "test": ["SA12", "SA16", "SE03", "SE08"],
    },
    "UpFall": {
        "train": ["S01", "S03", "S04", "S05", "S06", "S08", "S09", "S10", "S11", "S12", "S13", "S16", "S17"],
        "dev": ["S14", "S15"],
        "test": ["S02", "S07"],
    },
}


def attach_client_split(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, split_map in CLIENT_SPLIT_ASSIGNMENTS.items():
        for split, clients in split_map.items():
            rows.extend({"dataset": dataset, "client": client, "split": split} for client in clients)
    assignments = pd.DataFrame(rows)
    result = frame.merge(assignments, on=["dataset", "client"], how="left", validate="many_to_one")
    if result["split"].isna().any():
        missing = result.loc[result["split"].isna(), ["dataset", "client"]].drop_duplicates()
        raise ValueError(f"Missing client split assignments:\n{missing.to_string(index=False)}")
    return result


def training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = attach_client_split(frame)
    return result[result["split"] == "train"].drop(columns="split").reset_index(drop=True)


def split_row_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = attach_client_split(frame)["split"].value_counts()
    return {split: int(counts.get(split, 0)) for split in ["train", "dev", "test"]}
