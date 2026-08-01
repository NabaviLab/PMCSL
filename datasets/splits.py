from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def create_patient_level_folds(metadata_csv, output_csv, n_splits=5, seed=42):
    """Create stratified patient-level cross-validation folds."""
    frame = pd.read_csv(metadata_csv)
    required = {"slide_id", "label", "patient_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    frame["fold"] = -1
    for fold, (_, val_idx) in enumerate(
        splitter.split(frame, y=frame["label"], groups=frame["patient_id"])
    ):
        frame.loc[val_idx, "fold"] = fold

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_patient_level_folds(
        args.metadata_csv,
        args.output_csv,
        args.n_splits,
        args.seed,
    )


if __name__ == "__main__":
    main()
