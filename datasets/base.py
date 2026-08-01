from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SlideRecord:
    """Metadata associated with one whole-slide image."""

    slide_id: str
    wsi_path: Path
    label: int
    patient_id: str | None = None
    center: str | None = None


class BaseProstateWSIDataset(Dataset):
    """
    Base metadata dataset.

    Required CSV columns:
        slide_id, relative_path, label

    Optional:
        patient_id, center
    """

    required_columns = {"slide_id", "relative_path", "label"}

    def __init__(
        self,
        metadata_csv: str | Path,
        wsi_root: str | Path,
        *,
        validate_paths: bool = True,
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.wsi_root = Path(wsi_root)
        self.records = self._load_records(validate_paths)

    def _load_records(self, validate_paths: bool) -> List[SlideRecord]:
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_csv}")

        frame = pd.read_csv(self.metadata_csv)
        missing = self.required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        records: List[SlideRecord] = []
        for row in frame.to_dict(orient="records"):
            path = self.wsi_root / str(row["relative_path"])
            if validate_paths and not path.exists():
                raise FileNotFoundError(f"WSI not found: {path}")

            label = int(row["label"])
            if label not in {0, 1, 2, 3}:
                raise ValueError(f"Expected label in {{0,1,2,3}}, got {label}")

            records.append(
                SlideRecord(
                    slide_id=str(row["slide_id"]),
                    wsi_path=path,
                    label=label,
                    patient_id=None if pd.isna(row.get("patient_id")) else str(row.get("patient_id")),
                    center=None if pd.isna(row.get("center")) else str(row.get("center")),
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]
        return {
            "slide_id": record.slide_id,
            "wsi_path": str(record.wsi_path),
            "label": torch.tensor(record.label, dtype=torch.long),
            "patient_id": record.patient_id,
            "center": record.center,
        }
