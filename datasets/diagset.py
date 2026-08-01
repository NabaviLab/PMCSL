from .base import BaseProstateWSIDataset


class DiagSetDataset(BaseProstateWSIDataset):
    """DiagSet whole-slide image metadata loader."""
    dataset_name = "DiagSet"
