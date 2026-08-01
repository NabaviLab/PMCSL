from .base import BaseProstateWSIDataset


class TCGAPRADataset(BaseProstateWSIDataset):
    """TCGA-PRAD whole-slide image metadata loader."""
    dataset_name = "TCGA-PRAD"
