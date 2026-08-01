from .base import BaseProstateWSIDataset


class Gleason19Dataset(BaseProstateWSIDataset):
    """GLEASON19 whole-slide image metadata loader."""
    dataset_name = "GLEASON19"
