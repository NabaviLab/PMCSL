from .diagset import DiagSetDataset
from .gleason19 import Gleason19Dataset
from .tcga_prad import TCGAPRADataset

_DATASETS = {
    "tcga_prad": TCGAPRADataset,
    "gleason19": Gleason19Dataset,
    "diagset": DiagSetDataset,
}


def build_dataset(name, metadata_csv, wsi_root, *, validate_paths=True):
    """Build a supported prostate WSI dataset by name."""
    key = name.lower()
    if key not in _DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(_DATASETS)}")
    return _DATASETS[key](
        metadata_csv=metadata_csv,
        wsi_root=wsi_root,
        validate_paths=validate_paths,
    )
