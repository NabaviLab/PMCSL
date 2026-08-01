from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import yaml
import openslide

from preprocessing.patch_extraction import MultiScalePatchExtractor, ScaleSpec
from preprocessing.tissue_segmentation import TissueSegmenter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--base-mpp", type=float, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    slide = openslide.OpenSlide(args.wsi)
    size = int(cfg["preprocessing"]["thumbnail_size"])
    thumb = slide.get_thumbnail((size, size)).convert("RGB")
    slide.close()

    segmenter = TissueSegmenter(
        saturation_threshold=int(cfg["preprocessing"]["saturation_threshold"]),
        min_component_area=int(cfg["preprocessing"]["min_component_area"]),
        kernel_size=int(cfg["preprocessing"]["morphology_kernel_size"]),
    )
    result = segmenter(np.asarray(thumb))

    mask_dir = Path(args.output_root) / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"{Path(args.wsi).stem}_mask.png"
    Image.fromarray(result.mask).save(mask_path)

    scales = [
        ScaleSpec(item["name"], float(item["target_mpp"]))
        for item in cfg["data"]["magnifications"]
    ]
    extractor = MultiScalePatchExtractor(
        patch_size=int(cfg["data"]["patch_size"]),
        stride=int(cfg["data"]["stride"]),
        min_tissue_fraction=float(cfg["data"]["min_tissue_fraction"]),
        jpeg_quality=int(cfg["preprocessing"]["jpeg_quality"]),
    )
    counts = extractor.extract(
        args.wsi,
        result.mask,
        Path(args.output_root) / "patches",
        scales,
        base_mpp=args.base_mpp,
    )
    print({"mask": str(mask_path), "patch_counts": counts})


if __name__ == "__main__":
    main()
