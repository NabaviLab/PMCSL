from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import openslide
except ImportError:
    openslide = None


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    target_mpp: float


class MultiScalePatchExtractor:
    """Extract tissue patches at 5x, 10x, and 20x magnifications."""

    def __init__(self, patch_size=224, stride=224, min_tissue_fraction=0.60, jpeg_quality=95):
        self.patch_size = patch_size
        self.stride = stride
        self.min_tissue_fraction = min_tissue_fraction
        self.jpeg_quality = jpeg_quality

    def _resolve_base_mpp(self, slide, base_mpp):
        if base_mpp is not None:
            return float(base_mpp)
        value = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        if value is None:
            raise ValueError("Missing MPP metadata; provide base_mpp.")
        return float(value)

    def _best_level(self, slide, desired_downsample):
        downsamples = np.asarray(slide.level_downsamples, dtype=np.float32)
        return int(np.argmin(np.abs(downsamples - desired_downsample)))

    def extract(self, wsi_path, tissue_mask, output_root, scales, base_mpp=None):
        if openslide is None:
            raise ImportError("Install openslide-python.")

        wsi_path = Path(wsi_path)
        slide = openslide.OpenSlide(str(wsi_path))
        base_mpp = self._resolve_base_mpp(slide, base_mpp)
        output_root = Path(output_root)
        summary = {}

        for scale in scales:
            desired_downsample = scale.target_mpp / base_mpp
            level = self._best_level(slide, desired_downsample)
            level_w, level_h = slide.level_dimensions[level]
            resized_mask = cv2.resize(
                tissue_mask, (level_w, level_h), interpolation=cv2.INTER_NEAREST
            )

            scale_dir = output_root / wsi_path.stem / scale.name
            scale_dir.mkdir(parents=True, exist_ok=True)
            count = 0

            for y in range(0, max(level_h - self.patch_size + 1, 1), self.stride):
                for x in range(0, max(level_w - self.patch_size + 1, 1), self.stride):
                    region = resized_mask[y:y+self.patch_size, x:x+self.patch_size]
                    if float((region > 0).mean()) < self.min_tissue_fraction:
                        continue

                    downsample = slide.level_downsamples[level]
                    location = (int(x * downsample), int(y * downsample))
                    patch = slide.read_region(
                        location, level, (self.patch_size, self.patch_size)
                    ).convert("RGB")
                    name = f"{wsi_path.stem}_{scale.name}_x{x}_y{y}.jpg"
                    patch.save(scale_dir / name, quality=self.jpeg_quality, subsampling=0)
                    count += 1

            summary[scale.name] = count

        slide.close()
        return summary
