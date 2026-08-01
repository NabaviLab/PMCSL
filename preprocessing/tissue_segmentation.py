from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TissueSegmentationResult:
    mask: np.ndarray
    contours: list[np.ndarray]


class TissueSegmenter:
    """CLAM-style low-resolution tissue mask generation."""

    def __init__(self, saturation_threshold=20, min_component_area=500, kernel_size=7):
        self.saturation_threshold = saturation_threshold
        self.min_component_area = min_component_area
        self.kernel_size = kernel_size

    def __call__(self, rgb_thumbnail):
        if rgb_thumbnail.ndim != 3 or rgb_thumbnail.shape[-1] != 3:
            raise ValueError("Expected RGB image [H,W,3].")

        hsv = cv2.cvtColor(rgb_thumbnail, cv2.COLOR_RGB2HSV)
        mask = (hsv[..., 1] >= self.saturation_threshold).astype(np.uint8) * 255
        kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        cleaned = np.zeros_like(mask)
        for idx in range(1, count):
            if stats[idx, cv2.CC_STAT_AREA] >= self.min_component_area:
                cleaned[labels == idx] = 255

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return TissueSegmentationResult(mask=cleaned, contours=contours)
