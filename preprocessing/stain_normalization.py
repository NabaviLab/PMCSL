import cv2
import numpy as np


class ReinhardNormalizer:
    """Reinhard stain normalization using a fitted reference image."""

    def __init__(self):
        self.target_mean = None
        self.target_std = None

    def fit(self, rgb_reference):
        lab = cv2.cvtColor(rgb_reference, cv2.COLOR_RGB2LAB).astype(np.float32)
        pixels = lab.reshape(-1, 3)
        self.target_mean = pixels.mean(axis=0)
        self.target_std = pixels.std(axis=0) + 1e-6

    def transform(self, rgb_image):
        if self.target_mean is None:
            raise RuntimeError("Call fit() using a reference stain image first.")

        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB).astype(np.float32)
        source = lab.reshape(-1, 3)
        mean = source.mean(axis=0)
        std = source.std(axis=0) + 1e-6

        normalized = (lab - mean) / std
        normalized = normalized * self.target_std + self.target_mean
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        return cv2.cvtColor(normalized, cv2.COLOR_LAB2RGB)
