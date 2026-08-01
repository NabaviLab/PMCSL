from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


class PatchDataset(Dataset):
    def __init__(self, patch_dir):
        self.paths = sorted(
            list(Path(patch_dir).glob("*.jpg"))
            + list(Path(patch_dir).glob("*.png"))
        )
        if not self.paths:
            raise FileNotFoundError(f"No patches found in {patch_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), path.name


class PatchFeatureExtractor:
    """Extract frozen ResNet-50 patch embeddings."""

    def __init__(self, device="cuda", pretrained=True):
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(device)
        self.device = torch.device(device)

    @torch.inference_mode()
    def encode(self, patch_dir, output_file, batch_size=128, num_workers=8):
        dataset = PatchDataset(patch_dir)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        features, names = [], []
        for images, batch_names in tqdm(loader, desc="Extracting features"):
            images = images.to(self.device, non_blocking=True)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda",
            ):
                output = self.model(images)
            features.append(output.cpu())
            names.extend(batch_names)

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"features": torch.cat(features), "patch_names": names},
            output_file,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    PatchFeatureExtractor(device=args.device).encode(
        args.patch_dir,
        args.output_file,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
