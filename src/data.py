import json
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Zero-pad a grayscale image to a centered square (X-ray background is dark)."""
    h, w = img.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(
        img, top, side - h - top, left, side - w - left,
        borderType=cv2.BORDER_CONSTANT, value=0,
    )

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (uint8 grayscale)."""
    return _CLAHE.apply(img)

def compute_stats(train_df: pd.DataFrame, max_images: int | None = None) -> dict:
    """Compute pixel mean/std (on [0,1]) over the training cache + target stats.

    Saved to ``data/stats.json`` and used for both image normalization and
    target (bone-age) standardization.
    """
    ids = train_df["id"].tolist()
    if max_images is not None:
        ids = ids[:max_images]

    psum = psum_sq = count = 0.0
    for img_id in tqdm(ids, desc="pixel stats"):
        p = Path(f"../data/cache/train/{img_id}.png")
        if not p.exists():  continue
        arr = cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
        psum += arr.sum()
        psum_sq += (arr ** 2).sum()
        count += arr.size

    mean = psum / count
    std = float(np.sqrt(max(psum_sq / count - mean ** 2, 1e-12)))

    stats = {
        "pixel_mean": float(mean),
        "pixel_std": float(std),
        "boneage_mean": float(train_df["boneage"].mean()),
        "boneage_std": float(train_df["boneage"].std()),
        "n_images": int(count // (256 * 256)),
    }
    Path("../data").mkdir(parents=True, exist_ok=True)
    Path("../data/stats.json").write_text(json.dumps(stats, indent=2))
    print("Saved stats:", stats)
    return stats

@dataclass
class TargetScaler:
    """Standardize the regression target (bone age, months) and invert back.

    Loss is computed in standardized space (stable training); metrics are
    reported in months via ``inverse``.
    """
    mean: float
    std: float

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    @classmethod
    def from_stats(cls, stats: dict) -> "TargetScaler":
        return cls(mean=stats["boneage_mean"], std=stats["boneage_std"])

class BoneAgeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache_dir: Path, transform: A.Compose, target_scaler: TargetScaler):
        self.df = df.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        self.scaler = target_scaler

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = cv2.imread(self.cache_dir / f"{row['id']}.png", cv2.IMREAD_GRAYSCALE)[..., None]  # H,W,1
        image = self.transform(image=img)["image"]  # 1,H,W float
        gender = torch.tensor([float(row["male"])], dtype=torch.float32)
        target = self.scaler.transform(float(row["boneage"]))
        return image, gender, torch.tensor([target], dtype=torch.float32)


def make_loaders():

    stats = json.loads(Path("../data/stats.json").read_text()) # required for normalization/scaling

    dfs = {}
    paths = [("train", "../data/raw/train_y.csv"),("valid", "../data/raw/valid_y.csv"),
             ("test",  "../data/raw/test_y.csv")]
    for split, path in paths:
        d = pd.read_csv(path)
        d["id"] = d["id"].astype(str)
        dfs[split] = d

    scaler = TargetScaler.from_stats(stats)
    mean, std = stats["pixel_mean"], stats["pixel_std"]

    tf_train = A.Compose([
        A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.7),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=(mean,), std=(std,), max_pixel_value=255.0),
        ToTensorV2(),
    ])

    tf_eval = A.Compose([
        A.Normalize(mean=(mean,), std=(std,), max_pixel_value=255.0),
        ToTensorV2(),
    ])

    datasets = {
        "train": BoneAgeDataset(dfs["train"],  Path("../data/cache/train"), tf_train, scaler),
        "valid": BoneAgeDataset(dfs["valid"], Path("../data/cache/valid"), tf_eval, scaler),
        "test": BoneAgeDataset(dfs["test"], Path("../data/cache/test"), tf_eval, scaler),
    }
    loaders = {
        split: DataLoader(
            ds, batch_size=32, shuffle=(split == "train"), num_workers=4, pin_memory=False,
            drop_last=(split == "train")
        )
        for split, ds in datasets.items()
    }
    return loaders, scaler
