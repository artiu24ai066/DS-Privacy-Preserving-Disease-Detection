"""
data_loader.py
--------------
Handles all dataset loading, preprocessing, train/val splitting,
and client partitioning for federated learning simulation.

Dataset: Chest X-Ray Images (Pneumonia) — Kermany et al. 2018
Classes: NORMAL (0), PNEUMONIA (1)
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


# ── ImageNet normalization constants ──────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]   # index 0 = NORMAL, index 1 = PNEUMONIA


# ── Transforms ────────────────────────────────────────────────────────────────
def get_train_transform():
    """Light augmentation for training images."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform():
    """Deterministic transform for validation/test images."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Dataset class ─────────────────────────────────────────────────────────────
class ChestXRayDataset(Dataset):
    """
    Loads chest X-ray images from a directory that contains
    NORMAL/ and PNEUMONIA/ sub-folders.
    """

    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir : path to folder containing NORMAL/ and PNEUMONIA/
            transform: torchvision transform to apply
        """
        self.transform = transform
        self.samples: list = []  # list of (file_path: str, label: int)

        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = Path(root_dir) / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Expected folder not found: {class_dir}")
            for img_path in sorted(class_dir.glob("*.jpeg")):
                self.samples.append((str(img_path), label))
            # also pick up .jpg and .png just in case
            for ext in ("*.jpg", "*.png"):
                for img_path in sorted(class_dir.glob(ext)):
                    if str(img_path) not in [s[0] for s in self.samples]:
                        self.samples.append((str(img_path), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_labels(self) -> list:
        """Return all labels (used for stratified splitting)."""
        return [s[1] for s in self.samples]


# ── Train / Validation splitter ───────────────────────────────────────────────
def make_train_val_split(
    train_dir: str,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> tuple:
    """
    Creates a stratified train/val split from the training directory.
    The official val/ folder has only 16 images, so we build our own.

    Returns:
        train_indices, val_indices — index lists into the full train dataset
    """
    full_ds = ChestXRayDataset(train_dir, transform=None)
    labels  = full_ds.get_labels()

    rng = random.Random(seed)

    # Separate indices by class
    normal_idx    = [i for i, l in enumerate(labels) if l == 0]
    pneumonia_idx = [i for i, l in enumerate(labels) if l == 1]

    rng.shuffle(normal_idx)
    rng.shuffle(pneumonia_idx)

    n_val_normal    = int(len(normal_idx)    * val_fraction)
    n_val_pneumonia = int(len(pneumonia_idx) * val_fraction)

    val_idx   = normal_idx[:n_val_normal]    + pneumonia_idx[:n_val_pneumonia]
    train_idx = normal_idx[n_val_normal:]    + pneumonia_idx[n_val_pneumonia:]

    rng.shuffle(val_idx)
    rng.shuffle(train_idx)

    return train_idx, val_idx


# ── Federated client partitioner ──────────────────────────────────────────────
def partition_for_clients(
    indices: list,
    labels: list,
    n_clients: int = 3,
    seed: int = 42,
) -> list:
    """
    Splits training indices into n_clients approximately equal,
    class-balanced partitions (simulating separate hospitals).

    Returns:
        List of n_clients lists of indices.
    """
    rng = random.Random(seed)

    normal_idx    = [i for i in indices if labels[i] == 0]
    pneumonia_idx = [i for i in indices if labels[i] == 1]

    rng.shuffle(normal_idx)
    rng.shuffle(pneumonia_idx)

    client_indices = [[] for _ in range(n_clients)]

    for c_idx, chunk in enumerate(np.array_split(normal_idx, n_clients)):
        client_indices[c_idx].extend(chunk.tolist())
    for c_idx, chunk in enumerate(np.array_split(pneumonia_idx, n_clients)):
        client_indices[c_idx].extend(chunk.tolist())

    for c in client_indices:
        rng.shuffle(c)

    return client_indices


# ── DataLoader factory ────────────────────────────────────────────────────────
def get_loaders(
    train_dir: str,
    test_dir:  str,
    batch_size: int = 32,
    val_fraction: float = 0.20,
    seed: int = 42,
    num_workers: int = 0,
):
    """
    Returns:
        train_loader, val_loader, test_loader, full_train_dataset
    """
    train_ds_full = ChestXRayDataset(train_dir, transform=get_train_transform())
    val_transform_ds = ChestXRayDataset(train_dir, transform=get_eval_transform())

    train_idx, val_idx = make_train_val_split(train_dir, val_fraction, seed)

    train_subset = Subset(train_ds_full,       train_idx)
    val_subset   = Subset(val_transform_ds,    val_idx)

    test_ds = ChestXRayDataset(test_dir, transform=get_eval_transform())

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,      batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds_full


# ── Dataset summary ───────────────────────────────────────────────────────────
def dataset_summary(train_dir: str, test_dir: str, val_fraction: float = 0.20):
    """Prints and returns a summary DataFrame of image counts per split/class."""
    train_ds = ChestXRayDataset(train_dir, transform=None)
    test_ds  = ChestXRayDataset(test_dir,  transform=None)

    train_idx, val_idx = make_train_val_split(train_dir, val_fraction)
    labels = train_ds.get_labels()

    def count(indices, label):
        return sum(1 for i in indices if labels[i] == label)

    rows = [
        {"Split": "Train",      "NORMAL": count(train_idx, 0),
                                "PNEUMONIA": count(train_idx, 1),
                                "Total": len(train_idx)},
        {"Split": "Validation", "NORMAL": count(val_idx,   0),
                                "PNEUMONIA": count(val_idx,   1),
                                "Total": len(val_idx)},
        {"Split": "Test",       "NORMAL": sum(1 for _,l in test_ds.samples if l==0),
                                "PNEUMONIA": sum(1 for _,l in test_ds.samples if l==1),
                                "Total": len(test_ds)},
    ]
    df = pd.DataFrame(rows)
    print("\n=== Dataset Summary ===")
    print(df.to_string(index=False))
    return df
