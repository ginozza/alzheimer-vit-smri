"""DataLoader factories and custom collate utilities for MRI batches."""

from typing import Any, Dict, List
import torch
from torch.utils.data import DataLoader, Dataset


def mri_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a list of MRISample dictionaries into batched tensors.

    Validates:
    - Stacked images tensor has shape (B, 3, 224, 224) float32.
    - Stacked labels tensor has shape (B,) int64 with values in {0, 1, 2}.
    - Preserves metadata lists for Subject_ID, Session_ID, and Cohort_Source.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            f"Collated images shape mismatch. Expected (B, 3, H, W), got {tuple(images.shape)}"
        )
    if images.dtype != torch.float32:
        raise ValueError(f"Collated images must have dtype float32, got {images.dtype}")
    if labels.ndim != 1 or labels.dtype != torch.int64:
        raise ValueError(f"Collated labels must have shape (B,) and dtype int64, got {labels.shape}")

    return {
        "images": images,
        "labels": labels,
        "subject_ids": [item["subject_id"] for item in batch],
        "session_ids": [item["session_id"] for item in batch],
        "cohort_sources": [item["cohort_source"] for item in batch],
        "diagnosis_names": [item["diagnosis_name"] for item in batch],
    }


def create_mri_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Build a DataLoader configured with safety checks for MRI ViT processing.

    Args:
        dataset: PyTorch Dataset returning dictionaries with 'image' and 'label'.
        batch_size: Number of slices per training/evaluation step.
        shuffle: Whether to randomize sample order at each epoch.
        num_workers: Subprocesses for data loading (default 0 for cross-platform stability).
        drop_last: Whether to drop the trailing incomplete batch.
        pin_memory: Whether to pin memory for GPU transfer.
        generator: Optional PyTorch random generator for reproducible shuffling.

    Returns:
        Configured DataLoader instance.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=mri_collate_fn,
        drop_last=drop_last,
        pin_memory=pin_memory,
        generator=generator,
    )
