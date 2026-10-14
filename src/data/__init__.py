"""
Data management, inventory parsing, subject-level partitioning, and 2.5D representations.
"""

from .dataset import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, MRI25DDataset, MRISample, SyntheticMRIDataset
from .dataloader import create_mri_dataloader, mri_collate_fn
from .inventory_loader import DatasetInventoryLoader, SubjectRecord
from .subject_split import SubjectSplitter, DataSplitResult
from .slice_representation import CanonicalTripletExtractor, Slice25DConfig

__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "CanonicalTripletExtractor",
    "DataSplitResult",
    "DatasetInventoryLoader",
    "MRI25DDataset",
    "MRISample",
    "Slice25DConfig",
    "SubjectRecord",
    "SubjectSplitter",
    "SyntheticMRIDataset",
    "create_mri_dataloader",
    "mri_collate_fn",
]
