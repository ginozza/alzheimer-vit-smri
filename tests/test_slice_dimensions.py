"""
Unit tests for 2.5D slice representation and ViT-B/16 dimension compatibility.
Aligns with baseline requirement: "Entrada en formato 2.5D a partir de tres cortes
axiales contiguos, organizados como canales de una imagen [3, 224, 224]".
"""

import numpy as np
import pytest
from src.data.slice_representation import CanonicalTripletExtractor, Slice25DConfig


def test_synthetic_batch_dimensions():
    extractor = CanonicalTripletExtractor()
    batch_size = 8
    inputs, labels = extractor.generate_synthetic_batch(batch_size=batch_size)

    # ViT-B/16 input tensor: (B, C, H, W) = (8, 3, 224, 224)
    assert inputs.shape == (batch_size, 3, 224, 224)
    assert labels.shape == (batch_size,)
    assert set(np.unique(labels)).issubset({0, 1, 2})


def test_triplet_extraction_from_volume():
    config = Slice25DConfig(canonical_z_index=50, slice_step=1, target_height=224, target_width=224)
    extractor = CanonicalTripletExtractor(config=config)

    # Simulated MNI152 volume of shape (182, 218, 182)
    fake_mni_volume = np.random.uniform(0, 1000, size=(182, 218, 182)).astype(np.float32)

    triplet = extractor.extract_triplet_from_volume(fake_mni_volume)

    assert triplet.shape == (3, 224, 224)
    assert not np.isnan(triplet).any()
    assert not np.isinf(triplet).any()


def test_triplet_out_of_bounds_raises():
    config = Slice25DConfig(canonical_z_index=200)  # Beyond volume depth
    extractor = CanonicalTripletExtractor(config=config)
    fake_volume = np.zeros((100, 100, 50), dtype=np.float32)

    with pytest.raises(IndexError):
        extractor.extract_triplet_from_volume(fake_volume)
