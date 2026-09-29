"""Scientific invariants for E2 geometry and subject-local intensity handling."""

import nibabel as nib
import numpy as np
import pytest

from scripts.run_e2_demo import save_nifti, synthetic_volume
from src.data.preprocessing import PreprocessingConfig, QCError, generate_input, load_volume
from src.data.slice_representation import CanonicalTripletExtractor, Slice25DConfig


@pytest.fixture
def image_data():
    affine = np.eye(4)
    affine[:3, 3] = [-8, -9, -20]
    return synthetic_volume(2, (16, 18, 20)), affine


def test_background_preserved_and_statistics_use_only_signal():
    image = np.zeros((10, 10), dtype=np.float32)
    image[2:8, 2:8] = np.arange(1, 37).reshape(6, 6)
    output, stats = CanonicalTripletExtractor().normalize_with_metadata(image)
    values = np.clip(image[image > 0], *np.percentile(image[image > 0], [1, 99]))
    assert stats["mean"] == pytest.approx(values.mean())
    assert np.all(output[image == 0] == 0)
    assert output[image > 0].mean() == pytest.approx(0, abs=1e-7)
    assert output[image > 0].std() == pytest.approx(1)
    bigger = np.pad(image, 20)
    assert np.array_equal(CanonicalTripletExtractor().normalize_slice(bigger)[20:30, 20:30], output)


@pytest.mark.parametrize("value", [0.0, 1.0, np.nan, np.inf, -1.0])
def test_invalid_slice_signal_rejected(value):
    with pytest.raises(ValueError):
        CanonicalTripletExtractor().normalize_slice(np.full((10, 10), value))


@pytest.mark.parametrize("kwargs", [{"normalization": "typo"}, {"channels": 4}, {"slice_step": 2},
                                    {"clip_percentiles": (99, 1)}, {"target_height": 0}])
def test_invalid_extractor_configuration(kwargs):
    with pytest.raises(ValueError):
        Slice25DConfig(**kwargs)


def test_min_max_mode():
    array = np.arange(100, dtype=np.float32).reshape(10, 10)
    output = CanonicalTripletExtractor(Slice25DConfig(normalization="min_max")).normalize_slice(array)
    assert output.min() == 0 and output.max() == 1


def test_affine_locates_physical_triplet_and_orientation_is_invariant(tmp_path, image_data):
    volume, affine = image_data
    original = tmp_path / "original.nii"
    reversed_path = tmp_path / "reversed.nii"
    save_nifti(original, volume, affine)
    flipped = affine.copy()
    flipped[0, 0] = -1
    flipped[0, 3] += volume.shape[0] - 1
    save_nifti(reversed_path, volume[::-1], flipped)
    config = PreprocessingConfig()
    data, geometry = load_volume(original, config)
    reversed_data, reversed_geometry = load_volume(reversed_path, config)
    result, metadata = generate_input(data, geometry, config)
    reverse_result, _ = generate_input(reversed_data, reversed_geometry, config)
    assert metadata["slice_indices"] == [7, 8, 9]
    assert metadata["slice_z_mm"] == [-13, -12, -11]
    assert geometry["content_sha256"] == reversed_geometry["content_sha256"]
    assert np.array_equal(result, reverse_result)
    assert result.dtype == np.float32 and result.shape == (3, 224, 224)


@pytest.mark.parametrize("defect,reason", [("4d", "dimensions"), ("nan", "nonfinite"),
    ("inf", "nonfinite"), ("negative", "negative_intensity"), ("empty", "foreground"),
    ("spacing", "spacing"), ("units", "units"), ("conflict", "affine_conflict"),
    ("oblique", "oblique"), ("unknown", "affine")])
def test_volume_qc_rejections(tmp_path, image_data, defect, reason):
    volume, affine = image_data
    if defect == "4d":
        volume = volume[..., None]
    elif defect in {"nan", "inf", "negative"}:
        volume[0, 0, 0] = {"nan": np.nan, "inf": np.inf, "negative": -1}[defect]
    elif defect == "empty":
        volume[:] = 0
    elif defect == "spacing":
        affine[0, 0] = 2
    elif defect == "oblique":
        theta = 0.1
        affine[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    path = tmp_path / "sample.nii"
    save_nifti(path, volume, affine)
    loaded = nib.load(path)
    image = nib.Nifti1Image(loaded.get_fdata(dtype=np.float32).copy(), loaded.affine, loaded.header)
    if defect == "units":
        image.header.set_xyzt_units("unknown")
    elif defect == "conflict":
        changed = affine.copy()
        changed[2, 3] += 5
        image.set_qform(changed, code=4)
    elif defect == "unknown":
        image.set_qform(None, code=0)
        image.set_sform(None, code=0)
    nib.save(image, path)
    with pytest.raises(QCError) as error:
        load_volume(path, PreprocessingConfig())
    assert error.value.reason == reason


def test_bounds_and_resource_limit(tmp_path, image_data):
    path = tmp_path / "sample.nii"
    save_nifti(path, *image_data)
    with pytest.raises(QCError, match="voxel limit"):
        load_volume(path, PreprocessingConfig(max_voxels=10))
    data, geometry = load_volume(path, PreprocessingConfig())
    with pytest.raises(QCError, match="outside volume"):
        generate_input(data, geometry, PreprocessingConfig(canonical_z_mm=100))


def test_crop_cannot_silently_remove_signal():
    volume = np.ones((226, 10, 5), dtype=np.float32)
    affine = np.eye(4)
    affine[2, 3] = -14
    with pytest.raises(QCError) as error:
        generate_input(volume, {"canonical_affine": affine.tolist()}, PreprocessingConfig())
    assert error.value.reason == "foreground_crop"


def test_seeded_synthetic_generator():
    generator = CanonicalTripletExtractor()
    first = generator.generate_synthetic_batch(seed=12)
    second = generator.generate_synthetic_batch(seed=12)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_axis_permutation_is_canonicalized(tmp_path, image_data):
    volume, affine = image_data
    original = tmp_path / "original.nii"
    permuted = tmp_path / "permuted.nii"
    save_nifti(original, volume, affine)
    permuted_affine = affine.copy()
    permuted_affine[:3, :3] = affine[:3, [2, 0, 1]]
    save_nifti(permuted, np.transpose(volume, (2, 0, 1)), permuted_affine)
    data, geometry = load_volume(original, PreprocessingConfig())
    other, other_geometry = load_volume(permuted, PreprocessingConfig())
    assert np.array_equal(data, other)
    assert geometry["canonical_affine"] == other_geometry["canonical_affine"]


def test_singular_affine_rejected(tmp_path, image_data):
    volume, affine = image_data
    image = nib.Nifti1Image(volume, affine)
    image.header.set_xyzt_units("mm")
    singular = affine.copy()
    singular[0, :3] = 0
    image.set_sform(singular, code=4)
    path = tmp_path / "singular.nii"
    nib.save(image, path)
    with pytest.raises(QCError) as error:
        load_volume(path, PreprocessingConfig())
    assert error.value.reason == "affine"


@pytest.mark.parametrize("kwargs", [{"canonical_z_mm": float("nan")}, {"voxel_spacing_mm": 0},
                                    {"spacing_tolerance_mm": -1}, {"max_voxels": 0},
                                    {"affine_tolerance": 1}])
def test_invalid_preprocessing_configuration(kwargs):
    with pytest.raises(ValueError):
        PreprocessingConfig(**kwargs)
