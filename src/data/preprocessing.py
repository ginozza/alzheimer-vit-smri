"""Technical QC for brain-extracted, registered MNI152 NIfTI derivatives."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import math

import nibabel as nib
import numpy as np

from .slice_representation import CanonicalTripletExtractor, Slice25DConfig


class QCError(ValueError):
    """A recoverable input rejection with a stable machine-readable reason."""

    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class PreprocessingConfig:
    canonical_z_mm: float = -12.0
    voxel_spacing_mm: float = 1.0
    spacing_tolerance_mm: float = 0.05
    affine_tolerance: float = 0.0001
    max_voxels: int = 20000000
    normalization: str = "z_score"
    clip_percentiles: tuple[float, float] = (1.0, 99.0)

    def __post_init__(self):
        for name in ("canonical_z_mm", "voxel_spacing_mm", "spacing_tolerance_mm", "affine_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"Invalid {name}")
        if self.voxel_spacing_mm <= 0 or not 0 <= self.spacing_tolerance_mm < self.voxel_spacing_mm:
            raise ValueError("Invalid voxel spacing or tolerance")
        if not 0 < self.affine_tolerance < 0.01:
            raise ValueError("affine_tolerance must be between 0 and 0.01")
        if type(self.max_voxels) is not int or self.max_voxels < 1:
            raise ValueError("max_voxels must be positive")
        Slice25DConfig(normalization=self.normalization, clip_percentiles=self.clip_percentiles)

    @classmethod
    def load(cls, path: str | Path):
        values = json.loads(Path(path).read_text())
        return cls(**values)

    def as_dict(self):
        return asdict(self)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(condition: bool, reason: str, message: str):
    if not condition:
        raise QCError(reason, message)


def load_volume(path: Path, config: PreprocessingConfig) -> tuple[np.ndarray, dict]:
    """Check geometry before loading the array; canonicalization only permutes axes."""
    _check(path.name.endswith((".nii", ".nii.gz")), "format", "Expected .nii or .nii.gz")
    try:
        image = nib.load(path)
        _check(isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)), "format", "Expected NIfTI")
        _check(len(image.shape) == 3 and min(image.shape) > 1, "dimensions", "Expected nondegenerate 3D volume")
        _check(math.prod(image.shape) <= config.max_voxels, "size_limit", "Volume exceeds configured voxel limit")
        _check(image.header.get_xyzt_units()[0] == "mm", "units", "Spatial units must be mm")
        affine = image.affine
        _check(np.isfinite(affine).all() and abs(np.linalg.det(affine[:3, :3])) > 1e-8,
               "affine", "Affine must be finite and invertible")
        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
        _check(bool(qcode or scode), "affine", "No coded spatial transform")
        if qcode and scode:
            _check(np.allclose(qform, sform, atol=config.affine_tolerance, rtol=0),
                   "affine_conflict", "qform and sform disagree; resolve provenance upstream")
        spacing = nib.affines.voxel_sizes(affine)
        _check(np.allclose(spacing, image.header.get_zooms(), atol=config.affine_tolerance, rtol=0),
               "spacing", "Header zooms and affine disagree")
        _check(np.all(np.abs(spacing - config.voxel_spacing_mm) <= config.spacing_tolerance_mm),
               "spacing", "Unexpected spacing; upstream resampling is required")
        canonical = nib.as_closest_canonical(image)
        linear = canonical.affine[:3, :3]
        _check(np.allclose(linear, np.diag(np.diag(linear)), atol=config.affine_tolerance, rtol=0),
               "oblique", "Oblique or sheared grid is not an axial MNI grid")
        volume = canonical.get_fdata(dtype=np.float32)
        _check(np.isfinite(volume).all(), "nonfinite", "NaN/Inf intensities")
        _check(not np.any(volume < 0), "negative_intensity", "Expected unnormalized nonnegative intensities")
        _check(np.any(volume > 0) and np.any(volume == 0), "foreground", "Expected positive signal and zero background")
        digest = hashlib.sha256(np.asarray(volume, dtype="<f4").tobytes())
        digest.update(np.asarray(volume.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(canonical.affine, dtype="<f8").tobytes())
        return volume, {
            "original_shape": list(image.shape), "canonical_shape": list(volume.shape),
            "original_orientation": list(nib.aff2axcodes(affine)), "canonical_orientation": list(nib.aff2axcodes(canonical.affine)),
            "original_affine": affine.tolist(), "canonical_affine": canonical.affine.tolist(),
            "spacing_mm": nib.affines.voxel_sizes(canonical.affine).tolist(),
            "qform_code": int(qcode), "sform_code": int(scode),
            "intensity_min": float(volume.min()), "intensity_max": float(volume.max()),
            "foreground_voxels": int(np.count_nonzero(volume)),
            "content_sha256": digest.hexdigest(),
        }
    except QCError:
        raise
    except (OSError, ValueError, EOFError, nib.filebasedimages.ImageFileError) as error:
        raise QCError("unreadable", str(error)) from error


def generate_input(volume: np.ndarray, geometry: dict, config: PreprocessingConfig) -> tuple[np.ndarray, dict]:
    affine = np.array(geometry["canonical_affine"])
    index_float = (config.canonical_z_mm - affine[2, 3]) / affine[2, 2]
    center = int(np.floor(index_float + 0.5))
    indices = [center - 1, center, center + 1]
    _check(indices[0] >= 0 and indices[-1] < volume.shape[2], "slice_bounds", "Canonical triplet outside volume")
    signal = np.stack([volume[:, :, z] > 0 for z in indices])
    cropped = CanonicalTripletExtractor._resize_or_pad(signal, 224, 224)
    _check(np.count_nonzero(signal) == np.count_nonzero(cropped), "foreground_crop", "Center crop would discard foreground")
    extraction_config = replace(Slice25DConfig(), canonical_z_index=center,
                                normalization=config.normalization, clip_percentiles=config.clip_percentiles)
    try:
        tensor, statistics = CanonicalTripletExtractor(extraction_config).extract_with_metadata(volume)
    except ValueError as error:
        raise QCError("slice_signal", str(error)) from error
    _check(tensor.shape == (3, 224, 224) and np.isfinite(tensor).all(), "output", "Invalid output tensor")
    return tensor, {
        "slice_indices": indices,
        "slice_z_mm": [float(affine[2, 2] * z + affine[2, 3]) for z in indices],
        "center_offset_mm": float(affine[2, 2] * center + affine[2, 3] - config.canonical_z_mm),
        "normalization": statistics, "spatial_operation": "center_crop_or_zero_pad",
        "output_axes": "CXY; X increases left-to-right, Y posterior-to-anterior",
    }
