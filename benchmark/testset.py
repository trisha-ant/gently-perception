"""
Offline Testset for Perception Benchmarks.

Loads session data and pairs with ground truth for sequential testing.
"""

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Dict

import numpy as np

from .ground_truth import GroundTruth

# Lazy imports
tifffile = None
PIL_Image = None


def _ensure_dependencies():
    """Ensure required dependencies are available."""
    global tifffile, PIL_Image

    if tifffile is None:
        import tifffile as _tifffile
        tifffile = _tifffile

    if PIL_Image is None:
        from PIL import Image as _Image
        PIL_Image = _Image


@dataclass
class TestCase:
    """A single test case for perception benchmark."""

    embryo_id: str
    timepoint: int
    image_b64: str  # Combined view (for backward compatibility)
    top_image_b64: Optional[str]  # TOP view only
    side_image_b64: Optional[str]  # SIDE view only
    midplane_b64: Optional[str]  # Single XY slice at z=Z//2 (no projection)
    volume: Optional[np.ndarray]
    ground_truth_stage: Optional[str]


def _discover_volumes(session_dir: Path, embryo_id: Optional[str] = None) -> Dict[str, List[Path]]:
    """Discover volume files in a session directory."""
    from datetime import datetime

    if not session_dir.exists():
        return {}

    tif_files = (
        list(session_dir.glob("*.tif")) + list(session_dir.glob("*.tiff"))
        + list(session_dir.glob("**/*.tif")) + list(session_dir.glob("**/*.tiff"))
    )
    # Deduplicate (flat + recursive may overlap)
    tif_files = list({f.resolve(): f for f in tif_files}.values())
    embryo_volumes = {}

    for f in tif_files:
        parts = f.stem.split("_")
        if len(parts) >= 3:
            eid = f"{parts[0]}_{parts[1]}"

            try:
                timestamp_str = f"{parts[2]}_{parts[3]}" if len(parts) >= 4 else parts[2]
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except (ValueError, IndexError):
                timestamp = datetime.fromtimestamp(f.stat().st_mtime)

            if embryo_id is None or eid == embryo_id:
                if eid not in embryo_volumes:
                    embryo_volumes[eid] = []
                embryo_volumes[eid].append((timestamp, f))

    result = {}
    for eid, volumes in embryo_volumes.items():
        volumes.sort(key=lambda x: x[0])
        result[eid] = [v[1] for v in volumes]

    return result


def _load_volume(path: Path) -> np.ndarray:
    """Load a volume from TIFF file, extract View A if dual-view."""
    _ensure_dependencies()
    vol = tifffile.imread(str(path))
    vol = np.squeeze(vol)
    if vol.ndim == 3:
        z_depth, height, width = vol.shape
        if width > height * 2:
            vol = vol[:, :, :width // 2]
    return vol


def _normalize_image(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Normalize image to 0-255 uint8."""
    img = img.astype(np.float32)
    vmin = np.percentile(img, p_low)
    vmax = np.percentile(img, p_high)
    if vmax > vmin:
        img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    else:
        img = np.zeros_like(img)
    return (img * 255).astype(np.uint8)


def _compute_crop_bounds(volume: np.ndarray, padding: int = 20, sigma_mult: float = 3.5):
    """Compute crop bounds using center-of-mass of bright pixels."""
    if volume.ndim != 3:
        return (0, volume.shape[0], 0, volume.shape[1])
    max_proj = np.max(volume, axis=0).astype(np.float32)
    threshold = np.percentile(max_proj, 95)
    mask = max_proj > threshold
    y_coords, x_coords = np.where(mask)
    if len(y_coords) < 10:
        return (0, volume.shape[1], 0, volume.shape[2])
    cy, cx = np.mean(y_coords), np.mean(x_coords)
    y_std = max(np.std(y_coords), 20)
    x_std = max(np.std(x_coords), 20)
    y_min = int(max(0, cy - sigma_mult * y_std - padding))
    y_max = int(min(volume.shape[1], cy + sigma_mult * y_std + padding))
    x_min = int(max(0, cx - sigma_mult * x_std - padding))
    x_max = int(min(volume.shape[2], cx + sigma_mult * x_std + padding))
    return (y_min, y_max, x_min, x_max)


def _projection_three_view(
    volume: np.ndarray,
    voxel_size: tuple = (1.0, 0.1625, 0.1625),
) -> np.ndarray:
    """Generate three orthogonal views layout from a 3D volume.

    Layout: [XY|YZ] top row, [XZ] bottom row.
    """
    _ensure_dependencies()

    z_depth, height, width = volume.shape
    dz, dy, dx = voxel_size

    xy_proj = _normalize_image(np.max(volume, axis=0))
    xz_proj = _normalize_image(np.max(volume, axis=1))
    yz_proj = _normalize_image(np.max(volume, axis=2))

    xy_h, xy_w = xy_proj.shape
    z_scale = dz / dx
    xz_display_h = max(1, int(z_depth * z_scale))

    pil_xz = PIL_Image.fromarray(xz_proj)
    pil_xz = pil_xz.resize((xy_w, xz_display_h), PIL_Image.Resampling.LANCZOS)
    xz_scaled = np.array(pil_xz)

    yz_rotated = yz_proj.T
    yz_display_w = xz_display_h

    pil_yz = PIL_Image.fromarray(yz_rotated)
    pil_yz = pil_yz.resize((yz_display_w, xy_h), PIL_Image.Resampling.LANCZOS)
    yz_scaled = np.array(pil_yz)

    sep = 3
    v_sep = np.ones((xy_h, sep), dtype=np.uint8) * 128
    top_row = np.concatenate([xy_proj, v_sep, yz_scaled], axis=1)
    total_width = top_row.shape[1]

    if xz_scaled.shape[1] < total_width:
        pad = np.zeros((xz_scaled.shape[0], total_width - xz_scaled.shape[1]), dtype=np.uint8)
        bottom_row = np.concatenate([xz_scaled, pad], axis=1)
    else:
        bottom_row = xz_scaled[:, :total_width]

    h_sep = np.ones((sep, total_width), dtype=np.uint8) * 128
    return np.concatenate([top_row, h_sep, bottom_row], axis=0)


def _create_three_view_image(volume: np.ndarray, max_dim: int = 1500) -> str:
    """Create three-view orthogonal projection from volume, return base64.

    Uses the shared projection utility to generate:
    - XY (top-left): Looking down - best for shape, curvature, folding
    - YZ (top-right): Looking from side - best for depth, body height
    - XZ (bottom): Looking from front - best for symmetry, coiling

    Parameters
    ----------
    volume : np.ndarray
        3D volume array (Z, Y, X)
    max_dim : int
        Maximum dimension in pixels (default 1500 to stay under API limits)
    """
    _ensure_dependencies()

    # Auto-crop to embryo region
    bounds = _compute_crop_bounds(volume)
    cropped = volume[:, bounds[0]:bounds[1], bounds[2]:bounds[3]]

    # Generate three-view projection
    three_view_img = _projection_three_view(cropped)

    # Convert to PIL for final processing
    pil_img = PIL_Image.fromarray(three_view_img)

    # Resize if too large (API limit is 8000px, use smaller for safety/performance)
    if max(pil_img.size) > max_dim:
        scale = max_dim / max(pil_img.size)
        new_size = (int(pil_img.size[0] * scale), int(pil_img.size[1] * scale))
        pil_img = pil_img.resize(new_size, PIL_Image.Resampling.LANCZOS)

    # Convert to JPEG base64
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=90)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _create_slice_image(
    volume: np.ndarray, z: Optional[int] = None, max_dim: int = 800
) -> str:
    """Render a single XY z-slice as base64 JPEG.

    Unlike the projection helpers, this preserves depth information at one
    plane: overlapping body segments stay separated rather than fusing.
    Uses the same crop as the three-view projection so framing matches.

    Parameters
    ----------
    volume : np.ndarray
        3D volume array (Z, Y, X)
    z : int, optional
        Slice index. Defaults to the midplane (Z // 2).
    max_dim : int
        Maximum output dimension in pixels.
    """
    _ensure_dependencies()

    if z is None:
        z = volume.shape[0] // 2
    z = max(0, min(z, volume.shape[0] - 1))

    bounds = _compute_crop_bounds(volume)
    slice_img = volume[z, bounds[0]:bounds[1], bounds[2]:bounds[3]]
    slice_norm = _normalize_image(slice_img)

    pil_img = PIL_Image.fromarray(slice_norm)
    if max(pil_img.size) > max_dim:
        scale = max_dim / max(pil_img.size)
        new_size = (int(pil_img.size[0] * scale), int(pil_img.size[1] * scale))
        pil_img = pil_img.resize(new_size, PIL_Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _create_separate_view_images(volume: np.ndarray, max_dim: int = 1000) -> Tuple[str, str]:
    """Create separate TOP and SIDE view images from volume, return base64 tuple.

    Parameters
    ----------
    volume : np.ndarray
        3D volume array (Z, Y, X)
    max_dim : int
        Maximum dimension in pixels

    Returns
    -------
    Tuple[str, str]
        (top_image_b64, side_image_b64)
    """
    _ensure_dependencies()

    z_depth, height, width = volume.shape

    # TOP: max along Z (axis 0) -> shape (Y, X) - looking down
    top_proj = np.max(volume, axis=0)

    # SIDE: max along Y (axis 1) -> shape (Z, X) - looking from front
    side_proj = np.max(volume, axis=1)

    # Normalize
    top_norm = _normalize_image(top_proj)
    side_norm = _normalize_image(side_proj)

    # Scale side view to make Z dimension more visible
    target_width = top_norm.shape[1]
    side_new_h = max(height // 3, int(z_depth * 3))

    side_pil = PIL_Image.fromarray(side_norm)
    side_scaled = side_pil.resize((target_width, side_new_h), PIL_Image.Resampling.LANCZOS)

    # Create PIL images
    top_pil = PIL_Image.fromarray(top_norm)

    # Resize if too large
    if max(top_pil.size) > max_dim:
        scale = max_dim / max(top_pil.size)
        new_size = (int(top_pil.size[0] * scale), int(top_pil.size[1] * scale))
        top_pil = top_pil.resize(new_size, PIL_Image.Resampling.LANCZOS)

    if max(side_scaled.size) > max_dim:
        scale = max_dim / max(side_scaled.size)
        new_size = (int(side_scaled.size[0] * scale), int(side_scaled.size[1] * scale))
        side_scaled = side_scaled.resize(new_size, PIL_Image.Resampling.LANCZOS)

    # Convert to JPEG base64
    top_buffer = io.BytesIO()
    top_pil.save(top_buffer, format="JPEG", quality=90)
    top_b64 = base64.b64encode(top_buffer.getvalue()).decode("utf-8")

    side_buffer = io.BytesIO()
    side_scaled.save(side_buffer, format="JPEG", quality=90)
    side_b64 = base64.b64encode(side_buffer.getvalue()).decode("utf-8")

    return top_b64, side_b64


class OfflineTestset:
    """
    Offline testset for perception benchmarks.

    Loads session data from disk and pairs with ground truth labels.
    Supports sequential iteration through embryos to simulate real-time acquisition.
    """

    def __init__(
        self,
        session_path: Path,
        ground_truth: GroundTruth,
        load_volumes: bool = True,
    ):
        """
        Parameters
        ----------
        session_path : Path
            Path to session directory containing TIF volumes
        ground_truth : GroundTruth
            Ground truth labels for this session
        load_volumes : bool
            Whether to load full 3D volumes (for view_embryo tool testing)
        """
        self.session_path = Path(session_path)
        self.ground_truth = ground_truth
        self.load_volumes = load_volumes

        # Discover available volumes
        self._embryo_volumes = _discover_volumes(self.session_path)

    @property
    def embryo_ids(self) -> List[str]:
        """Get list of embryo IDs with both volumes and ground truth."""
        gt_embryos = set(self.ground_truth.embryo_ids)
        vol_embryos = set(self._embryo_volumes.keys())
        return sorted(gt_embryos & vol_embryos)

    def get_timepoint_count(self, embryo_id: str) -> int:
        """Get number of timepoints for an embryo."""
        return len(self._embryo_volumes.get(embryo_id, []))

    def iter_embryo(
        self,
        embryo_id: str,
        start_timepoint: int = 0,
        end_timepoint: Optional[int] = None,
    ) -> Iterator[TestCase]:
        """
        Iterate through timepoints for an embryo sequentially.

        Yields TestCase objects in temporal order, simulating real-time acquisition.

        Parameters
        ----------
        embryo_id : str
            Embryo ID to iterate
        start_timepoint : int
            First timepoint to include
        end_timepoint : int, optional
            Last timepoint to include (exclusive)

        Yields
        ------
        TestCase
            Test case with image, volume, and ground truth
        """
        if embryo_id not in self._embryo_volumes:
            return

        volumes = self._embryo_volumes[embryo_id]

        if end_timepoint is None:
            end_timepoint = len(volumes)

        for timepoint in range(start_timepoint, min(end_timepoint, len(volumes))):
            vol_path = volumes[timepoint]

            # Load volume
            volume = _load_volume(vol_path) if self.load_volumes else None

            # Create images
            vol = volume if volume is not None else _load_volume(vol_path)
            image_b64 = _create_three_view_image(vol)
            top_b64, side_b64 = _create_separate_view_images(vol)
            midplane_b64 = _create_slice_image(vol)
            if volume is None:
                del vol

            # Get ground truth
            gt_stage = self.ground_truth.get_stage_at(embryo_id, timepoint)

            yield TestCase(
                embryo_id=embryo_id,
                timepoint=timepoint,
                image_b64=image_b64,
                top_image_b64=top_b64,
                side_image_b64=side_b64,
                midplane_b64=midplane_b64,
                volume=volume,
                ground_truth_stage=gt_stage,
            )

    def iter_all(self) -> Iterator[Tuple[str, Iterator[TestCase]]]:
        """
        Iterate through all embryos in the testset.

        Yields (embryo_id, test_case_iterator) pairs.
        """
        for embryo_id in self.embryo_ids:
            yield embryo_id, self.iter_embryo(embryo_id)
