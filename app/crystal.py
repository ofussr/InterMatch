from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .matcher import as_2d_basis


@dataclass
class SurfaceModel:
    path: Path
    structure: object
    slab: object
    basis_2d: np.ndarray
    hkl: tuple[int, int, int]
    termination_shift: float


def require_pymatgen():
    try:
        from pymatgen.core import Structure  # noqa: F401
        from pymatgen.core.surface import SlabGenerator  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pymatgen is required for CIF/slab operations. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc


def load_structure(path: str | Path):
    require_pymatgen()
    from pymatgen.core import Structure

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return Structure.from_file(str(path))


def _unique_float_values(values, *, atol: float = 1e-10) -> list[float]:
    """Return finite float values with near-duplicates removed, preserving order."""
    result: list[float] = []
    for value in values:
        value = float(value)
        if not np.isfinite(value):
            continue
        if not any(abs(value - existing) <= atol for existing in result):
            result.append(value)
    return result


def _possible_termination_shifts(gen) -> list[float]:
    """List termination shifts WITHOUT building and comparing every slab.

    Recent pymatgen versions offer a lightweight public method. On older
    versions compute mid-gaps between atomic layers in the oriented unit cell
    directly. Crucially, never call ``get_slabs`` here: that method constructs
    many full slabs and may spend minutes in StructureMatcher for a complex CIF.
    """
    public = getattr(gen, "gen_possible_terminations", None)
    if callable(public):
        return sorted(_unique_float_values(public()))

    oriented = getattr(gen, "oriented_unit_cell", None)
    if oriented is None:
        raise RuntimeError("SlabGenerator has no oriented_unit_cell")

    coordinates = np.asarray(oriented.frac_coords, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Invalid oriented unit-cell coordinates")
    if len(coordinates) == 0:
        return []

    z = np.sort(np.mod(coordinates[:, 2], 1.0))
    if len(z) == 1:
        return [float((z[0] + 0.5) % 1.0)]

    height = getattr(gen, "_proj_height", None)
    if height is None:
        lattice = np.asarray(oriented.lattice.matrix, dtype=float)
        height = abs(np.linalg.det(lattice)) / np.linalg.norm(
            np.cross(lattice[0], lattice[1])
        )
    height = float(height)
    if not np.isfinite(height) or height <= 0:
        raise ValueError("Invalid projected cell height")

    # Cluster adjacent atomic heights across the periodic boundary.  This is
    # a one-dimensional O(N log N) procedure, not an O(N²) slab comparison.
    threshold_fractional = min(0.49, 0.1 / height)
    gaps = np.diff(np.r_[z, z[0] + 1.0])
    boundaries = np.flatnonzero(gaps > threshold_fractional)
    if len(boundaries) == 0:
        # One periodic atomic layer spanning the entire c period.
        angle = np.angle(np.mean(np.exp(2j * np.pi * z)))
        return [float((angle / (2 * np.pi) + 0.5) % 1.0)]

    # Start immediately after a cluster boundary, then walk cyclically.
    start = (int(boundaries[0]) + 1) % len(z)
    groups: list[list[float]] = []
    group: list[float] = []
    for offset in range(len(z)):
        idx = (start + offset) % len(z)
        group.append(float(z[idx]))
        if gaps[idx] > threshold_fractional:
            groups.append(group)
            group = []

    centers = sorted(float(np.angle(np.mean(np.exp(2j * np.pi * np.asarray(group))))
                           / (2 * np.pi) % 1.0) for group in groups)
    shifts = [(first + second) / 2 for first, second in zip(centers, centers[1:])]
    shifts.append(((centers[-1] + centers[0] + 1.0) / 2.0) % 1.0)
    return sorted(_unique_float_values(shifts))


def _make_slab_generator(structure, hkl, slab_thickness: float):
    from pymatgen.core.surface import SlabGenerator

    return SlabGenerator(
        structure,
        hkl,
        min_slab_size=float(slab_thickness),
        min_vacuum_size=3.0,
        center_slab=False,
        in_unit_planes=False,
        primitive=True,
        max_normal_search=max(abs(x) for x in hkl) or 1,
        reorient_lattice=True,
    )


def build_surface(
    path: str | Path,
    hkl: tuple[int, int, int],
    *,
    slab_thickness: float = 15.0,
    termination_shift: float | None = None,
):
    require_pymatgen()

    structure = load_structure(path)
    if hkl == (0, 0, 0):
        raise ValueError("Miller index (0 0 0) is invalid")

    gen = _make_slab_generator(structure, hkl, slab_thickness)
    if termination_shift is None:
        shifts = _possible_termination_shifts(gen)
        termination_shift = shifts[0] if shifts else 0.0

    slab = gen.get_slab(shift=float(termination_shift))
    a3, b3 = slab.lattice.matrix[0], slab.lattice.matrix[1]
    basis = as_2d_basis(a3, b3)
    return SurfaceModel(
        path=Path(path),
        structure=structure,
        slab=slab,
        basis_2d=basis,
        hkl=hkl,
        termination_shift=float(termination_shift),
    )


def possible_terminations(
    path: str | Path,
    hkl: tuple[int, int, int],
    *,
    slab_thickness: float = 15.0,
) -> list[float]:
    require_pymatgen()

    structure = load_structure(path)
    if hkl == (0, 0, 0):
        raise ValueError("Miller index (0 0 0) is invalid")

    gen = _make_slab_generator(structure, hkl, slab_thickness)
    return _possible_termination_shifts(gen)
