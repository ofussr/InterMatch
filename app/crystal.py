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
    """Get termination shifts across old and new pymatgen releases.

    pymatgen-core 2026.8.13 made ``SlabGenerator.gen_possible_terminations``
    public. Older pymatgen versions either expose the same calculation through
    the private ``_calculate_possible_shifts`` helper or only through
    ``get_slabs()``, whose returned ``Slab`` objects carry their ``shift``.

    Keeping all three paths here lets Interface Matcher work with existing
    pymatgen installations instead of requiring a specific recent release.
    """
    public = getattr(gen, "gen_possible_terminations", None)
    if callable(public):
        return _unique_float_values(public())

    private = getattr(gen, "_calculate_possible_shifts", None)
    if callable(private):
        try:
            return _unique_float_values(private(tol=0.1))
        except TypeError:
            # Some historical releases used a different/no explicit signature.
            return _unique_float_values(private())

    # Last-resort public API for older pymatgen versions.  get_slabs() has
    # generated unique terminations for many years and each Slab stores the
    # fractional c shift used to create it.
    try:
        slabs = gen.get_slabs(filter_out_sym_slabs=True)
    except TypeError:
        # Very old releases do not have filter_out_sym_slabs.
        slabs = gen.get_slabs()

    return _unique_float_values(
        slab.shift for slab in slabs if hasattr(slab, "shift")
    )


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
