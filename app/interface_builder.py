from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matcher import InterfaceMatch


@dataclass
class BuiltInterface:
    interface: object
    film_supercell: object
    substrate_supercell: object


def _scale_matrix_3d(m2: np.ndarray) -> np.ndarray:
    m2 = np.asarray(m2, dtype=int)
    return np.array(
        [[m2[0, 0], m2[0, 1], 0], [m2[1, 0], m2[1, 1], 0], [0, 0, 1]],
        dtype=int,
    )


def build_interface(film_surface, substrate_surface, match: InterfaceMatch, *, gap: float = 2.0, vacuum: float = 15.0, offset=(0.0, 0.0)) -> BuiltInterface:
    try:
        from pymatgen.core.interface import Interface
    except ImportError as exc:
        raise RuntimeError("pymatgen is required to build the atomic interface") from exc

    film = film_surface.slab.copy()
    substrate = substrate_surface.slab.copy()

    film.make_supercell(_scale_matrix_3d(match.film_matrix))
    substrate.make_supercell(_scale_matrix_3d(match.substrate_matrix))

    interface = Interface.from_slabs(
        substrate,
        film,
        in_plane_offset=(float(offset[0]), float(offset[1])),
        gap=float(gap),
        vacuum_over_film=float(vacuum),
        interface_properties={
            "film_hkl": tuple(film_surface.hkl),
            "substrate_hkl": tuple(substrate_surface.hkl),
            "film_matrix": np.asarray(match.film_matrix, dtype=int).tolist(),
            "substrate_matrix": np.asarray(match.substrate_matrix, dtype=int).tolist(),
            "max_abs_strain": float(match.max_abs_strain),
            "rotation_deg": float(match.rotation_deg),
        },
        center_slab=True,
    )
    return BuiltInterface(interface=interface, film_supercell=film, substrate_supercell=substrate)


def save_cif(structure, path: str):
    try:
        from pymatgen.io.cif import CifWriter
    except ImportError as exc:
        raise RuntimeError("pymatgen is required to export CIF") from exc
    CifWriter(structure, symprec=None).write_file(path)
