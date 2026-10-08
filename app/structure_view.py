from __future__ import annotations

import colorsys
from itertools import product

import numpy as np


# Compact Jmol-like palette for common elements. Unknown elements get a stable
# colour derived from their atomic number, so arbitrary CIF files still render.
_JMOL_COLOURS = {
    "H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "Na": "#AB5CF2", "Mg": "#8AFF00", "Al": "#BFA6A6",
    "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F",
    "K": "#8F40D4", "Ca": "#3DFF00", "Ti": "#BFC2C7", "V": "#A6A6AB",
    "Cr": "#8A99C7", "Mn": "#9C7AC7", "Fe": "#E06633", "Co": "#F090A0",
    "Ni": "#50D050", "Cu": "#C88033", "Zn": "#7D80B0", "Ga": "#C28F8F",
    "Ge": "#668F8F", "As": "#BD80E3", "Se": "#FFA100", "Br": "#A62929",
    "Rb": "#702EB0", "Sr": "#00FF00", "Y": "#94FFFF", "Zr": "#94E0E0",
    "Nb": "#73C2C9", "Mo": "#54B5B5", "Ru": "#248F8F", "Rh": "#0A7D8C",
    "Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F", "In": "#A67573",
    "Sn": "#668080", "Sb": "#9E63B5", "Te": "#D47A00", "I": "#940094",
    "Cs": "#57178F", "Ba": "#00C900", "La": "#70D4FF", "Ce": "#FFFFC7",
    "Hf": "#4DC2FF", "Ta": "#4DA6FF", "W": "#2194D6", "Re": "#267DAB",
    "Os": "#266696", "Ir": "#175487", "Pt": "#D0D0E0", "Au": "#FFD123",
    "Hg": "#B8B8D0", "Tl": "#A6544D", "Pb": "#575961", "Bi": "#9E4FB5",
    "Li": "#CC80FF",
}


def install_fast_interface_preview_profile() -> None:
    """Use a lighter atom mesh for large interface supercells.

    unit-cell-gui deliberately owns rendering while the host owns scene
    preparation.  Its default atom sphere is tuned for ordinary unit cells.
    Interface supercells can contain hundreds of sites, so this application
    substitutes a lower-tessellation sphere generator before the viewer is
    instantiated.  Rendering behaviour and the public unit-cell-gui API stay
    unchanged; only the number of triangles per atom is reduced.
    """
    import unit_cell_gui.opengl_viewer as gl_viewer
    from unit_cell_gui.gl_geometry import sphere_mesh as base_sphere_mesh

    if getattr(gl_viewer, "_interface_matcher_fast_spheres", False):
        return

    def fast_sphere_mesh(center, radius, **_kwargs):
        return base_sphere_mesh(
            center,
            radius,
            latitude_segments=8,
            longitude_segments=12,
        )

    gl_viewer.sphere_mesh = fast_sphere_mesh
    gl_viewer._interface_matcher_fast_spheres = True


def _element_symbol(specie) -> str:
    symbol = getattr(specie, "symbol", None)
    if symbol:
        return str(symbol)
    element = getattr(specie, "element", None)
    symbol = getattr(element, "symbol", None)
    if symbol:
        return str(symbol)
    text = str(specie)
    letters = "".join(ch for ch in text if ch.isalpha())
    return letters[:2].capitalize() if letters else "X"


def _atomic_number(specie) -> int:
    for obj in (specie, getattr(specie, "element", None)):
        if obj is None:
            continue
        value = getattr(obj, "Z", None)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            pass
    return 1


def _colour_for(specie) -> str:
    symbol = _element_symbol(specie)
    if symbol in _JMOL_COLOURS:
        return _JMOL_COLOURS[symbol]
    hue = (_atomic_number(specie) * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.45, 0.92)
    return f"#{round(red * 255):02X}{round(green * 255):02X}{round(blue * 255):02X}"


def _radius_for(specie) -> float:
    """Return a compact sphere radius suitable for unit-cell-gui."""
    element = getattr(specie, "element", specie)
    value = getattr(element, "atomic_radius", None)
    try:
        radius_angstrom = float(value)
    except (TypeError, ValueError):
        radius_angstrom = 1.1
    return min(0.38, max(0.13, 0.11 + 0.12 * radius_angstrom))


def _group_layers(indices: list[int], projection: np.ndarray, tolerance: float) -> list[list[int]]:
    """Group indices into approximately coplanar layers along the slab normal."""
    ordered = sorted(indices, key=lambda index: float(projection[index]))
    if not ordered:
        return []

    groups: list[list[int]] = [[ordered[0]]]
    mean_value = float(projection[ordered[0]])
    for index in ordered[1:]:
        value = float(projection[index])
        if abs(value - mean_value) <= tolerance:
            groups[-1].append(index)
            mean_value = float(np.mean([projection[item] for item in groups[-1]]))
        else:
            groups.append([index])
            mean_value = value
    return groups


def interface_layer_indices(interface, layers_per_side: int, *, tolerance: float = 0.35) -> list[int]:
    """Return N atomic layers nearest the film/substrate interface on each side.

    ``pymatgen.core.interface.Interface`` exposes exact ``film_indices`` and
    ``substrate_indices``.  Atomic layers are then grouped by projection onto
    the interface c-axis.  Which end of each slab faces the interface is inferred
    from the relative mean positions, so the routine is not tied to +z.
    """
    count = max(1, int(layers_per_side))
    film = [int(index) for index in interface.film_indices]
    substrate = [int(index) for index in interface.substrate_indices]
    if not film or not substrate:
        return list(range(len(interface)))

    matrix = np.asarray(interface.lattice.matrix, dtype=float)
    normal = matrix[2].copy()
    length = float(np.linalg.norm(normal))
    if length <= 1e-12:
        return list(range(len(interface)))
    normal /= length

    coords = np.asarray(interface.cart_coords, dtype=float)
    projection = coords @ normal
    film_groups = _group_layers(film, projection, tolerance)
    substrate_groups = _group_layers(substrate, projection, tolerance)

    film_mean = float(np.mean(projection[film]))
    substrate_mean = float(np.mean(projection[substrate]))
    if film_mean >= substrate_mean:
        # Conventional Interface.from_slabs layout: film above substrate.
        film_selected = film_groups[:count]
        substrate_selected = substrate_groups[-count:]
    else:
        film_selected = film_groups[-count:]
        substrate_selected = substrate_groups[:count]

    selected = {
        index
        for group in (*film_selected, *substrate_selected)
        for index in group
    }
    return sorted(selected)


def scene_from_structure(structure, atom_indices: list[int] | None = None):
    """Convert a pymatgen Structure/Interface to a unit-cell-gui Scene.

    This intentionally keeps the first integration light: atoms + unit-cell
    geometry are rendered by unit-cell-gui, while bond and polyhedron inference
    remain outside the interface matcher for now.
    """
    from unit_cell_gui import Atom, AtomComponent, Scene

    if atom_indices is None:
        atom_indices = list(range(len(structure)))
    else:
        atom_indices = [int(index) for index in atom_indices]

    lattice_rows = np.asarray(structure.lattice.matrix, dtype=float)
    direct = lattice_rows.T  # unit-cell-gui/XRD Combine convention: basis as columns.
    cell_centre = direct @ np.full(3, 0.5)

    atoms = []
    centers = []
    elements: set[str] = set()
    for display_index, structure_index in enumerate(atom_indices):
        site = structure[structure_index]
        components = []
        dominant_symbol = "X"
        for component_index, (specie, occupancy) in enumerate(site.species.items()):
            symbol = _element_symbol(specie)
            if component_index == 0:
                dominant_symbol = symbol
            elements.add(symbol)
            label = getattr(site, "label", None) or f"{symbol}{structure_index + 1}"
            components.append(
                AtomComponent(
                    key=f"{structure_index}:{component_index}:{symbol}",
                    label=str(label),
                    element=symbol,
                    occupancy=float(occupancy),
                    colour=_colour_for(specie),
                    radius=_radius_for(specie),
                )
            )
        if not components:
            continue
        site_label = getattr(site, "label", None) or f"{dominant_symbol}{structure_index + 1}"
        atoms.append(
            Atom(
                site_key=f"site:{structure_index}",
                site_label=str(site_label),
                components=tuple(components),
            )
        )
        centers.append(np.asarray(site.coords, dtype=float) - cell_centre)

    fractional_corners = np.asarray(list(product((0.0, 1.0), repeat=3)), dtype=float)
    cell_vertices = fractional_corners @ direct.T - cell_centre
    cell_edges = np.asarray(
        [
            (first, second)
            for first in range(8)
            for second in range(first + 1, 8)
            if np.count_nonzero(fractional_corners[first] != fractional_corners[second]) == 1
        ],
        dtype=int,
    ).reshape(-1, 2)

    atom_centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    extents = [float(np.linalg.norm(point)) for point in cell_vertices]
    extents.extend(float(np.linalg.norm(point)) + 0.4 for point in atom_centers)
    base_radius = max(max(extents, default=1.0) * 1.08, 1e-6)

    return Scene(
        atoms=tuple(atoms),
        atom_centers=atom_centers,
        bonds=np.empty((0, 2), dtype=int),
        cell_vertices=cell_vertices,
        cell_edges=cell_edges,
        basis_vectors=direct,
        polyhedra=(),
        base_radius=base_radius,
        elements=tuple(sorted(elements)),
    )
