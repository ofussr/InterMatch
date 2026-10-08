from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SupercellCandidate:
    """One unique 2D supercell of a parent surface lattice."""

    matrix: np.ndarray          # 2x2 integer matrix acting on parent surface rows
    basis: np.ndarray           # 2x2 reduced basis, row-vector convention
    area: float
    determinant: int
    angle_deg: float            # orientation of first supercell vector vs parent a_surf


@dataclass(frozen=True)
class InterfaceMatch:
    film_matrix: np.ndarray
    substrate_matrix: np.ndarray
    film_basis: np.ndarray
    substrate_basis: np.ndarray
    film_area: float
    substrate_area: float
    target_area: float
    film_det: int
    substrate_det: int
    atom_count: int
    rotation_deg: float
    eps_xx: float
    eps_yy: float
    eps_xy: float
    principal_1: float
    principal_2: float
    max_abs_strain: float
    area_mismatch: float
    score: float


def as_2d_basis(a3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """Convert two 3D in-plane vectors to an equivalent right-handed 2D basis.

    Rotation in 3D is deliberately discarded; lengths and the included angle are kept.
    Rows are lattice vectors.
    """

    a3 = np.asarray(a3, dtype=float)
    b3 = np.asarray(b3, dtype=float)
    la = np.linalg.norm(a3)
    if la <= 0:
        raise ValueError("Zero-length surface vector a")
    ahat = a3 / la
    bx = float(np.dot(b3, ahat))
    by2 = float(np.dot(b3, b3) - bx * bx)
    by = np.sqrt(max(0.0, by2))
    if by <= 1e-12:
        raise ValueError("Surface vectors are collinear")
    return np.array([[la, 0.0], [bx, by]], dtype=float)


def cell_area(basis: np.ndarray) -> float:
    return abs(float(np.linalg.det(np.asarray(basis, dtype=float))))


def hnf_matrices(max_det: int) -> Iterable[np.ndarray]:
    """Yield 2D Hermite-normal-form matrices with determinant <= max_det.

    H = [[a, b], [0, d]], a*d = n, 0 <= b < d.
    This enumerates sublattices without the huge duplication of a raw integer search.
    """

    if max_det < 1:
        return
    for n in range(1, max_det + 1):
        for a in range(1, n + 1):
            if n % a:
                continue
            d = n // a
            for b in range(d):
                yield np.array([[a, b], [0, d]], dtype=int)


def gauss_reduce_2d(basis: np.ndarray, transform: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-reduce a 2D row basis and retain the integer transform.

    The termination test is written explicitly as ``|projection| <= 1/2``.
    Using ``np.rint`` directly at the half-integer boundary can make floating
    point round-off alternate between +1 and -1, producing a two-state cycle
    for perfectly valid lattices (square surfaces are a common trigger).
    """

    b = np.array(basis, dtype=float, copy=True)
    u = np.eye(2, dtype=int) if transform is None else np.array(transform, dtype=int, copy=True)

    for _ in range(100):
        n0 = float(np.dot(b[0], b[0]))
        n1 = float(np.dot(b[1], b[1]))
        scale = max(n0, n1, 1.0)
        tol = 1e-12 * scale

        # Keep the shortest vector first.  Do not swap vectors whose lengths
        # differ only by floating-point noise, otherwise equal-length bases can
        # oscillate between equivalent representations.
        if n1 < n0 - tol:
            b[[0, 1]] = b[[1, 0]]
            u[[0, 1]] = u[[1, 0]]
            continue

        if n0 <= 1e-20:
            raise ValueError("Degenerate 2D basis")

        projection = float(np.dot(b[0], b[1]) / n0)
        projection_tol = 1e-12

        # Lagrange/Gauss reduced condition.  At exactly +/-1/2 both choices are
        # equally short, so stopping is the deterministic and stable choice.
        if abs(projection) <= 0.5 + projection_tol:
            break

        # Nearest integer, with an explicit rule away from the ambiguous half
        # boundary.  At this point |projection| is strictly greater than 1/2.
        if projection > 0.0:
            mu = int(np.floor(projection + 0.5))
        else:
            mu = int(np.ceil(projection - 0.5))
        if mu == 0:  # Defensive guard; should be impossible after the test above.
            break

        b[1] -= mu * b[0]
        u[1] -= mu * u[0]
    else:
        raise RuntimeError("2D Gauss reduction did not converge")

    # Ensure a right-handed basis. Multiplying one row by -1 preserves the lattice.
    if np.linalg.det(b) < 0:
        b[1] *= -1
        u[1] *= -1

    # Make the first vector point into a deterministic half-plane. Multiplying both rows
    # by -1 keeps handedness and only changes the representation, not the lattice.
    if b[0, 0] < -1e-12 or (abs(b[0, 0]) <= 1e-12 and b[0, 1] < 0):
        b *= -1
        u *= -1

    return b, u


def generate_supercells(parent_basis: np.ndarray, max_area: float) -> list[SupercellCandidate]:
    parent_basis = np.asarray(parent_basis, dtype=float)
    base_area = cell_area(parent_basis)
    if base_area <= 0:
        raise ValueError("Degenerate parent surface lattice")
    max_det = max(1, int(np.floor(max_area / base_area + 1e-12)))

    out: list[SupercellCandidate] = []
    seen: set[tuple[float, ...]] = set()
    for h in hnf_matrices(max_det):
        raw = h @ parent_basis
        reduced, u = gauss_reduce_2d(raw)
        m_eff = u @ h

        # De-duplicate only identical shape *and orientation* representations.
        # The orientation relative to the parent surface is physically relevant because
        # it becomes the epitaxial in-plane rotation between film and substrate.
        gram = reduced @ reduced.T
        cand_angle = degrees(atan2(reduced[0, 1], reduced[0, 0])) % 180.0
        key = tuple(np.round(gram.ravel(), 8)) + (round(cand_angle, 8),)
        if key in seen:
            continue
        seen.add(key)

        area = cell_area(reduced)
        if area > max_area * (1 + 1e-10):
            continue

        if round(np.linalg.det(m_eff)) < 0:
            m_eff = m_eff.copy()
            reduced = reduced.copy()
            m_eff[1] *= -1
            reduced[1] *= -1

        angle = cand_angle
        out.append(
            SupercellCandidate(
                matrix=m_eff.astype(int),
                basis=reduced,
                area=area,
                determinant=abs(int(round(np.linalg.det(m_eff)))),
                angle_deg=angle,
            )
        )
    return out


def _strain_for_mapping(film_basis: np.ndarray, substrate_basis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row-vector map T, small/Green strain E, and principal engineering strains.

    The matched lattices obey B_film @ T = B_substrate. For column vectors the
    deformation gradient is F = T.T. Principal engineering strains are singular
    values(F) - 1, which are intuitive for ranking coherent matches.
    """

    t = np.linalg.solve(film_basis, substrate_basis)
    f = t.T
    e_green = 0.5 * (f.T @ f - np.eye(2))
    svals = np.linalg.svd(f, compute_uv=False)
    principal = np.sort(svals - 1.0)[::-1]
    return t, e_green, principal


def _angle_difference_deg(a: float, b: float) -> float:
    d = (b - a) % 180.0
    # ±90 gives a compact signed representation for an unoriented lattice direction.
    if d > 90.0:
        d -= 180.0
    return d


def find_matches(
    film_basis: np.ndarray,
    substrate_basis: np.ndarray,
    *,
    max_strain: float = 0.03,
    max_area: float = 300.0,
    max_atoms: int = 300,
    film_atoms_per_surface_cell: int = 1,
    substrate_atoms_per_surface_cell: int = 1,
    max_results: int = 250,
) -> list[InterfaceMatch]:
    """Search coherent 2D lattice matches using our own HNF + reduction implementation.

    max_strain is fractional (0.03 = 3%). The film is coherently deformed onto the
    substrate candidate; no elastic-energy model is assumed here.
    """

    if max_strain < 0:
        raise ValueError("max_strain must be non-negative")

    films = generate_supercells(film_basis, max_area)
    subs = generate_supercells(substrate_basis, max_area)

    # Area is an inexpensive pruning criterion. A 2D deformation bounded by e on both
    # principal stretches changes area roughly within (1±e)^2.
    area_ratio_limit = (1 + max_strain) ** 2 / max((1 - max_strain) ** 2, 1e-12)

    matches: list[InterfaceMatch] = []
    for fc in films:
        f_atoms = film_atoms_per_surface_cell * fc.determinant
        if f_atoms > max_atoms:
            continue

        for sc in subs:
            total_atoms = f_atoms + substrate_atoms_per_surface_cell * sc.determinant
            if total_atoms > max_atoms:
                continue

            ratio = max(fc.area, sc.area) / min(fc.area, sc.area)
            if ratio > area_ratio_limit * 1.02:
                continue

            # Remove the rigid epitaxial rotation before evaluating coherent strain.
            # Row-vector rotation matrix: v_rot = v @ R.
            rotation = _angle_difference_deg(fc.angle_deg, sc.angle_deg)
            th = np.deg2rad(rotation)
            c, s = float(np.cos(th)), float(np.sin(th))
            r_row = np.array([[c, s], [-s, c]], dtype=float)
            film_aligned = fc.basis @ r_row

            try:
                _, e, principal = _strain_for_mapping(film_aligned, sc.basis)
            except np.linalg.LinAlgError:
                continue

            max_abs = float(np.max(np.abs(principal)))
            if max_abs > max_strain + 1e-12:
                continue

            # Green-Lagrange components are reported after removing rigid rotation.
            eps_xx = float(e[0, 0])
            eps_yy = float(e[1, 1])
            eps_xy = float(e[0, 1])
            area_mismatch = (sc.area - fc.area) / fc.area
            target_area = sc.area

            # Primary goal: low strain; secondary goals: compact cell and fewer atoms.
            score = max_abs + 2e-4 * target_area + 2e-5 * total_atoms + 0.05 * abs(eps_xy)

            matches.append(
                InterfaceMatch(
                    film_matrix=fc.matrix.copy(),
                    substrate_matrix=sc.matrix.copy(),
                    film_basis=fc.basis.copy(),
                    substrate_basis=sc.basis.copy(),
                    film_area=fc.area,
                    substrate_area=sc.area,
                    target_area=target_area,
                    film_det=fc.determinant,
                    substrate_det=sc.determinant,
                    atom_count=total_atoms,
                    rotation_deg=rotation,
                    eps_xx=eps_xx,
                    eps_yy=eps_yy,
                    eps_xy=eps_xy,
                    principal_1=float(principal[0]),
                    principal_2=float(principal[1]),
                    max_abs_strain=max_abs,
                    area_mismatch=float(area_mismatch),
                    score=float(score),
                )
            )

    matches.sort(key=lambda m: (m.score, m.max_abs_strain, m.target_area, m.atom_count))
    return matches[:max_results]


def matrix_text(m: np.ndarray) -> str:
    m = np.asarray(m, dtype=int)
    return f"[[{m[0,0]}, {m[0,1]}], [{m[1,0]}, {m[1,1]}]]"
