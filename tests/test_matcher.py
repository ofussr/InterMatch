import numpy as np

from app.matcher import as_2d_basis, find_matches, gauss_reduce_2d, generate_supercells, hnf_matrices


def test_as_2d_basis_preserves_metric():
    a = np.array([2.0, 0.0, 0.0])
    b = np.array([1.0, 3.0, 0.0])
    basis = as_2d_basis(a, b)
    assert np.allclose(basis @ basis.T, np.array([[4.0, 2.0], [2.0, 10.0]]))


def test_gauss_reduction_preserves_lattice_area():
    basis = np.array([[4.0, 0.0], [3.9, 1.0]])
    reduced, u = gauss_reduce_2d(basis)
    assert abs(round(np.linalg.det(u))) == 1
    assert np.isclose(abs(np.linalg.det(reduced)), abs(np.linalg.det(basis)))
    assert np.linalg.norm(reduced[0]) <= np.linalg.norm(reduced[1]) + 1e-12


def test_identical_square_lattices_have_zero_strain_match():
    b = np.array([[4.0, 0.0], [0.0, 4.0]])
    matches = find_matches(b, b, max_strain=1e-8, max_area=32, max_atoms=50)
    assert matches
    assert matches[0].max_abs_strain < 1e-10


def test_two_by_two_vs_three_by_three_commensurate_match():
    film = np.array([[3.0, 0.0], [0.0, 3.0]])
    sub = np.array([[2.0, 0.0], [0.0, 2.0]])
    matches = find_matches(film, sub, max_strain=1e-8, max_area=40, max_atoms=100)
    assert matches
    assert any(m.film_det == 4 and m.substrate_det == 9 and m.max_abs_strain < 1e-10 for m in matches)


def test_supercells_respect_area_limit():
    b = np.array([[2.0, 0.0], [0.0, 3.0]])
    cells = generate_supercells(b, 18.0)
    assert cells
    assert max(c.area for c in cells) <= 18.0 + 1e-9


def test_rigid_rotation_is_not_counted_as_strain():
    film = np.array([[2.0, 0.0], [0.0, 3.0]])
    th = np.deg2rad(30.0)
    r = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    sub = film @ r
    matches = find_matches(film, sub, max_strain=1e-8, max_area=6.1, max_atoms=10)
    assert matches
    assert matches[0].max_abs_strain < 1e-10


def test_gauss_reduction_half_projection_does_not_cycle():
    # Regression for square-surface HNF cells such as [[1, 1], [0, 9]].
    # The old implementation alternated forever between two equivalent bases
    # when the projection landed numerically on +/-1/2.
    a = 5.653
    basis = np.array([[a, a], [0.0, 9.0 * a]])
    reduced, u = gauss_reduce_2d(basis)
    assert abs(round(np.linalg.det(u))) == 1
    assert np.isclose(abs(np.linalg.det(reduced)), abs(np.linalg.det(basis)))
    projection = float(np.dot(reduced[0], reduced[1]) / np.dot(reduced[0], reduced[0]))
    assert abs(projection) <= 0.5 + 1e-10


def test_square_supercell_generation_handles_large_hnf_set():
    # CdTe/GaAs-like cubic surfaces should enumerate normally rather than hit
    # the half-integer Gauss-reduction cycle.
    a = 5.653
    basis = np.array([[a, 0.0], [0.0, a]])
    cells = generate_supercells(basis, 12.0 * a * a)
    assert cells
    assert all(np.isfinite(cell.area) and cell.area > 0 for cell in cells)


def test_gauss_reduction_many_square_hnf_cells_converge():
    # Stress the exact family that exposed the bug.
    a = 6.48
    parent = np.array([[a, 0.0], [0.0, a]])
    for h in hnf_matrices(40):
        reduced, u = gauss_reduce_2d(h @ parent)
        assert abs(round(np.linalg.det(u))) == 1
        assert np.isclose(abs(np.linalg.det(reduced)), abs(np.linalg.det(h @ parent)))


def test_cdte_111_gaas_100_reference_geometry_has_sub_one_percent_match():
    # Geometric benchmark using standard cubic lattice constants.  Primitive
    # fcc(111) is triangular with a/sqrt(2); primitive fcc(100) is square with
    # a/sqrt(2).  The known CdTe(111)/GaAs(100) coincidence family contains a
    # sub-1% direction/2D supercell match at modest area.
    import math

    a_cdte = 6.482
    a_gaas = 5.6533
    l_cdte = a_cdte / math.sqrt(2.0)
    l_gaas = a_gaas / math.sqrt(2.0)
    film = np.array(
        [[l_cdte, 0.0], [0.5 * l_cdte, 0.5 * math.sqrt(3.0) * l_cdte]]
    )
    substrate = np.array([[l_gaas, 0.0], [0.0, l_gaas]])

    matches = find_matches(
        film,
        substrate,
        max_strain=0.03,
        max_area=300.0,
        max_atoms=1000,
        max_results=20,
    )
    assert matches
    assert min(match.max_abs_strain for match in matches) < 0.01
    assert min(match.target_area for match in matches) < 150.0
