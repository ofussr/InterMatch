import numpy as np

from app.structure_view import interface_layer_indices


class _Lattice:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float)


class _Interface:
    def __init__(self, z, film_indices, substrate_indices, c_sign=1.0):
        self.cart_coords = np.column_stack((np.zeros(len(z)), np.zeros(len(z)), np.asarray(z, dtype=float)))
        self.film_indices = list(film_indices)
        self.substrate_indices = list(substrate_indices)
        self.lattice = _Lattice([[5, 0, 0], [0, 5, 0], [0, 0, 30 * c_sign]])

    def __len__(self):
        return len(self.cart_coords)


def test_layer_limit_selects_layers_nearest_interface():
    # Substrate layers at z=0,1,2 and film layers at z=5,6,7.
    interface = _Interface(
        [0.0, 0.05, 1.0, 1.05, 2.0, 2.05, 5.0, 5.05, 6.0, 6.05, 7.0, 7.05],
        film_indices=range(6, 12),
        substrate_indices=range(0, 6),
    )
    selected = interface_layer_indices(interface, 1, tolerance=0.2)
    assert selected == [4, 5, 6, 7]


def test_layer_limit_selects_two_layers_per_side():
    interface = _Interface(
        [0.0, 0.05, 1.0, 1.05, 2.0, 2.05, 5.0, 5.05, 6.0, 6.05, 7.0, 7.05],
        film_indices=range(6, 12),
        substrate_indices=range(0, 6),
    )
    selected = interface_layer_indices(interface, 2, tolerance=0.2)
    assert selected == [2, 3, 4, 5, 6, 7, 8, 9]


def test_layer_limit_handles_reversed_film_substrate_order():
    # Film is now below substrate; nearest layers are still chosen by relative means.
    interface = _Interface(
        [0.0, 0.05, 1.0, 1.05, 2.0, 2.05, 5.0, 5.05, 6.0, 6.05, 7.0, 7.05],
        film_indices=range(0, 6),
        substrate_indices=range(6, 12),
    )
    selected = interface_layer_indices(interface, 1, tolerance=0.2)
    assert selected == [4, 5, 6, 7]


def test_layer_limit_is_independent_of_c_vector_sign():
    interface = _Interface(
        [0.0, 0.05, 1.0, 1.05, 2.0, 2.05, 5.0, 5.05, 6.0, 6.05, 7.0, 7.05],
        film_indices=range(6, 12),
        substrate_indices=range(0, 6),
        c_sign=-1.0,
    )
    selected = interface_layer_indices(interface, 1, tolerance=0.2)
    assert selected == [4, 5, 6, 7]
