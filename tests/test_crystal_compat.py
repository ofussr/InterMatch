from types import SimpleNamespace
import numpy as np
import pytest

from app.crystal import _possible_termination_shifts


def fake_generator(positions, height=10.0):
    coordinates = np.zeros((len(positions), 3))
    coordinates[:, 2] = positions

    class Generator:
        oriented_unit_cell = SimpleNamespace(frac_coords=coordinates)
        _proj_height = height

        def get_slabs(self, *args, **kwargs):
            raise AssertionError("get_slabs must never be called to list terminations")

    return Generator()


def test_termination_shifts_new_public_api():
    class Generator:
        def gen_possible_terminations(self):
            return [0.625, 0.125]

    assert _possible_termination_shifts(Generator()) == [0.125, 0.625]


def test_termination_shifts_legacy_fast_midplane_fallback():
    gen = fake_generator([0.0, 0.5])
    assert _possible_termination_shifts(gen) == pytest.approx([0.25, 0.75])


def test_termination_shifts_groups_nearly_coplanar_atoms():
    gen = fake_generator([0.0, 0.0001, 0.5, 0.5001])
    assert _possible_termination_shifts(gen) == pytest.approx([0.25005, 0.75005])


def test_termination_shifts_clusters_across_periodic_seam():
    gen = fake_generator([0.99, 0.01, 0.5], height=4.0)
    assert _possible_termination_shifts(gen) == pytest.approx([0.25, 0.75])


def test_termination_shifts_single_layer():
    gen = fake_generator([0.15])
    assert _possible_termination_shifts(gen) == pytest.approx([0.65])


def test_termination_shifts_empty_cell():
    gen = fake_generator([])
    assert _possible_termination_shifts(gen) == []


def test_termination_shifts_many_sites_without_slabs():
    gen = fake_generator(np.repeat(np.arange(16) / 16, 100), height=32.0)
    shifts = _possible_termination_shifts(gen)
    assert len(shifts) == 16
    assert shifts[0] == pytest.approx(1 / 32)
    assert shifts[-1] == pytest.approx(31 / 32)


def test_surface_generator_requests_primitive_slab(monkeypatch):
    import sys
    import types
    from app import crystal

    captured = {}

    class FakeSlabGenerator:
        def __init__(self, structure, hkl, **kwargs):
            captured.update(kwargs)

    fake_surface = types.ModuleType("pymatgen.core.surface")
    fake_surface.SlabGenerator = FakeSlabGenerator
    fake_core = types.ModuleType("pymatgen.core")
    fake_pkg = types.ModuleType("pymatgen")
    monkeypatch.setitem(sys.modules, "pymatgen", fake_pkg)
    monkeypatch.setitem(sys.modules, "pymatgen.core", fake_core)
    monkeypatch.setitem(sys.modules, "pymatgen.core.surface", fake_surface)

    crystal._make_slab_generator(object(), (1, 1, 1), 15.0)
    assert captured["primitive"] is True
