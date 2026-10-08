from types import SimpleNamespace

from app.crystal import _possible_termination_shifts


def test_termination_shifts_new_public_api():
    class Generator:
        def gen_possible_terminations(self):
            return [0.125, 0.625]

    assert _possible_termination_shifts(Generator()) == [0.125, 0.625]


def test_termination_shifts_legacy_private_api():
    class Generator:
        def _calculate_possible_shifts(self, tol=0.1):
            assert tol == 0.1
            return [0.2, 0.7]

    assert _possible_termination_shifts(Generator()) == [0.2, 0.7]


def test_termination_shifts_old_get_slabs_fallback():
    class Generator:
        def get_slabs(self, filter_out_sym_slabs=True):
            assert filter_out_sym_slabs is True
            return [SimpleNamespace(shift=0.15), SimpleNamespace(shift=0.65)]

    assert _possible_termination_shifts(Generator()) == [0.15, 0.65]


def test_termination_shifts_very_old_get_slabs_signature():
    class Generator:
        def get_slabs(self):
            return [SimpleNamespace(shift=0.3), SimpleNamespace(shift=0.8)]

    assert _possible_termination_shifts(Generator()) == [0.3, 0.8]


def test_termination_shifts_deduplicate_near_equal_values():
    class Generator:
        def gen_possible_terminations(self):
            return [0.1, 0.1 + 1e-12, 0.6]

    assert _possible_termination_shifts(Generator()) == [0.1, 0.6]


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
