from __future__ import annotations


def calculate_xrd(structure, wavelength: str = "CuKa", two_theta=(5.0, 90.0)):
    try:
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
    except ImportError as exc:
        raise RuntimeError("pymatgen is required for XRD calculation") from exc

    calc = XRDCalculator(wavelength=wavelength, symprec=0)
    return calc.get_pattern(structure, scaled=True, two_theta_range=tuple(map(float, two_theta)))
