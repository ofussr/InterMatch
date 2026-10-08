# Interface Matcher 0.2.4

Standalone PySide6 prototype for coherent film/substrate interface matching.

## What is implemented

- Load **film** and **substrate** CIF files.
- Set independent `(h k l)` surfaces.
- Generate slabs with a selected termination shift.
- Search 2D commensurate supercells with a **custom matcher**:
  - 2D Hermite Normal Form enumeration;
  - Gauss lattice reduction;
  - coherent deformation tensor;
  - principal strain filtering;
  - ranking by strain, area, shear and atom count.
- Show candidate area, atom count, rotation, strain components and integer transformation matrices.
- Build an atomistic interface from the selected match.
- Adjustable interface gap, vacuum and fractional in-plane registry offset.
- Interactive instanced OpenGL preview using **unit-cell-gui 0.2.3**.
- Background **Find terminations** (window stays responsive while the oriented
  cell is prepared); older pymatgen versions do not create and compare complete
  slabs merely to enumerate candidate shifts.
- Optional preview limit to **N atomic layers per material nearest the interface**. This affects only rendering, not the built structure.
- Export the complete resulting structure to CIF.
- Calculate a Cu Kα XRD stick pattern of the complete periodic interface supercell.

The matching algorithm is in `app/matcher.py` and does **not** call pymatgen's `ZSLGenerator`.
Pymatgen is used for CIF parsing, slab construction, `Interface.from_slabs()` and XRD atomic scattering factors.

## Install / update dependencies

Python 3.11+ is recommended.

```powershell
py -m venv .venv
.venv\Scripts\activate
py -m pip install -U pip
pip install -r requirements.txt
```

If you are updating an existing Interface Matcher virtual environment from 0.1.x,
install the new viewer dependency once:

```powershell
pip install "unit-cell-gui>=0.2.3,<0.3"
```

## Run

```powershell
py main.py
```

## Layout and workflow

The main layout is now:

- **left/top** — material and search controls;
- **right/top** — visualisation tabs (`Structure`, `Match geometry`, `XRD`);
- **bottom** — candidate-match table and build/export buttons.

Workflow:

1. Choose film CIF and substrate CIF.
2. Set `(h k l)` for both.
3. Leave termination as **auto**, or press **Find terminations** and choose a specific shift.
4. Set maximum strain, maximum interface area and maximum atom count.
5. Press **Find matches**.
6. Select a row. `Match geometry` shows the two matched 2D cells.
7. Press **Build interface**. The `Structure` tab displays the atomistic interface in `unit-cell-gui`.
8. Optionally enable **Limit to layers nearest interface** and select the number of atomic layers to display on each side.
9. Change in-plane offset `(x, y)` and rebuild to inspect different registries.
10. Save the complete CIF or calculate XRD.

### Layer-limited preview

The layer checkbox is deliberately **display-only**. It uses pymatgen `Interface.film_indices`
and `Interface.substrate_indices`, groups atomic positions into planes along the interface c-axis,
and keeps the requested number of planes nearest the interface from each material.

It does **not** remove atoms from:

- the built `Interface` object;
- the saved CIF;
- the XRD calculation.

## unit-cell-gui integration

`unit-cell-gui` is used only as the OpenGL renderer, in the way its public API is designed.
`app/structure_view.py` converts the already-built pymatgen structure into a `unit_cell_gui.Scene`.
The first integration renders atoms, unit-cell edges and basis vectors. Bond and polyhedron inference
are intentionally left outside the renderer and can be added later without changing the matching engine.

Version 0.2.4 removes the old monkeypatch of `unit_cell_gui.opengl_viewer.sphere_mesh`.
The newer viewer uses GPU-instanced atom meshes, so patching its private rendering
internals is both ineffective and fragile. Interface Matcher now relies only on
public renderer APIs (`UnitCellViewer`, `Scene`, `DisplayOptions`).

**CIF parsing warnings** about fractional coordinates rounded to ideal values
are not fatal. A search that genuinely fails will show an error dialog, while a
running `Find terminations` job reports its progress in the status bar.

### 0.2.4 fixes

- Run termination enumeration in a `QThread` instead of blocking the main GUI.
- Avoid the historical `get_slabs()` fallback, which could hang on structure
  grouping for large/slightly imperfect CIF files. On older pymatgen versions,
  shifts are derived directly from the oriented unit cell and periodic layers.
- Keep the newer `unit-cell-gui` GPU instancing and remove renderer monkeypatching.

## Important physical limitation

Version 0.2 is a **geometric coherent-interface builder**, not an energetic interface optimiser.
It does not decide which termination or lateral registry is physically stable, and it does not relax atomic positions.
For a physical interface-energy calculation, export candidate cells to a DFT / ML-potential workflow and relax them.

The displayed XRD is the kinematic powder-style pattern of the *periodically repeated interface supercell*.
It is useful for structural inspection, but it is not yet a thin-film θ–2θ/RSM simulator.

## Project layout

- `main.py` — launcher
- `app/gui.py` — PySide6 GUI
- `app/matcher.py` — custom HNF/reduction/strain matching engine
- `app/crystal.py` — CIF and slab handling
- `app/interface_builder.py` — atomistic interface construction/export
- `app/structure_view.py` — pymatgen → unit-cell-gui adapter and layer-limited preview
- `app/xrd.py` — XRD calculation
- `tests/test_matcher.py` — pure-numpy matcher tests
- `tests/test_structure_view.py` — layer-selection tests

## Pymatgen compatibility

Interface Matcher does not require the newer public `SlabGenerator.gen_possible_terminations()` method.
It automatically falls back to the older termination APIs used by pre-2026.8.13 pymatgen releases.


## 0.2.1 preview changes

- Layer-limited preview is enabled by default at 3 layers per side.
- Preview settings are applied explicitly with the **Apply** button; changing the spin box no longer rebuilds the OpenGL scene on every click.
- Interface previews use a lighter atom-sphere tessellation with unit-cell-gui to keep large supercells responsive. This affects display only, not the built structure, CIF export, or XRD calculation.


## 0.2.2 matcher fix

- Fixed a two-state cycle in 2D Gauss/Lagrange reduction at the exact half-projection boundary (`|mu| = 1/2`).
- The reduced condition is now tested explicitly and equal-length/equivalent bases are handled deterministically.
- Added regression coverage for square cubic surfaces and large HNF enumeration, including the family that is triggered by CdTe/GaAs-style tests.


## 0.2.3 primitive surface-cell fix

Surface slabs are now reduced to their primitive cells before lattice matching (`SlabGenerator(..., primitive=True)`). This is important for centered lattices such as zinc-blende CdTe and GaAs: using the unreduced conventional slab cell can hide valid coincidence lattices below the chosen `Max area` limit because HNF enumeration can only build larger sublattices, never recover missing primitive surface translations.

For the CdTe(111)/GaAs(100) geometry, the primitive 2D benchmark gives a coincidence cell near 127.8 Å² with about 0.7% principal strain for standard room-temperature lattice constants; the unreduced surface representation can push the same geometric family to roughly four times the area.
