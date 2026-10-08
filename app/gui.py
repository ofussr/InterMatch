from __future__ import annotations

from pathlib import Path
import traceback

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Import the OpenGL viewer before QApplication is constructed.  unit-cell-gui
# uses this import point to configure one shared OpenGL surface format.
from unit_cell_gui import DisplayOptions, UnitCellViewer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .crystal import build_surface, possible_terminations
from .interface_builder import build_interface, save_cif
from .matcher import find_matches, matrix_text
from .structure_view import (
    install_fast_interface_preview_profile,
    interface_layer_indices,
    scene_from_structure,
)
from .xrd import calculate_xrd


class PlotCanvas(FigureCanvas):
    def __init__(self, projection=None):
        self.figure = Figure(figsize=(5, 4), constrained_layout=True)
        self.ax = self.figure.add_subplot(111, projection=projection)
        super().__init__(self.figure)


class InterfaceMatcherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interface Matcher 0.2.3")
        self.resize(1450, 900)

        self.film_surface = None
        self.substrate_surface = None
        self.matches = []
        self.built = None

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        # Top: controls on the left, visualisation on the right.
        upper = QSplitter(Qt.Orientation.Horizontal)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        upper.addWidget(controls)

        controls_layout.addWidget(self._make_material_box("Film", True))
        controls_layout.addWidget(self._make_material_box("Substrate", False))
        controls_layout.addWidget(self._make_search_box())
        controls_layout.addWidget(self._make_preview_box())
        controls_layout.addStretch(1)

        self.tabs = QTabWidget()
        install_fast_interface_preview_profile()
        self.structure_view = UnitCellViewer()
        self.structure_view.set_display_options(
            DisplayOptions(
                show_atoms=True,
                show_external_atoms=True,
                show_bonds=False,
                show_cell=True,
                show_basis=True,
                show_polyhedra=False,
            )
        )
        self.match_preview = PlotCanvas(projection="3d")
        self.xrd_plot = PlotCanvas()
        self.tabs.addTab(self.structure_view, "Structure")
        self.tabs.addTab(self.match_preview, "Match geometry")
        self.tabs.addTab(self.xrd_plot, "XRD")
        upper.addWidget(self.tabs)
        upper.setSizes([420, 1000])

        # Bottom: the candidate table now occupies the old preview position.
        results = QWidget()
        results_layout = QVBoxLayout(results)
        results_layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(
            ["#", "Area Å²", "Atoms", "Rot. °", "εmax %", "εxx %", "εyy %", "εxy %", "Film M", "Sub M", "Score"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        results_layout.addWidget(self.table, 1)

        button_row = QHBoxLayout()
        self.build_btn = QPushButton("Build interface")
        self.save_btn = QPushButton("Save CIF")
        self.xrd_btn = QPushButton("Calculate XRD")
        self.build_btn.clicked.connect(self._build_interface)
        self.save_btn.clicked.connect(self._save_cif)
        self.xrd_btn.clicked.connect(self._calculate_xrd)
        for button in (self.build_btn, self.save_btn, self.xrd_btn):
            button.setEnabled(False)
            button_row.addWidget(button)
        button_row.addStretch(1)
        results_layout.addLayout(button_row)

        vertical = QSplitter(Qt.Orientation.Vertical)
        vertical.addWidget(upper)
        vertical.addWidget(results)
        vertical.setStretchFactor(0, 3)
        vertical.setStretchFactor(1, 2)
        vertical.setSizes([590, 300])
        root_layout.addWidget(vertical, 1)

        self.statusBar().showMessage("Load two CIF files to start")

    def _make_material_box(self, title: str, film: bool):
        box = QGroupBox(title)
        grid = QGridLayout(box)

        path_edit = QLineEdit()
        browse = QPushButton("Browse…")
        h, k, l = QSpinBox(), QSpinBox(), QSpinBox()
        for spin in (h, k, l):
            spin.setRange(-20, 20)
        l.setValue(1)

        thickness = QDoubleSpinBox()
        thickness.setRange(2.0, 100.0)
        thickness.setValue(15.0)
        thickness.setSuffix(" Å")

        termination = QComboBox()
        termination.setEditable(True)
        termination.addItem("auto")

        grid.addWidget(QLabel("CIF"), 0, 0)
        grid.addWidget(path_edit, 0, 1, 1, 3)
        grid.addWidget(browse, 0, 4)
        grid.addWidget(QLabel("Plane (h k l)"), 1, 0)
        grid.addWidget(h, 1, 1)
        grid.addWidget(k, 1, 2)
        grid.addWidget(l, 1, 3)
        grid.addWidget(QLabel("Slab thickness"), 2, 0)
        grid.addWidget(thickness, 2, 1, 1, 2)
        grid.addWidget(QLabel("Termination shift"), 3, 0)
        grid.addWidget(termination, 3, 1, 1, 2)
        term_btn = QPushButton("Find terminations")
        grid.addWidget(term_btn, 3, 3, 1, 2)

        attrs = {
            "film": film,
            "path": path_edit,
            "h": h,
            "k": k,
            "l": l,
            "thickness": thickness,
            "termination": termination,
        }
        if film:
            self.film_ui = attrs
        else:
            self.substrate_ui = attrs

        browse.clicked.connect(lambda: self._browse(path_edit))
        term_btn.clicked.connect(lambda: self._fill_terminations(attrs))
        return box

    def _make_search_box(self):
        box = QGroupBox("Search")
        form = QFormLayout(box)

        self.max_strain = QDoubleSpinBox()
        self.max_strain.setRange(0.0, 20.0)
        self.max_strain.setDecimals(2)
        self.max_strain.setValue(3.0)
        self.max_strain.setSuffix(" %")

        self.max_area = QDoubleSpinBox()
        self.max_area.setRange(1.0, 5000.0)
        self.max_area.setValue(300.0)
        self.max_area.setSuffix(" Å²")

        self.max_atoms = QSpinBox()
        self.max_atoms.setRange(2, 10000)
        self.max_atoms.setValue(300)

        self.gap = QDoubleSpinBox()
        self.gap.setRange(0.5, 10.0)
        self.gap.setValue(2.0)
        self.gap.setSuffix(" Å")

        self.vacuum = QDoubleSpinBox()
        self.vacuum.setRange(0.0, 100.0)
        self.vacuum.setValue(15.0)
        self.vacuum.setSuffix(" Å")

        self.offset_x = QDoubleSpinBox()
        self.offset_x.setRange(0.0, 0.999)
        self.offset_x.setDecimals(3)
        self.offset_y = QDoubleSpinBox()
        self.offset_y.setRange(0.0, 0.999)
        self.offset_y.setDecimals(3)
        offset = QWidget()
        offset_layout = QHBoxLayout(offset)
        offset_layout.setContentsMargins(0, 0, 0, 0)
        offset_layout.addWidget(QLabel("x"))
        offset_layout.addWidget(self.offset_x)
        offset_layout.addWidget(QLabel("y"))
        offset_layout.addWidget(self.offset_y)

        self.find_btn = QPushButton("Find matches")
        self.find_btn.clicked.connect(self._find_matches)

        form.addRow("Max strain", self.max_strain)
        form.addRow("Max area", self.max_area)
        form.addRow("Max atoms", self.max_atoms)
        form.addRow("Interface gap", self.gap)
        form.addRow("Vacuum", self.vacuum)
        form.addRow("In-plane offset", offset)
        form.addRow(self.find_btn)
        return box

    def _make_preview_box(self):
        box = QGroupBox("Structure preview")
        form = QFormLayout(box)

        self.limit_layers = QCheckBox("Limit to layers nearest interface")
        self.limit_layers.setChecked(True)

        self.preview_layers = QSpinBox()
        self.preview_layers.setRange(1, 100)
        self.preview_layers.setValue(3)
        self.preview_layers.setEnabled(True)
        self.preview_layers.setSuffix(" / side")

        self.preview_apply_btn = QPushButton("Apply")
        self.preview_apply_btn.setEnabled(False)
        self.preview_apply_btn.clicked.connect(self._apply_preview_settings)

        layer_row = QWidget()
        layer_layout = QHBoxLayout(layer_row)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        layer_layout.addWidget(self.preview_layers, 1)
        layer_layout.addWidget(self.preview_apply_btn)

        self.preview_info = QLabel("No interface built")
        self.preview_info.setWordWrap(True)

        tooltip = (
            "Display only N atomic layers nearest the interface in each material. "
            "Press Apply after changing this setting. The built structure, saved "
            "CIF and XRD calculation remain unchanged."
        )
        self.limit_layers.setToolTip(tooltip)
        self.preview_layers.setToolTip(tooltip)
        self.preview_apply_btn.setToolTip(tooltip)

        self.limit_layers.toggled.connect(self._preview_settings_changed)
        self.preview_layers.valueChanged.connect(self._preview_settings_changed)

        form.addRow(self.limit_layers)
        form.addRow("Layers", layer_row)
        form.addRow(self.preview_info)
        return box

    def _preview_settings_changed(self, *_args):
        self.preview_layers.setEnabled(self.limit_layers.isChecked())
        self.preview_apply_btn.setEnabled(self.built is not None)

    def _apply_preview_settings(self):
        self._refresh_structure_preview()
        self.preview_apply_btn.setEnabled(False)

    def _browse(self, edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Open CIF", "", "CIF files (*.cif);;All files (*)")
        if path:
            edit.setText(path)

    @staticmethod
    def _hkl(ui):
        return (ui["h"].value(), ui["k"].value(), ui["l"].value())

    @staticmethod
    def _termination_value(ui):
        text = ui["termination"].currentText().strip().lower()
        if not text or text == "auto":
            return None
        return float(text)

    def _fill_terminations(self, ui):
        try:
            shifts = possible_terminations(
                ui["path"].text(), self._hkl(ui), slab_thickness=ui["thickness"].value()
            )
            ui["termination"].clear()
            for shift in shifts:
                ui["termination"].addItem(f"{shift:.8f}")
            self.statusBar().showMessage(f"Found {len(shifts)} possible termination(s)")
        except Exception as exc:
            self._error(exc)

    def _find_matches(self):
        try:
            self.statusBar().showMessage("Building surfaces…")
            QApplication.processEvents()
            self.film_surface = build_surface(
                self.film_ui["path"].text(),
                self._hkl(self.film_ui),
                slab_thickness=self.film_ui["thickness"].value(),
                termination_shift=self._termination_value(self.film_ui),
            )
            self.substrate_surface = build_surface(
                self.substrate_ui["path"].text(),
                self._hkl(self.substrate_ui),
                slab_thickness=self.substrate_ui["thickness"].value(),
                termination_shift=self._termination_value(self.substrate_ui),
            )

            self.statusBar().showMessage("Searching HNF supercells…")
            QApplication.processEvents()
            self.matches = find_matches(
                self.film_surface.basis_2d,
                self.substrate_surface.basis_2d,
                max_strain=self.max_strain.value() / 100.0,
                max_area=self.max_area.value(),
                max_atoms=self.max_atoms.value(),
                film_atoms_per_surface_cell=len(self.film_surface.slab),
                substrate_atoms_per_surface_cell=len(self.substrate_surface.slab),
            )
            self._fill_table()
            self.built = None
            self.save_btn.setEnabled(False)
            self.xrd_btn.setEnabled(False)
            self.build_btn.setEnabled(bool(self.matches))
            self.statusBar().showMessage(f"Found {len(self.matches)} match(es)")
        except Exception as exc:
            self._error(exc)

    def _fill_table(self):
        self.table.setRowCount(len(self.matches))
        for row, match in enumerate(self.matches):
            values = [
                str(row + 1),
                f"{match.target_area:.3f}",
                str(match.atom_count),
                f"{match.rotation_deg:.3f}",
                f"{100 * match.max_abs_strain:.3f}",
                f"{100 * match.eps_xx:.3f}",
                f"{100 * match.eps_yy:.3f}",
                f"{100 * match.eps_xy:.3f}",
                matrix_text(match.film_matrix),
                matrix_text(match.substrate_matrix),
                f"{match.score:.6f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in (0, 1, 2, 3, 4, 5, 6, 7, 10):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, col, item)
        if self.matches:
            self.table.selectRow(0)

    def _selected_match(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        index = rows[0].row()
        return self.matches[index] if 0 <= index < len(self.matches) else None

    def _selection_changed(self):
        match = self._selected_match()
        self.build_btn.setEnabled(match is not None)
        if match is not None:
            self._plot_lattice_match(match)

    def _plot_lattice_match(self, match):
        ax = self.match_preview.ax
        ax.clear()
        for basis, z, label in (
            (match.substrate_basis, 0.0, "substrate"),
            (match.film_basis, 1.5, "film (unstrained)"),
        ):
            points = np.array([[0, 0], basis[0], basis[0] + basis[1], basis[1], [0, 0]], dtype=float)
            ax.plot(points[:, 0], points[:, 1], np.full(len(points), z), label=label)
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_zlabel("preview")
        ax.legend()
        self.match_preview.draw_idle()
        self.tabs.setCurrentWidget(self.match_preview)

    def _build_interface(self):
        match = self._selected_match()
        if match is None:
            return
        try:
            self.built = build_interface(
                self.film_surface,
                self.substrate_surface,
                match,
                gap=self.gap.value(),
                vacuum=self.vacuum.value(),
                offset=(self.offset_x.value(), self.offset_y.value()),
            )
            self._refresh_structure_preview()
            self.preview_apply_btn.setEnabled(False)
            self.save_btn.setEnabled(True)
            self.xrd_btn.setEnabled(True)
            self.tabs.setCurrentWidget(self.structure_view)
            self.statusBar().showMessage(f"Interface built: {len(self.built.interface)} atoms")
        except Exception as exc:
            self._error(exc)

    def _refresh_structure_preview(self):
        if self.built is None:
            return
        try:
            indices = None
            if self.limit_layers.isChecked():
                indices = interface_layer_indices(
                    self.built.interface,
                    self.preview_layers.value(),
                )
            scene = scene_from_structure(self.built.interface, indices)
            self.structure_view.set_scene(scene)
            shown = len(scene.atoms)
            total = len(self.built.interface)
            if self.limit_layers.isChecked():
                self.preview_info.setText(
                    f"Showing {shown} / {total} atoms "
                    f"({self.preview_layers.value()} layer(s) per side)"
                )
            else:
                self.preview_info.setText(
                    f"Showing all {shown} atoms. Large supercells may render more slowly."
                )
        except Exception as exc:
            self._error(exc)

    def _save_cif(self):
        if self.built is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save interface CIF", "interface.cif", "CIF files (*.cif)")
        if not path:
            return
        try:
            save_cif(self.built.interface, path)
            self.statusBar().showMessage(f"Saved {Path(path).name}")
        except Exception as exc:
            self._error(exc)

    def _calculate_xrd(self):
        if self.built is None:
            return
        try:
            pattern = calculate_xrd(self.built.interface, wavelength="CuKa", two_theta=(5, 90))
            ax = self.xrd_plot.ax
            ax.clear()
            ax.vlines(pattern.x, 0, pattern.y)
            ax.set_xlabel("2θ (°)")
            ax.set_ylabel("Relative intensity")
            ax.set_xlim(5, 90)
            ax.set_ylim(bottom=0)
            ax.set_title("Periodic interface supercell — Cu Kα")
            self.xrd_plot.draw_idle()
            self.tabs.setCurrentWidget(self.xrd_plot)
        except Exception as exc:
            self._error(exc)

    def _error(self, exc: Exception):
        traceback.print_exc()
        QMessageBox.critical(self, "Error", str(exc))
        self.statusBar().showMessage("Error")
