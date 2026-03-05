from __future__ import annotations

from pathlib import Path

from SyMBac.config_models import (
    SimulationCellSpec,
    SimulationGeometrySpec,
    SimulationPhysicsSpec,
    SimulationRuntimeSpec,
    SimulationSpec,
)
from SyMBac.napari.io.yaml_store import load_simulation_spec, model_from_yaml_text, model_to_yaml_text, save_model
from SyMBac.napari.workers.tasks import start_worker


def _default_simulation_spec() -> SimulationSpec:
    return SimulationSpec(
        geometry=SimulationGeometrySpec(
            trench_length=14.25,
            trench_width=1.45,
            pix_mic_conv=0.065,
            resize_amount=3,
        ),
        cell=SimulationCellSpec(
            cell_max_length=6.65,
            cell_width=1.1,
            max_length_std=0.0,
            width_std=0.0,
            lysis_p=0.01,
        ),
        physics=SimulationPhysicsSpec(gravity=0.0, phys_iters=15),
        runtime=SimulationRuntimeSpec(sim_length=500, save_dir="/tmp/symbac_napari_sim", substeps=100),
    )


class SimulationDock:
    def __init__(self, controller, layer_manager):
        from qtpy.QtWidgets import (
            QCheckBox,
            QDoubleSpinBox,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QPlainTextEdit,
            QScrollArea,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        self.controller = controller
        self.layer_manager = layer_manager

        self.widget = QWidget()
        root = QVBoxLayout(self.widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll)
        container = QWidget()
        layout = QVBoxLayout(container)
        scroll.setWidget(container)

        defaults = _default_simulation_spec()

        geometry_group = QGroupBox("Geometry")
        geometry_form = QFormLayout(geometry_group)
        self.trench_length = QDoubleSpinBox()
        self.trench_length.setRange(0.01, 1000)
        self.trench_length.setDecimals(3)
        self.trench_length.setValue(defaults.geometry.trench_length)
        geometry_form.addRow("trench_length", self.trench_length)

        self.trench_width = QDoubleSpinBox()
        self.trench_width.setRange(0.01, 1000)
        self.trench_width.setDecimals(3)
        self.trench_width.setValue(defaults.geometry.trench_width)
        geometry_form.addRow("trench_width", self.trench_width)

        self.pix_mic_conv = QDoubleSpinBox()
        self.pix_mic_conv.setRange(0.0001, 10)
        self.pix_mic_conv.setDecimals(5)
        self.pix_mic_conv.setValue(defaults.geometry.pix_mic_conv)
        geometry_form.addRow("pix_mic_conv", self.pix_mic_conv)

        self.resize_amount = QSpinBox()
        self.resize_amount.setRange(1, 100)
        self.resize_amount.setValue(defaults.geometry.resize_amount)
        geometry_form.addRow("resize_amount", self.resize_amount)
        layout.addWidget(geometry_group)

        cell_group = QGroupBox("Cell")
        cell_form = QFormLayout(cell_group)
        self.cell_max_length = QDoubleSpinBox()
        self.cell_max_length.setRange(0.01, 1000)
        self.cell_max_length.setDecimals(3)
        self.cell_max_length.setValue(defaults.cell.cell_max_length)
        cell_form.addRow("cell_max_length", self.cell_max_length)

        self.cell_width = QDoubleSpinBox()
        self.cell_width.setRange(0.01, 1000)
        self.cell_width.setDecimals(3)
        self.cell_width.setValue(defaults.cell.cell_width)
        cell_form.addRow("cell_width", self.cell_width)

        self.max_length_std = QDoubleSpinBox()
        self.max_length_std.setRange(0, 1000)
        self.max_length_std.setDecimals(3)
        self.max_length_std.setValue(defaults.cell.max_length_std)
        cell_form.addRow("max_length_std", self.max_length_std)

        self.width_std = QDoubleSpinBox()
        self.width_std.setRange(0, 1000)
        self.width_std.setDecimals(3)
        self.width_std.setValue(defaults.cell.width_std)
        cell_form.addRow("width_std", self.width_std)

        self.lysis_p = QDoubleSpinBox()
        self.lysis_p.setRange(0, 1)
        self.lysis_p.setDecimals(4)
        self.lysis_p.setValue(defaults.cell.lysis_p)
        cell_form.addRow("lysis_p", self.lysis_p)
        layout.addWidget(cell_group)

        physics_group = QGroupBox("Physics")
        physics_form = QFormLayout(physics_group)
        self.gravity = QDoubleSpinBox()
        self.gravity.setRange(-1000, 1000)
        self.gravity.setDecimals(3)
        self.gravity.setValue(defaults.physics.gravity)
        physics_form.addRow("gravity", self.gravity)

        self.phys_iters = QSpinBox()
        self.phys_iters.setRange(1, 100000)
        self.phys_iters.setValue(defaults.physics.phys_iters)
        physics_form.addRow("phys_iters", self.phys_iters)
        layout.addWidget(physics_group)

        runtime_group = QGroupBox("Runtime")
        runtime_form = QFormLayout(runtime_group)
        self.sim_length = QSpinBox()
        self.sim_length.setRange(1, 10_000_000)
        self.sim_length.setValue(defaults.runtime.sim_length)
        runtime_form.addRow("sim_length", self.sim_length)

        self.substeps = QSpinBox()
        self.substeps.setRange(1, 10_000)
        self.substeps.setValue(defaults.runtime.substeps)
        runtime_form.addRow("substeps", self.substeps)

        self.save_dir = QLineEdit(defaults.runtime.save_dir)
        runtime_form.addRow("save_dir", self.save_dir)
        layout.addWidget(runtime_group)

        toolbar = QHBoxLayout()
        self.apply_form_button = QPushButton("Apply Form")
        self.sync_yaml_button = QPushButton("Sync YAML -> Form")
        toolbar.addWidget(self.apply_form_button)
        toolbar.addWidget(self.sync_yaml_button)
        layout.addLayout(toolbar)

        self.advanced_group = QGroupBox("Advanced: SimulationSpec YAML")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        advanced_layout = QVBoxLayout(self.advanced_group)

        self.spec_editor = QPlainTextEdit()
        self.spec_editor.setPlainText(model_to_yaml_text(defaults))
        advanced_layout.addWidget(self.spec_editor)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Spec path"))
        self.path_edit = QLineEdit("/tmp/symbac_simulation_spec.yaml")
        path_row.addWidget(self.path_edit)
        advanced_layout.addLayout(path_row)

        file_row = QHBoxLayout()
        self.load_button = QPushButton("Load YAML")
        self.save_button = QPushButton("Save YAML")
        self.reset_button = QPushButton("Reset Example")
        file_row.addWidget(self.load_button)
        file_row.addWidget(self.save_button)
        file_row.addWidget(self.reset_button)
        advanced_layout.addLayout(file_row)

        layout.addWidget(self.advanced_group)

        run_row = QHBoxLayout()
        self.show_window_checkbox = QCheckBox("show_window")
        self.show_window_checkbox.setChecked(False)
        self.run_button = QPushButton("Run Simulation")
        self.draw_button = QPushButton("Draw OPL")
        run_row.addWidget(self.show_window_checkbox)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.draw_button)
        layout.addLayout(run_row)
        layout.addStretch(1)

        self.apply_form_button.clicked.connect(self._apply_form)
        self.sync_yaml_button.clicked.connect(self._sync_yaml_to_form)
        self.load_button.clicked.connect(self._load_yaml)
        self.save_button.clicked.connect(self._save_yaml)
        self.reset_button.clicked.connect(self._reset_example)
        self.run_button.clicked.connect(self._run_simulation)
        self.draw_button.clicked.connect(self._draw_opl)

    def _spec_from_form(self) -> SimulationSpec:
        return SimulationSpec(
            geometry=SimulationGeometrySpec(
                trench_length=float(self.trench_length.value()),
                trench_width=float(self.trench_width.value()),
                pix_mic_conv=float(self.pix_mic_conv.value()),
                resize_amount=int(self.resize_amount.value()),
            ),
            cell=SimulationCellSpec(
                cell_max_length=float(self.cell_max_length.value()),
                cell_width=float(self.cell_width.value()),
                max_length_std=float(self.max_length_std.value()),
                width_std=float(self.width_std.value()),
                lysis_p=float(self.lysis_p.value()),
            ),
            physics=SimulationPhysicsSpec(
                gravity=float(self.gravity.value()),
                phys_iters=int(self.phys_iters.value()),
            ),
            runtime=SimulationRuntimeSpec(
                sim_length=int(self.sim_length.value()),
                substeps=int(self.substeps.value()),
                save_dir=self.save_dir.text().strip(),
            ),
        )

    def _set_form_from_spec(self, spec: SimulationSpec) -> None:
        self.trench_length.setValue(float(spec.geometry.trench_length))
        self.trench_width.setValue(float(spec.geometry.trench_width))
        self.pix_mic_conv.setValue(float(spec.geometry.pix_mic_conv))
        self.resize_amount.setValue(int(spec.geometry.resize_amount))

        self.cell_max_length.setValue(float(spec.cell.cell_max_length))
        self.cell_width.setValue(float(spec.cell.cell_width))
        self.max_length_std.setValue(float(spec.cell.max_length_std))
        self.width_std.setValue(float(spec.cell.width_std))
        self.lysis_p.setValue(float(spec.cell.lysis_p))

        self.gravity.setValue(float(spec.physics.gravity))
        self.phys_iters.setValue(int(spec.physics.phys_iters))

        self.sim_length.setValue(int(spec.runtime.sim_length))
        self.substeps.setValue(int(spec.runtime.substeps))
        self.save_dir.setText(spec.runtime.save_dir)

    def _spec_from_editor(self) -> SimulationSpec:
        return model_from_yaml_text(SimulationSpec, self.spec_editor.toPlainText())

    def _current_spec(self) -> SimulationSpec:
        if self.advanced_group.isChecked():
            spec = self._spec_from_editor()
        else:
            spec = self._spec_from_form()
        self.controller.set_simulation_spec(spec)
        return spec

    def _apply_form(self):
        from napari.utils.notifications import show_error, show_info

        try:
            spec = self._spec_from_form()
        except Exception as exc:
            show_error(f"Invalid simulation form values: {exc}")
            return
        self.spec_editor.setPlainText(model_to_yaml_text(spec))
        self.controller.set_simulation_spec(spec)
        show_info("Applied form values to simulation spec.")

    def _sync_yaml_to_form(self):
        from napari.utils.notifications import show_error, show_info

        try:
            spec = self._spec_from_editor()
        except Exception as exc:
            show_error(f"Invalid SimulationSpec YAML: {exc}")
            return
        self._set_form_from_spec(spec)
        self.controller.set_simulation_spec(spec)
        show_info("Synced YAML into form.")

    def _load_yaml(self):
        from napari.utils.notifications import show_error, show_info

        try:
            spec = load_simulation_spec(Path(self.path_edit.text()))
        except Exception as exc:
            show_error(f"Failed to load SimulationSpec YAML: {exc}")
            return
        self.spec_editor.setPlainText(model_to_yaml_text(spec))
        self._set_form_from_spec(spec)
        self.controller.set_simulation_spec(spec)
        show_info("Loaded SimulationSpec YAML.")

    def _save_yaml(self):
        from napari.utils.notifications import show_error, show_info

        try:
            spec = self._current_spec()
            save_model(spec, Path(self.path_edit.text()))
        except Exception as exc:
            show_error(f"Failed to save SimulationSpec YAML: {exc}")
            return
        show_info("Saved SimulationSpec YAML.")

    def _reset_example(self):
        spec = _default_simulation_spec()
        self._set_form_from_spec(spec)
        self.spec_editor.setPlainText(model_to_yaml_text(spec))

    def _run_simulation(self):
        from napari.utils.notifications import show_error, show_info

        try:
            self._current_spec()
        except Exception as exc:
            show_error(f"Invalid simulation configuration: {exc}")
            return

        def _task():
            return self.controller.run_simulation(show_window=self.show_window_checkbox.isChecked())

        def _done(_simulation):
            show_info("Simulation completed.")

        start_worker(_task, on_return=_done)

    def _draw_opl(self):
        from napari.utils.notifications import show_error, show_info

        if self.controller.state.simulation is None:
            show_error("Run a simulation before drawing OPL scenes.")
            return

        def _task():
            return self.controller.draw_opl(do_transformation=False, label_masks=True)

        def _done(simulation):
            self.layer_manager.update_simulation_layers(simulation)
            show_info("OPL scenes rendered.")

        start_worker(_task, on_return=_done)
