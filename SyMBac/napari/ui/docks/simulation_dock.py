from __future__ import annotations

from pathlib import Path

from SyMBac.config_models import (
    SimulationCellSpec,
    SimulationGeometrySpec,
    SimulationPhysicsSpec,
    SimulationRuntimeSpec,
    SimulationSpec,
)
from SyMBac.napari.io.yaml_store import (
    load_simulation_spec,
    model_from_yaml_text,
    model_to_yaml_text,
    save_model,
)
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
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QPlainTextEdit,
            QVBoxLayout,
            QWidget,
        )

        self.controller = controller
        self.layer_manager = layer_manager

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)

        layout.addWidget(QLabel("SimulationSpec YAML"))
        self.spec_editor = QPlainTextEdit()
        self.spec_editor.setPlainText(model_to_yaml_text(_default_simulation_spec()))
        layout.addWidget(self.spec_editor)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Spec path"))
        self.path_edit = QLineEdit("/tmp/symbac_simulation_spec.yaml")
        path_row.addWidget(self.path_edit)
        layout.addLayout(path_row)

        file_row = QHBoxLayout()
        self.load_button = QPushButton("Load YAML")
        self.save_button = QPushButton("Save YAML")
        self.reset_button = QPushButton("Reset Example")
        file_row.addWidget(self.load_button)
        file_row.addWidget(self.save_button)
        file_row.addWidget(self.reset_button)
        layout.addLayout(file_row)

        run_row = QHBoxLayout()
        self.show_window_checkbox = QCheckBox("show_window")
        self.show_window_checkbox.setChecked(False)
        self.run_button = QPushButton("Run Simulation")
        self.draw_button = QPushButton("Draw OPL")
        run_row.addWidget(self.show_window_checkbox)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.draw_button)
        layout.addLayout(run_row)

        self.load_button.clicked.connect(self._load_yaml)
        self.save_button.clicked.connect(self._save_yaml)
        self.reset_button.clicked.connect(self._reset_example)
        self.run_button.clicked.connect(self._run_simulation)
        self.draw_button.clicked.connect(self._draw_opl)

    def _spec_from_editor(self) -> SimulationSpec:
        spec = model_from_yaml_text(SimulationSpec, self.spec_editor.toPlainText())
        self.controller.set_simulation_spec(spec)
        return spec

    def _load_yaml(self):
        from napari.utils.notifications import show_info

        spec = load_simulation_spec(Path(self.path_edit.text()))
        self.spec_editor.setPlainText(model_to_yaml_text(spec))
        self.controller.set_simulation_spec(spec)
        show_info("Loaded SimulationSpec YAML.")

    def _save_yaml(self):
        from napari.utils.notifications import show_info

        spec = self._spec_from_editor()
        save_model(spec, Path(self.path_edit.text()))
        show_info("Saved SimulationSpec YAML.")

    def _reset_example(self):
        self.spec_editor.setPlainText(model_to_yaml_text(_default_simulation_spec()))

    def _run_simulation(self):
        from napari.utils.notifications import show_info

        self._spec_from_editor()

        def _task():
            return self.controller.run_simulation(show_window=self.show_window_checkbox.isChecked())

        def _done(_simulation):
            show_info("Simulation completed.")

        start_worker(_task, on_return=_done)

    def _draw_opl(self):
        from napari.utils.notifications import show_info

        def _task():
            return self.controller.draw_opl(do_transformation=False, label_masks=True)

        def _done(simulation):
            self.layer_manager.update_simulation_layers(simulation)
            show_info("OPL scenes rendered.")

        start_worker(_task, on_return=_done)
