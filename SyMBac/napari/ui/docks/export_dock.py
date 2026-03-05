from __future__ import annotations

from pathlib import Path

from SyMBac.config_models import (
    DatasetOutputConfig,
    RandomDatasetPlan,
    RenderConfig,
    TimeseriesDatasetPlan,
)
from SyMBac.napari.io.variant_loader import default_variants_yaml, load_variants_yaml_text
from SyMBac.napari.io.yaml_store import (
    model_from_yaml_text,
    model_to_yaml_text,
    save_model,
)
from SyMBac.napari.workers.tasks import start_worker


def _default_random_plan() -> RandomDatasetPlan:
    return RandomDatasetPlan(
        burn_in=40,
        n_samples=500,
        sample_amount=0.1,
        randomise_hist_match=True,
        randomise_noise_match=True,
    )


def _default_timeseries_plan() -> TimeseriesDatasetPlan:
    return TimeseriesDatasetPlan(
        burn_in=40,
        sample_amount=0.02,
        n_series=1,
        frames_per_series=200,
    )


def _default_output() -> DatasetOutputConfig:
    return DatasetOutputConfig(
        save_dir="/tmp/symbac_napari_export",
        image_format="png",
        mask_dtype="uint16",
        export_geff=True,
        n_jobs=1,
    )


class ExportDock:
    def __init__(self, controller):
        from qtpy.QtWidgets import (
            QComboBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPlainTextEdit,
            QPushButton,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        self.controller = controller
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["random", "timeseries", "batch_timeseries"])
        mode_row.addWidget(self.mode_combo)
        layout.addLayout(mode_row)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seed"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(42)
        seed_row.addWidget(self.seed_spin)
        layout.addLayout(seed_row)

        layout.addWidget(QLabel("Plan YAML"))
        self.plan_editor = QPlainTextEdit(model_to_yaml_text(_default_random_plan()))
        layout.addWidget(self.plan_editor)

        layout.addWidget(QLabel("Output YAML"))
        self.output_editor = QPlainTextEdit(model_to_yaml_text(_default_output()))
        layout.addWidget(self.output_editor)

        layout.addWidget(QLabel("Batch variants YAML"))
        self.variants_editor = QPlainTextEdit(default_variants_yaml())
        layout.addWidget(self.variants_editor)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Export config path"))
        self.path_edit = QLineEdit("/tmp/symbac_export_output.yaml")
        path_row.addWidget(self.path_edit)
        layout.addLayout(path_row)

        button_row = QHBoxLayout()
        self.export_button = QPushButton("Run Export")
        self.save_output_button = QPushButton("Save Output YAML")
        button_row.addWidget(self.export_button)
        button_row.addWidget(self.save_output_button)
        layout.addLayout(button_row)

        self.mode_combo.currentTextChanged.connect(self._refresh_plan_template)
        self.export_button.clicked.connect(self._run_export)
        self.save_output_button.clicked.connect(self._save_output_yaml)

    def _refresh_plan_template(self, mode: str):
        if mode == "random":
            self.plan_editor.setPlainText(model_to_yaml_text(_default_random_plan()))
        else:
            self.plan_editor.setPlainText(model_to_yaml_text(_default_timeseries_plan()))

    def _parse_output(self) -> DatasetOutputConfig:
        return model_from_yaml_text(DatasetOutputConfig, self.output_editor.toPlainText())

    def _read_base_config(self) -> RenderConfig:
        cfg = self.controller.state.base_render_config
        if not isinstance(cfg, RenderConfig):
            return RenderConfig()
        return cfg

    def _run_export(self):
        from napari.utils.notifications import show_info

        mode = self.mode_combo.currentText()
        seed = int(self.seed_spin.value())
        output = self._parse_output()
        base_config = self._read_base_config()

        if mode == "random":
            plan = model_from_yaml_text(RandomDatasetPlan, self.plan_editor.toPlainText())

            def _task():
                return self.controller.export_dataset(plan=plan, output=output, base_config=base_config, seed=seed)

        elif mode == "timeseries":
            plan = model_from_yaml_text(TimeseriesDatasetPlan, self.plan_editor.toPlainText())

            def _task():
                return self.controller.export_dataset(plan=plan, output=output, base_config=base_config, seed=seed)

        else:
            plan = model_from_yaml_text(TimeseriesDatasetPlan, self.plan_editor.toPlainText())
            variants = load_variants_yaml_text(self.variants_editor.toPlainText())

            def _task():
                return self.controller.export_batch_timeseries(
                    variants=variants,
                    plan=plan,
                    output=output,
                    base_config=base_config,
                    seed=seed,
                )

        def _done(_meta):
            show_info(f"Export complete. Output: {output.save_dir}")

        start_worker(_task, on_return=_done)

    def _save_output_yaml(self):
        from napari.utils.notifications import show_info

        output = self._parse_output()
        save_model(output, Path(self.path_edit.text()))
        show_info("Saved output config YAML.")
