from __future__ import annotations

from pathlib import Path

from SyMBac.config_models import DatasetOutputConfig, RandomDatasetPlan, RenderConfig, TimeseriesDatasetPlan
from SyMBac.napari.io.variant_loader import default_variants_yaml, load_variants_yaml_text
from SyMBac.napari.io.yaml_store import model_from_yaml_text, model_to_yaml_text, save_model
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
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPlainTextEdit,
            QPushButton,
            QScrollArea,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        self.controller = controller
        self.widget = QWidget()
        root = QVBoxLayout(self.widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll)
        container = QWidget()
        layout = QVBoxLayout(container)
        scroll.setWidget(container)

        mode_group = QGroupBox("Mode")
        mode_row = QHBoxLayout(mode_group)
        mode_row.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["random", "timeseries", "batch_timeseries"])
        mode_row.addWidget(self.mode_combo)
        mode_row.addWidget(QLabel("Seed"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(42)
        mode_row.addWidget(self.seed_spin)
        layout.addWidget(mode_group)

        random_group = QGroupBox("Random Plan")
        random_form = QFormLayout(random_group)
        default_random = _default_random_plan()
        self.random_burn_in = QSpinBox()
        self.random_burn_in.setRange(0, 1_000_000)
        self.random_burn_in.setValue(default_random.burn_in)
        random_form.addRow("burn_in", self.random_burn_in)

        self.random_n_samples = QSpinBox()
        self.random_n_samples.setRange(1, 10_000_000)
        self.random_n_samples.setValue(default_random.n_samples)
        random_form.addRow("n_samples", self.random_n_samples)

        self.random_sample_amount = QDoubleSpinBox()
        self.random_sample_amount.setRange(0.0, 10.0)
        self.random_sample_amount.setDecimals(4)
        self.random_sample_amount.setValue(default_random.sample_amount)
        random_form.addRow("sample_amount", self.random_sample_amount)

        self.random_hist = QCheckBox()
        self.random_hist.setChecked(default_random.randomise_hist_match)
        random_form.addRow("randomise_hist_match", self.random_hist)

        self.random_noise = QCheckBox()
        self.random_noise.setChecked(default_random.randomise_noise_match)
        random_form.addRow("randomise_noise_match", self.random_noise)

        self.random_fourier = QCheckBox()
        self.random_fourier.setChecked(default_random.randomise_fourier_match)
        random_form.addRow("randomise_fourier_match", self.random_fourier)
        layout.addWidget(random_group)

        timeseries_group = QGroupBox("Timeseries Plan")
        timeseries_form = QFormLayout(timeseries_group)
        default_ts = _default_timeseries_plan()
        self.ts_burn_in = QSpinBox()
        self.ts_burn_in.setRange(0, 1_000_000)
        self.ts_burn_in.setValue(default_ts.burn_in)
        timeseries_form.addRow("burn_in", self.ts_burn_in)

        self.ts_sample_amount = QDoubleSpinBox()
        self.ts_sample_amount.setRange(0.0, 10.0)
        self.ts_sample_amount.setDecimals(4)
        self.ts_sample_amount.setValue(default_ts.sample_amount)
        timeseries_form.addRow("sample_amount", self.ts_sample_amount)

        self.ts_n_series = QSpinBox()
        self.ts_n_series.setRange(1, 1_000_000)
        self.ts_n_series.setValue(default_ts.n_series)
        timeseries_form.addRow("n_series", self.ts_n_series)

        self.ts_frames = QSpinBox()
        self.ts_frames.setRange(0, 10_000_000)
        self.ts_frames.setValue(default_ts.frames_per_series or 0)
        self.ts_frames.setToolTip("Set to 0 to use all available frames after burn-in.")
        timeseries_form.addRow("frames_per_series", self.ts_frames)
        layout.addWidget(timeseries_group)

        output_group = QGroupBox("Output")
        output_form = QFormLayout(output_group)
        default_out = _default_output()

        self.save_dir_edit = QLineEdit(default_out.save_dir)
        output_form.addRow("save_dir", self.save_dir_edit)

        self.image_format = QComboBox()
        self.image_format.addItems(["png", "tif", "tiff"])
        self.image_format.setCurrentText(default_out.image_format)
        output_form.addRow("image_format", self.image_format)

        self.mask_dtype = QComboBox()
        self.mask_dtype.addItems(["uint8", "uint16", "uint32", "int32"])
        self.mask_dtype.setCurrentText(default_out.mask_dtype)
        output_form.addRow("mask_dtype", self.mask_dtype)

        self.n_jobs = QSpinBox()
        self.n_jobs.setRange(-64, 64)
        self.n_jobs.setValue(default_out.n_jobs)
        output_form.addRow("n_jobs", self.n_jobs)

        self.export_geff = QCheckBox()
        self.export_geff.setChecked(default_out.export_geff)
        output_form.addRow("export_geff", self.export_geff)

        self.prefix_edit = QLineEdit("")
        self.prefix_edit.setPlaceholderText("optional")
        output_form.addRow("prefix", self.prefix_edit)
        layout.addWidget(output_group)

        self.sync_yaml_button = QPushButton("Sync Form -> YAML")
        layout.addWidget(self.sync_yaml_button)

        self.advanced_group = QGroupBox("Advanced YAML")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        advanced_layout = QVBoxLayout(self.advanced_group)

        advanced_layout.addWidget(QLabel("Plan YAML"))
        self.plan_editor = QPlainTextEdit(model_to_yaml_text(default_random))
        advanced_layout.addWidget(self.plan_editor)

        advanced_layout.addWidget(QLabel("Output YAML"))
        self.output_editor = QPlainTextEdit(model_to_yaml_text(default_out))
        advanced_layout.addWidget(self.output_editor)

        advanced_layout.addWidget(QLabel("Batch variants YAML"))
        self.variants_editor = QPlainTextEdit(default_variants_yaml())
        advanced_layout.addWidget(self.variants_editor)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Output config path"))
        self.path_edit = QLineEdit("/tmp/symbac_export_output.yaml")
        path_row.addWidget(self.path_edit)
        advanced_layout.addLayout(path_row)

        self.save_output_button = QPushButton("Save Output YAML")
        advanced_layout.addWidget(self.save_output_button)

        layout.addWidget(self.advanced_group)

        self.export_button = QPushButton("Run Export")
        layout.addWidget(self.export_button)
        layout.addStretch(1)

        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.sync_yaml_button.clicked.connect(self._sync_form_to_yaml)
        self.export_button.clicked.connect(self._run_export)
        self.save_output_button.clicked.connect(self._save_output_yaml)

        self._on_mode_changed(self.mode_combo.currentText())

    def _on_mode_changed(self, mode: str) -> None:
        is_random = mode == "random"
        self.random_burn_in.parent().setEnabled(is_random)
        self.ts_burn_in.parent().setEnabled(not is_random)
        self._sync_form_to_yaml()

    def _plan_from_form(self):
        mode = self.mode_combo.currentText()
        if mode == "random":
            return RandomDatasetPlan(
                burn_in=int(self.random_burn_in.value()),
                n_samples=int(self.random_n_samples.value()),
                sample_amount=float(self.random_sample_amount.value()),
                randomise_hist_match=bool(self.random_hist.isChecked()),
                randomise_noise_match=bool(self.random_noise.isChecked()),
                randomise_fourier_match=bool(self.random_fourier.isChecked()),
            )

        frames = int(self.ts_frames.value())
        return TimeseriesDatasetPlan(
            burn_in=int(self.ts_burn_in.value()),
            sample_amount=float(self.ts_sample_amount.value()),
            n_series=int(self.ts_n_series.value()),
            frames_per_series=(None if frames == 0 else frames),
        )

    def _output_from_form(self) -> DatasetOutputConfig:
        prefix_text = self.prefix_edit.text().strip()
        return DatasetOutputConfig(
            save_dir=self.save_dir_edit.text().strip(),
            image_format=self.image_format.currentText(),
            mask_dtype=self.mask_dtype.currentText(),
            n_jobs=int(self.n_jobs.value()),
            prefix=(prefix_text if prefix_text else None),
            export_geff=bool(self.export_geff.isChecked()),
        )

    def _parse_output_yaml(self) -> DatasetOutputConfig:
        return model_from_yaml_text(DatasetOutputConfig, self.output_editor.toPlainText())

    def _read_base_config(self) -> RenderConfig:
        cfg = self.controller.state.base_render_config
        if not isinstance(cfg, RenderConfig):
            return RenderConfig()
        return cfg

    def _sync_form_to_yaml(self):
        from napari.utils.notifications import show_error

        try:
            plan = self._plan_from_form()
            output = self._output_from_form()
        except Exception as exc:
            show_error(f"Invalid export form values: {exc}")
            return
        self.plan_editor.setPlainText(model_to_yaml_text(plan))
        self.output_editor.setPlainText(model_to_yaml_text(output))

    def _run_export(self):
        from napari.utils.notifications import show_error, show_info

        mode = self.mode_combo.currentText()
        seed = int(self.seed_spin.value())
        base_config = self._read_base_config()

        try:
            if self.advanced_group.isChecked():
                output = self._parse_output_yaml()
                if mode == "random":
                    plan = model_from_yaml_text(RandomDatasetPlan, self.plan_editor.toPlainText())
                else:
                    plan = model_from_yaml_text(TimeseriesDatasetPlan, self.plan_editor.toPlainText())
            else:
                plan = self._plan_from_form()
                output = self._output_from_form()
        except Exception as exc:
            show_error(f"Invalid export configuration: {exc}")
            return

        if mode == "batch_timeseries":
            if not isinstance(plan, TimeseriesDatasetPlan):
                show_error("Batch timeseries export requires a TimeseriesDatasetPlan.")
                return
            try:
                variants = load_variants_yaml_text(self.variants_editor.toPlainText())
            except Exception as exc:
                show_error(f"Invalid variants YAML: {exc}")
                return

            def _task():
                return self.controller.export_batch_timeseries(
                    variants=variants,
                    plan=plan,
                    output=output,
                    base_config=base_config,
                    seed=seed,
                )

        else:
            def _task():
                return self.controller.export_dataset(plan=plan, output=output, base_config=base_config, seed=seed)

        def _done(_meta):
            show_info(f"Export complete. Output: {output.save_dir}")

        start_worker(_task, on_return=_done)

    def _save_output_yaml(self):
        from napari.utils.notifications import show_error, show_info

        try:
            output = self._parse_output_yaml() if self.advanced_group.isChecked() else self._output_from_form()
            save_model(output, Path(self.path_edit.text()))
        except Exception as exc:
            show_error(f"Failed to save output config YAML: {exc}")
            return
        show_info("Saved output config YAML.")
