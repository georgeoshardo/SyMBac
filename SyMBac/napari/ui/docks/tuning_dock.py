from __future__ import annotations

from pathlib import Path

from SyMBac.config_models import RenderConfig
from SyMBac.napari.io.yaml_store import load_render_config, model_to_yaml_text, save_model

_SLIDER_STEPS = 1000


class TuningDock:
    def __init__(self, controller, layer_manager):
        from qtpy.QtCore import Qt, QTimer
        from qtpy.QtWidgets import (
            QCheckBox,
            QDoubleSpinBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QSlider,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        self.controller = controller
        self.layer_manager = layer_manager
        self._updating = False
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)

        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 100000)
        self.frame_spin.setValue(0)

        self.inputs: dict[str, object] = {}

        form = QFormLayout()
        form.addRow("frame_index", self.frame_spin)

        def _add_float(name, value, minimum=-1e6, maximum=1e6, decimals=4):
            row = QHBoxLayout()
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, _SLIDER_STEPS)
            spin = QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setRange(minimum, maximum)
            spin.setSingleStep(10 ** (-decimals))
            spin.setValue(float(value))
            # Set initial slider position
            frac = (float(value) - minimum) / (maximum - minimum) if maximum > minimum else 0
            slider.setValue(int(frac * _SLIDER_STEPS))

            def _slider_to_spin(pos, s=spin, mn=minimum, mx=maximum):
                if self._updating:
                    return
                self._updating = True
                s.setValue(mn + (pos / _SLIDER_STEPS) * (mx - mn))
                self._updating = False

            def _spin_to_slider(val, sl=slider, mn=minimum, mx=maximum):
                if self._updating:
                    return
                self._updating = True
                frac = (val - mn) / (mx - mn) if mx > mn else 0
                sl.setValue(int(frac * _SLIDER_STEPS))
                self._updating = False

            slider.valueChanged.connect(_slider_to_spin)
            spin.valueChanged.connect(_spin_to_slider)
            slider.valueChanged.connect(lambda *_: self._schedule_preview())
            spin.valueChanged.connect(lambda *_: self._schedule_preview())

            row.addWidget(slider, stretch=3)
            row.addWidget(spin, stretch=1)
            self.inputs[name] = spin
            form.addRow(name, row)

        def _add_bool(name, value):
            box = QCheckBox()
            box.setChecked(bool(value))
            box.stateChanged.connect(lambda *_: self._schedule_preview())
            self.inputs[name] = box
            form.addRow(name, box)

        defaults = RenderConfig()
        _add_float("media_multiplier", defaults.media_multiplier, minimum=0, maximum=500)
        _add_float("cell_multiplier", defaults.cell_multiplier, minimum=0, maximum=20)
        _add_float("device_multiplier", defaults.device_multiplier, minimum=0, maximum=500)
        _add_float("sigma", defaults.sigma, minimum=0.01, maximum=50)
        _add_bool("match_fourier", defaults.match_fourier)
        _add_bool("match_histogram", defaults.match_histogram)
        _add_bool("match_noise", defaults.match_noise)
        _add_float("noise_var", defaults.noise_var, minimum=0, maximum=1)
        _add_float("defocus", defaults.defocus, minimum=0, maximum=50)
        _add_float("halo_top_intensity", defaults.halo_top_intensity, minimum=-10, maximum=10)
        _add_float("halo_bottom_intensity", defaults.halo_bottom_intensity, minimum=-10, maximum=10)
        _add_float("halo_start", defaults.halo_start, minimum=0, maximum=1)
        _add_float("halo_end", defaults.halo_end, minimum=0, maximum=1)
        _add_float("cell_texture_strength", defaults.cell_texture_strength, minimum=0, maximum=10)
        _add_float("cell_texture_scale", defaults.cell_texture_scale, minimum=1, maximum=500)
        _add_float("edge_floor_opl", defaults.edge_floor_opl, minimum=0, maximum=10)

        self.frame_spin.valueChanged.connect(lambda *_: self._schedule_preview())

        layout.addLayout(form)

        self.live_check = QCheckBox("Live preview")
        self.live_check.setChecked(True)
        layout.addWidget(self.live_check)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("RenderConfig path"))
        self.path_edit = QLineEdit("/tmp/symbac_render_config.yaml")
        path_row.addWidget(self.path_edit)
        layout.addLayout(path_row)

        button_row = QHBoxLayout()
        self.preview_button = QPushButton("Preview Frame")
        self.sync_button = QPushButton("Sync From Active")
        self.save_button = QPushButton("Save YAML")
        self.load_button = QPushButton("Load YAML")
        button_row.addWidget(self.preview_button)
        button_row.addWidget(self.sync_button)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.load_button)
        layout.addLayout(button_row)

        self.preview_button.clicked.connect(self._preview)
        self.sync_button.clicked.connect(self._sync_from_state)
        self.save_button.clicked.connect(self._save_yaml)
        self.load_button.clicked.connect(self._load_yaml)

        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)
        self._debounce_timer.timeout.connect(self._live_preview)

    def _schedule_preview(self):
        if self.live_check.isChecked() and self.controller.state.renderer is not None:
            self._debounce_timer.start()

    def _live_preview(self):
        try:
            config = self._read_config()
            frame_index = int(self.frame_spin.value())
            result = self.controller.preview_frame(frame_index=frame_index, config=config)
            self.layer_manager.update_preview_layers(result)
        except Exception:
            pass  # silently skip if renderer not ready

    def _read_config(self) -> RenderConfig:
        payload = {}
        for name, widget in self.inputs.items():
            if hasattr(widget, "isChecked"):
                payload[name] = bool(widget.isChecked())
            else:
                payload[name] = float(widget.value())
        return RenderConfig(**payload)

    def _set_config(self, config: RenderConfig) -> None:
        old_live = self.live_check.isChecked()
        self.live_check.setChecked(False)
        values = config.model_dump(mode="python")
        for name, value in values.items():
            widget = self.inputs.get(name)
            if widget is None:
                continue
            if hasattr(widget, "setChecked"):
                widget.setChecked(bool(value))
            else:
                widget.setValue(float(value))
        self.live_check.setChecked(old_live)

    def _preview(self):
        from napari.utils.notifications import show_error, show_info

        try:
            config = self._read_config()
            frame_index = int(self.frame_spin.value())
            result = self.controller.preview_frame(frame_index=frame_index, config=config)
            self.layer_manager.update_preview_layers(result)
        except Exception as exc:
            show_error(f"Failed to render preview: {exc}")
            return
        show_info("Preview rendered.")

    def _sync_from_state(self):
        self._set_config(self.controller.state.base_render_config)

    def _save_yaml(self):
        from napari.utils.notifications import show_info

        config = self._read_config()
        save_model(config, Path(self.path_edit.text()))
        show_info("Saved RenderConfig YAML.")

    def _load_yaml(self):
        from napari.utils.notifications import show_info

        config = load_render_config(Path(self.path_edit.text()))
        self._set_config(config)
        self.controller.state.base_render_config = config
        show_info("Loaded RenderConfig YAML.")
