from __future__ import annotations

from SyMBac.misc import get_sample_images


_DEFAULT_PSF_PARAMS = {
    "radius": 50,
    "wavelength": 0.75,
    "NA": 1.2,
    "n": 1.3,
    "resize_amount": 3,
    "pix_mic_conv": 0.065,
    "apo_sigma": 20,
    "mode": "phase contrast",
    "condenser": "Ph3",
}

_DEFAULT_CAMERA_PARAMS = {
    "baseline": 100,
    "sensitivity": 2.9,
    "dark_noise": 8,
}


class OpticsDock:
    def __init__(self, controller, layer_manager):
        from qtpy.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        self.controller = controller
        self.layer_manager = layer_manager

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)

        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel("Real image"))
        self.sample_combo = QComboBox()
        self.sample_keys = sorted(get_sample_images().keys())
        self.sample_combo.addItems(self.sample_keys)
        sample_row.addWidget(self.sample_combo)
        self.load_image_button = QPushButton("Load Real Image")
        sample_row.addWidget(self.load_image_button)
        layout.addLayout(sample_row)

        self.psf_form = QFormLayout()
        self.psf_inputs: dict[str, object] = {}
        for key, value in _DEFAULT_PSF_PARAMS.items():
            if isinstance(value, str):
                combo = QComboBox()
                if key == "mode":
                    combo.addItems(["phase contrast", "3d fluo"])
                elif key == "condenser":
                    combo.addItems(["Ph1", "Ph2", "Ph3"])
                combo.setCurrentText(str(value))
                self.psf_inputs[key] = combo
                self.psf_form.addRow(key, combo)
            else:
                spin = QDoubleSpinBox()
                spin.setDecimals(4)
                spin.setMaximum(10000)
                spin.setMinimum(-10000)
                spin.setValue(float(value))
                self.psf_inputs[key] = spin
                self.psf_form.addRow(key, spin)
        layout.addLayout(self.psf_form)

        self.camera_enabled = QCheckBox("Enable camera")
        self.camera_enabled.setChecked(True)
        layout.addWidget(self.camera_enabled)

        self.camera_form = QFormLayout()
        self.camera_inputs: dict[str, QDoubleSpinBox] = {}
        for key, value in _DEFAULT_CAMERA_PARAMS.items():
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setMaximum(10000)
            spin.setMinimum(-10000)
            spin.setValue(float(value))
            self.camera_inputs[key] = spin
            self.camera_form.addRow(key, spin)
        layout.addLayout(self.camera_form)

        self.build_renderer_button = QPushButton("Build Renderer")
        layout.addWidget(self.build_renderer_button)

        self.load_image_button.clicked.connect(self._load_real_image)
        self.build_renderer_button.clicked.connect(self._build_renderer)

    def _load_real_image(self):
        from napari.utils.notifications import show_info

        key = self.sample_combo.currentText()
        image = get_sample_images()[key]
        self.controller.set_real_image(image)
        self.layer_manager.update_real_image(image)
        show_info(f"Loaded real image: {key}")

    def _read_psf_params(self) -> dict:
        out = {}
        for key, widget in self.psf_inputs.items():
            if hasattr(widget, "currentText"):
                out[key] = widget.currentText()
            elif key == "resize_amount":
                out[key] = int(round(widget.value()))
            else:
                out[key] = float(widget.value())
        return out

    def _read_camera_params(self) -> dict | None:
        if not self.camera_enabled.isChecked():
            return None
        return {key: float(widget.value()) for key, widget in self.camera_inputs.items()}

    def _build_renderer(self):
        from napari.utils.notifications import show_info

        renderer = self.controller.build_renderer(
            psf_params=self._read_psf_params(),
            camera_params=self._read_camera_params(),
        )
        renderer._ensure_image_params()
        show_info("Renderer created.")
