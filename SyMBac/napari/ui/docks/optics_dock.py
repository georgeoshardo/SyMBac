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
            QFileDialog,
            QGroupBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QScrollArea,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        self.controller = controller
        self.layer_manager = layer_manager
        self._QFileDialog = QFileDialog

        self.widget = QWidget()
        root = QVBoxLayout(self.widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        scroll.setWidget(container)
        root.addWidget(scroll)

        # --- Real image section ---
        image_group = QGroupBox("Real Image")
        image_layout = QVBoxLayout(image_group)

        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel("Sample"))
        self.sample_combo = QComboBox()
        self.sample_keys = sorted(get_sample_images().keys())
        self.sample_combo.addItems(self.sample_keys)
        sample_row.addWidget(self.sample_combo)
        self.load_sample_button = QPushButton("Load Sample")
        sample_row.addWidget(self.load_sample_button)
        image_layout.addLayout(sample_row)

        browse_row = QHBoxLayout()
        self.browse_button = QPushButton("Browse Image File...")
        browse_row.addWidget(self.browse_button)
        image_layout.addLayout(browse_row)

        layout.addWidget(image_group)

        # --- PSF section ---
        psf_group = QGroupBox("PSF")
        psf_layout = QVBoxLayout(psf_group)
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
                if key in {"radius", "resize_amount"}:
                    spin = QSpinBox()
                    spin.setMaximum(10000)
                    spin.setMinimum(1)
                    spin.setValue(int(value))
                else:
                    spin = QDoubleSpinBox()
                    spin.setDecimals(4)
                    spin.setMaximum(10000)
                    spin.setMinimum(0.0)
                    spin.setValue(float(value))
                self.psf_inputs[key] = spin
                self.psf_form.addRow(key, spin)
        psf_layout.addLayout(self.psf_form)
        self.preview_psf_button = QPushButton("Preview PSF")
        psf_layout.addWidget(self.preview_psf_button)
        layout.addWidget(psf_group)

        # --- Camera section ---
        camera_group = QGroupBox("Camera")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_enabled = QCheckBox("Enable camera")
        self.camera_enabled.setChecked(True)
        camera_layout.addWidget(self.camera_enabled)

        self.camera_form = QFormLayout()
        self.camera_inputs: dict[str, QDoubleSpinBox] = {}
        for key, value in _DEFAULT_CAMERA_PARAMS.items():
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setMaximum(10000)
            spin.setMinimum(0.0)
            spin.setValue(float(value))
            self.camera_inputs[key] = spin
            self.camera_form.addRow(key, spin)
        camera_layout.addLayout(self.camera_form)
        layout.addWidget(camera_group)

        self.build_renderer_button = QPushButton("Build Renderer")
        layout.addWidget(self.build_renderer_button)
        layout.addStretch(1)

        self.load_sample_button.clicked.connect(self._load_sample_image)
        self.browse_button.clicked.connect(self._browse_image_file)
        self.preview_psf_button.clicked.connect(self._preview_psf)
        self.build_renderer_button.clicked.connect(self._build_renderer)

    def _load_sample_image(self):
        from napari.utils.notifications import show_info

        key = self.sample_combo.currentText()
        image = get_sample_images()[key]
        self.controller.set_real_image(image)
        self.layer_manager.update_real_image(image)
        show_info(f"Loaded real image: {key}")

    def _browse_image_file(self):
        import numpy as np
        from napari.utils.notifications import show_error, show_info

        path, _ = self._QFileDialog.getOpenFileName(
            self.widget,
            "Open Real Image",
            "",
            "Images (*.tif *.tiff *.png);;All Files (*)",
        )
        if not path:
            return
        try:
            if path.lower().endswith(".png"):
                from imageio.v3 import imread
                image = np.asarray(imread(path))
            else:
                import tifffile
                image = np.asarray(tifffile.imread(path))
            if image.ndim == 3 and image.shape[-1] in (3, 4):
                image = np.mean(image[..., :3], axis=-1)
        except Exception as exc:
            show_error(f"Failed to load image: {exc}")
            return
        self.controller.set_real_image(image)
        self.layer_manager.update_real_image(image)
        show_info(f"Loaded real image: {path}")

    def _preview_psf(self):
        from napari.utils.notifications import show_error, show_info

        try:
            psf = self.controller.preview_psf(self._read_psf_params())
            self.layer_manager.update_psf_preview(psf.kernel)
        except Exception as exc:
            show_error(f"Failed to preview PSF: {exc}")
            return
        show_info("PSF preview updated.")

    def _read_psf_params(self) -> dict:
        out = {}
        for key, widget in self.psf_inputs.items():
            if hasattr(widget, "currentText"):
                out[key] = widget.currentText()
            elif key in {"radius", "resize_amount"}:
                out[key] = int(widget.value())
            else:
                out[key] = float(widget.value())
        return out

    def _read_camera_params(self) -> dict | None:
        if not self.camera_enabled.isChecked():
            return None
        return {key: float(widget.value()) for key, widget in self.camera_inputs.items()}

    def _build_renderer(self):
        from napari.utils.notifications import show_error, show_info

        try:
            renderer = self.controller.build_renderer(
                psf_params=self._read_psf_params(),
                camera_params=self._read_camera_params(),
            )
            renderer._ensure_image_params()
        except Exception as exc:
            show_error(f"Failed to build renderer: {exc}")
            return
        show_info("Renderer created.")
