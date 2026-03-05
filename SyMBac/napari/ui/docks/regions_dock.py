from __future__ import annotations


class RegionsDock:
    def __init__(self, controller, layer_manager):
        from qtpy.QtWidgets import (
            QComboBox,
            QFormLayout,
            QPushButton,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        self.controller = controller
        self.layer_manager = layer_manager

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)

        form = QFormLayout()
        self.classes_spin = QSpinBox()
        self.classes_spin.setRange(2, 8)
        self.classes_spin.setValue(3)
        form.addRow("Otsu classes", self.classes_spin)

        self.cells_combo = QComboBox()
        self.cells_combo.addItems(["dark", "light"])
        form.addRow("Cells", self.cells_combo)
        layout.addLayout(form)

        self.auto_button = QPushButton("Auto Segment Regions")
        self.apply_button = QPushButton("Apply Region Labels")
        self.clear_button = QPushButton("Clear Region Masks")
        layout.addWidget(self.auto_button)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.clear_button)

        self.auto_button.clicked.connect(self._auto_segment)
        self.apply_button.clicked.connect(self._apply_regions)
        self.clear_button.clicked.connect(self._clear_regions)

    def _auto_segment(self):
        from napari.utils.notifications import show_info

        masks = self.controller.auto_segment_regions(
            classes=int(self.classes_spin.value()),
            cells=self.cells_combo.currentText(),
        )
        self.layer_manager.set_region_layers(masks)
        self.controller.set_region_masks(masks)
        show_info("Region labels initialised in viewer.")

    def _apply_regions(self):
        from napari.utils.notifications import show_info

        masks = self.layer_manager.collect_region_masks()
        self.controller.set_region_masks(masks)
        show_info("Applied region masks to renderer.")

    def _clear_regions(self):
        from napari.utils.notifications import show_info

        if self.controller.state.renderer is None:
            raise ValueError("Renderer has not been created.")
        self.controller.state.renderer.clear_region_masks()
        show_info("Cleared explicit renderer region masks.")
