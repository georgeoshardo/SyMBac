from __future__ import annotations


class WorkflowDock:
    """Single-tabbed SyMBac workflow container for napari."""

    def __init__(self, context):
        from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget

        from SyMBac.napari.ui.docks.export_dock import ExportDock
        from SyMBac.napari.ui.docks.optics_dock import OpticsDock
        from SyMBac.napari.ui.docks.regions_dock import RegionsDock
        from SyMBac.napari.ui.docks.simulation_dock import SimulationDock
        from SyMBac.napari.ui.docks.tuning_dock import TuningDock

        self.context = context
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.North)
        layout.addWidget(self.tabs)

        controller = context.controller
        layers = context.layer_manager

        simulation = SimulationDock(controller, layers)
        optics = OpticsDock(controller, layers)
        regions = RegionsDock(controller, layers)
        tuning = TuningDock(controller, layers)
        export = ExportDock(controller)

        self.tabs.addTab(simulation.widget, "Simulation")
        self.tabs.addTab(optics.widget, "Optics")
        self.tabs.addTab(regions.widget, "Regions")
        self.tabs.addTab(tuning.widget, "Tuning")
        self.tabs.addTab(export.widget, "Export")

        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index: int) -> None:
        tab_name = self.tabs.tabText(index)
        self.context.layer_manager.show_only_layers(tab_name)
