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

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setTabPosition(QTabWidget.North)
        layout.addWidget(tabs)

        controller = context.controller
        layers = context.layer_manager

        simulation = context.docks.get("simulation")
        if simulation is None:
            simulation = SimulationDock(controller, layers)
            context.docks["simulation"] = simulation

        optics = context.docks.get("optics")
        if optics is None:
            optics = OpticsDock(controller, layers)
            context.docks["optics"] = optics

        regions = context.docks.get("regions")
        if regions is None:
            regions = RegionsDock(controller, layers)
            context.docks["regions"] = regions

        tuning = context.docks.get("tuning")
        if tuning is None:
            tuning = TuningDock(controller, layers)
            context.docks["tuning"] = tuning

        export = context.docks.get("export")
        if export is None:
            export = ExportDock(controller)
            context.docks["export"] = export

        tabs.addTab(simulation.widget, "Simulation")
        tabs.addTab(optics.widget, "Optics")
        tabs.addTab(regions.widget, "Regions")
        tabs.addTab(tuning.widget, "Tuning")
        tabs.addTab(export.widget, "Export")
