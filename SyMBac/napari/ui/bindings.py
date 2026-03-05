from __future__ import annotations

from dataclasses import dataclass, field

from SyMBac.napari.controllers.workflow_controller import WorkflowController
from SyMBac.napari.state import NapariSessionState
from SyMBac.napari.ui.docks.export_dock import ExportDock
from SyMBac.napari.ui.docks.optics_dock import OpticsDock
from SyMBac.napari.ui.docks.regions_dock import RegionsDock
from SyMBac.napari.ui.docks.simulation_dock import SimulationDock
from SyMBac.napari.ui.docks.tuning_dock import TuningDock
from SyMBac.napari.ui.layer_manager import LayerManager


@dataclass
class NapariUIContext:
    state: NapariSessionState
    controller: WorkflowController
    layer_manager: LayerManager
    docks: dict[str, object] = field(default_factory=dict)


def get_or_create_context(viewer) -> NapariUIContext:
    ctx = getattr(viewer, "_symbac_ui_context", None)
    if ctx is not None:
        return ctx

    state = NapariSessionState()
    layer_manager = LayerManager(viewer)
    controller = WorkflowController(state=state)
    ctx = NapariUIContext(state=state, controller=controller, layer_manager=layer_manager)
    setattr(viewer, "_symbac_ui_context", ctx)
    return ctx


def _get_or_build_dock(context: NapariUIContext, key: str):
    if key in context.docks:
        return context.docks[key]

    if key == "simulation":
        dock = SimulationDock(context.controller, context.layer_manager)
    elif key == "optics":
        dock = OpticsDock(context.controller, context.layer_manager)
    elif key == "regions":
        dock = RegionsDock(context.controller, context.layer_manager)
    elif key == "tuning":
        dock = TuningDock(context.controller, context.layer_manager)
    elif key == "export":
        dock = ExportDock(context.controller)
    else:
        raise ValueError(f"Unknown dock key: {key}")

    context.docks[key] = dock
    return dock


def get_simulation_widget(viewer):
    return _get_or_build_dock(get_or_create_context(viewer), "simulation").widget


def get_optics_widget(viewer):
    return _get_or_build_dock(get_or_create_context(viewer), "optics").widget


def get_regions_widget(viewer):
    return _get_or_build_dock(get_or_create_context(viewer), "regions").widget


def get_tuning_widget(viewer):
    return _get_or_build_dock(get_or_create_context(viewer), "tuning").widget


def get_export_widget(viewer):
    return _get_or_build_dock(get_or_create_context(viewer), "export").widget


def register_default_docks(viewer) -> None:
    targets = [
        ("SyMBac Simulation", get_simulation_widget),
        ("SyMBac Optics", get_optics_widget),
        ("SyMBac Regions", get_regions_widget),
        ("SyMBac Tuning", get_tuning_widget),
        ("SyMBac Export", get_export_widget),
    ]
    for name, factory in targets:
        widget = factory(viewer)
        viewer.window.add_dock_widget(widget, name=name, area="right")
