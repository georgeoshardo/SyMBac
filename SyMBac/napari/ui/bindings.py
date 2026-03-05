from __future__ import annotations

from dataclasses import dataclass, field
import weakref

from SyMBac.napari.controllers.workflow_controller import WorkflowController
from SyMBac.napari.state import NapariSessionState
from SyMBac.napari.ui.docks.export_dock import ExportDock
from SyMBac.napari.ui.docks.optics_dock import OpticsDock
from SyMBac.napari.ui.docks.regions_dock import RegionsDock
from SyMBac.napari.ui.docks.simulation_dock import SimulationDock
from SyMBac.napari.ui.docks.tuning_dock import TuningDock
from SyMBac.napari.ui.layer_manager import LayerManager
from SyMBac.napari.ui.workflow_dock import WorkflowDock

_CONTEXTS_BY_VIEWER_ID: dict[int, NapariUIContext] = {}
_VIEWER_REFS: dict[int, weakref.ref] = {}


@dataclass
class NapariUIContext:
    state: NapariSessionState
    controller: WorkflowController
    layer_manager: LayerManager
    docks: dict[str, object] = field(default_factory=dict)


def get_or_create_context(viewer) -> NapariUIContext:
    viewer_id = id(viewer)
    existing_ref = _VIEWER_REFS.get(viewer_id)
    if existing_ref is not None and existing_ref() is viewer:
        ctx = _CONTEXTS_BY_VIEWER_ID.get(viewer_id)
        if ctx is not None:
            return ctx

    state = NapariSessionState()
    layer_manager = LayerManager(viewer)
    controller = WorkflowController(state=state)
    ctx = NapariUIContext(state=state, controller=controller, layer_manager=layer_manager)

    _CONTEXTS_BY_VIEWER_ID[viewer_id] = ctx

    def _cleanup(_ref, _viewer_id=viewer_id):
        _VIEWER_REFS.pop(_viewer_id, None)
        _CONTEXTS_BY_VIEWER_ID.pop(_viewer_id, None)

    try:
        _VIEWER_REFS[viewer_id] = weakref.ref(viewer, _cleanup)
    except TypeError:
        # Fallback for objects that cannot be weak-referenced.
        _VIEWER_REFS[viewer_id] = lambda: viewer

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
    elif key == "workflow":
        dock = WorkflowDock(context)
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


def get_workflow_widget(viewer):
    return _get_or_build_dock(get_or_create_context(viewer), "workflow").widget


def register_default_docks(viewer) -> None:
    widget = get_workflow_widget(viewer)
    viewer.window.add_dock_widget(widget, name="SyMBac Workflow", area="right")
