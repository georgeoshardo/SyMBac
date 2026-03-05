from __future__ import annotations

from SyMBac.napari.ui.bindings import (
    get_export_widget,
    get_optics_widget,
    get_regions_widget,
    get_simulation_widget,
    get_tuning_widget,
)


__all__ = [
    "create_simulation_dock",
    "create_optics_dock",
    "create_regions_dock",
    "create_tuning_dock",
    "create_export_dock",
]


def create_simulation_dock(viewer):
    return get_simulation_widget(viewer)


def create_optics_dock(viewer):
    return get_optics_widget(viewer)


def create_regions_dock(viewer):
    return get_regions_widget(viewer)


def create_tuning_dock(viewer):
    return get_tuning_widget(viewer)


def create_export_dock(viewer):
    return get_export_widget(viewer)
