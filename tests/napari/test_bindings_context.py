import SyMBac.napari.ui.bindings as bindings

from SyMBac.napari.ui.bindings import get_or_create_context


class _NoNewAttrsViewer:
    __slots__ = ("layers",)

    def __init__(self):
        self.layers = []


def test_get_or_create_context_without_setting_viewer_attributes():
    viewer = _NoNewAttrsViewer()
    ctx1 = get_or_create_context(viewer)
    ctx2 = get_or_create_context(viewer)

    assert ctx1 is ctx2


def test_get_simulation_widget_builds_fresh_widget_instances(monkeypatch):
    class _FakeDock:
        def __init__(self, *_args, **_kwargs):
            self.widget = object()

    monkeypatch.setattr(bindings, "SimulationDock", _FakeDock)

    viewer = _NoNewAttrsViewer()
    widget_1 = bindings.get_simulation_widget(viewer)
    widget_2 = bindings.get_simulation_widget(viewer)

    assert widget_1 is not widget_2
