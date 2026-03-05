import SyMBac.napari.ui.bindings as bindings


class _Window:
    def __init__(self):
        self.calls = []

    def add_dock_widget(self, widget, name, area):
        self.calls.append((widget, name, area))


class _Viewer:
    def __init__(self):
        self.window = _Window()


def test_register_default_docks_adds_single_workflow(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(bindings, "get_workflow_widget", lambda _viewer: sentinel)

    viewer = _Viewer()
    bindings.register_default_docks(viewer)

    assert viewer.window.calls == [(sentinel, "SyMBac Workflow", "right")]
