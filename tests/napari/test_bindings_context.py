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
