from __future__ import annotations


def launch_napari(initial_real_image=None):
    import napari

    from SyMBac.napari.ui.bindings import get_or_create_context, register_default_docks

    viewer = napari.Viewer(title="SyMBac Napari")
    register_default_docks(viewer)

    if initial_real_image is not None:
        context = get_or_create_context(viewer)
        context.controller.set_real_image(initial_real_image)
        context.layer_manager.update_real_image(initial_real_image)

    return viewer


def main() -> None:
    import napari

    launch_napari()
    napari.run()
