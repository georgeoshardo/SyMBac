"""SyMBac package entrypoint."""


def launch_napari(*args, **kwargs):
    """Launch the SyMBac napari workflow UI."""
    from SyMBac.napari.app import launch_napari as _launch_napari

    return _launch_napari(*args, **kwargs)


__all__ = ["launch_napari"]
