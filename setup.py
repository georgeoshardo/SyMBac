from setuptools import setup


setup(
    entry_points={
        "napari.manifest": [
            "SyMBac = SyMBac.napari.resources:napari.yaml",
        ],
    },
)
