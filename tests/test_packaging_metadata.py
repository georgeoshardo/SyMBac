import re
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def _project_metadata():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject:
        return tomllib.load(pyproject)["project"]


def test_python_metadata_matches_supported_versions():
    metadata = _project_metadata()
    python_classifiers = {
        classifier
        for classifier in metadata["classifiers"]
        if classifier.startswith("Programming Language :: Python :: 3.")
    }

    assert metadata["requires-python"] == ">=3.11"
    assert python_classifiers == {
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    }


def test_runtime_dependencies_are_declared_directly():
    dependencies = _project_metadata()["dependencies"]
    dependency_names = {
        re.split(r"[<>=!~;\[]", dependency, maxsplit=1)[0].lower()
        for dependency in dependencies
    }

    assert {"numpy", "numba", "pillow", "networkx"}.issubset(dependency_names)
    assert not any(
        dependency.startswith(("ipython", "zarr", "optuna", "cmaes"))
        for dependency in dependencies
    )
