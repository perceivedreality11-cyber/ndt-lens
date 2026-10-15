"""Small experiment manifests without environment variables or user paths."""

import platform
from importlib.metadata import PackageNotFoundError, version


def environment_report():
    packages = {}
    for name in (
        "ndt-lens",
        "numpy",
        "Pillow",
        "opencv-python-headless",
        "PyYAML",
        "streamlit",
        "torch",
        "ultralytics",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(
        python=platform.python_version(),
        system=platform.system(),
        machine=platform.machine(),
        packages=packages,
    )
