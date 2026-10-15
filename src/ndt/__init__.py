"""Industrial X-ray inspection research starter."""

import os
from pathlib import Path

# Keep model-library settings with the project, including on restricted machines.
os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / "outputs" / "yolo-config").resolve()))

__version__ = "0.3.0"
