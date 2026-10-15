"""A reproducible reviewer walkthrough with explicit synthetic provenance."""

import io
import json
from dataclasses import asdict
from pathlib import Path

from ndt.synthetic import sample
from ndt.workspace import Workspace


def create_demo_workspace(root: str | Path) -> Workspace:
    workspace = Workspace(root)
    metadata_path = workspace.root / "workspace.json"
    if workspace.images() and not metadata_path.is_file():
        raise ValueError("演示目录中已有非演示数据，请指定新的目录。")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    if metadata and metadata.get("kind") != "synthetic-demo":
        raise ValueError("该目录不是演示工作空间。")
    if metadata.get("complete"):
        return workspace
    metadata = dict(
        kind="synthetic-demo",
        synthetic=True,
        seed=4200,
        complete=False,
        note="Procedural phantoms; confirmed labels are generated ground truth, not human review.",
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for index in range(15):
        image, boxes = sample(4200 + index, size=512, clean=index % 5 == 0)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        split = "train" if index < 8 or index >= 12 else ("val" if index < 10 else "test")
        image_id, created = workspace.import_image(
            f"SYNTHETIC_{index + 1:03d}.png", buffer.getvalue(), f"phantom-{index + 1:03d}", split
        )
        if created and index < 12:
            workspace.save_boxes(image_id, [asdict(box) for box in boxes], reviewed=True)
    metadata["complete"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return workspace
