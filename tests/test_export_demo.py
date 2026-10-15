import io
import json
import zipfile

import pytest

from ndt.dataset import validate_dataset
from ndt.demo import create_demo_workspace
from ndt.export import export_workspace
from ndt.workspace import Workspace


def test_demo_is_repeatable_has_independent_splits_and_does_not_reset_edits(tmp_path):
    workspace = create_demo_workspace(tmp_path)
    assert len(workspace.images()) == 15
    assert sum(row["reviewed"] for row in workspace.images()) == 12
    row = next(r for r in workspace.images() if r["reviewed"])
    workspace.save_boxes(row["id"], [], reviewed=False)
    again = create_demo_workspace(tmp_path)
    assert again.get(row["id"])["reviewed"] == 0
    assert again.metadata()["synthetic"]


def test_export_roundtrip_is_trainable_and_excludes_unreviewed_predictions(tmp_path):
    workspace = create_demo_workspace(tmp_path / "source")
    payload = export_workspace(workspace)
    destination = tmp_path / "unpacked"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(destination)
        manifest = json.loads(archive.read("manifest.json"))
        assert len(manifest) == 12
        assert all(row["reviewed"] for row in manifest)
        assert json.loads(archive.read("workspace.json"))["synthetic"]
    counts = validate_dataset(destination / "dataset.yaml")
    assert {key: value["images"] for key, value in counts.items()} == {"train": 8, "val": 2, "test": 2}
    for row in manifest:
        assert (
            destination / "images" / row["split"] / f"{row['id']}.png"
        ).read_bytes() == workspace.image_path(row["id"]).read_bytes()
    with zipfile.ZipFile(io.BytesIO(export_workspace(workspace, reviewed_only=False))) as archive:
        assert len(json.loads(archive.read("manifest.json"))) == 15


def test_empty_export_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="没有可导出"):
        export_workspace(Workspace(tmp_path))
