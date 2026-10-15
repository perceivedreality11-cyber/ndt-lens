import io
import json

import pytest
from streamlit.testing.v1 import AppTest

from ndt.dataset import validate_dataset
from ndt.detection import Detection
from ndt.synthetic import sample
from ndt.workspace import Workspace


def content(seed):
    buffer = io.BytesIO()
    image, boxes = sample(seed, size=128)
    image.save(buffer, format="PNG")
    return buffer.getvalue(), [{"label": b.label, "xyxy": list(b.xyxy)} for b in boxes]


def test_import_deduplicates_pixels_and_keeps_original(tmp_path):
    workspace = Workspace(tmp_path)
    data, _ = content(1)
    image_id, created = workspace.import_image("../../weld.png", data, "weld-1", "train")
    assert created
    assert workspace.import_image("copy.png", data, "weld-2", "val") == (image_id, False)
    assert len(workspace.images()) == 1
    assert (tmp_path / "originals" / f"{image_id}.bin").read_bytes() == data
    with pytest.raises(ValueError, match="同一工件"):
        workspace.import_image("b.png", content(2)[0], "weld-1", "val")


def test_reviewed_only_snapshot_and_revision_conflict(tmp_path):
    workspace = Workspace(tmp_path)
    for index, split in enumerate(["train", "val", "test"]):
        data, boxes = content(index)
        image_id, _ = workspace.import_image(f"{index}.png", data, str(index), split)
        workspace.save_boxes(image_id, boxes, reviewed=index < 2)
    config = workspace.snapshot()
    counts = validate_dataset(config)
    assert set(counts) == {"train", "val"}
    manifest = json.loads((config.parent / "manifest.json").read_text())
    assert all(row["reviewed"] for row in manifest)
    with pytest.raises(ValueError, match="另一页面"):
        workspace.save_boxes(image_id, [], expected_revision=0)


def test_no_training_without_review_and_invalid_boxes(tmp_path):
    workspace = Workspace(tmp_path)
    image_id, _ = workspace.import_image("one.png", content(1)[0], "1", "train")
    with pytest.raises(ValueError, match="复核"):
        workspace.snapshot()
    for coords in [[0, 0, 129, 128], [0, 0, 0, 2], [0, 0, float("nan"), 20], [None, 0, 10, 20]]:
        with pytest.raises(ValueError):
            workspace.save_boxes(image_id, [{"label": "crack", "xyxy": coords}])
    assert not workspace.get(image_id)["reviewed"]


def test_auto_label_preserves_reviewed_including_negatives(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path)
    image_ids = [workspace.import_image(f"{i}.png", content(i)[0], str(i), "train")[0] for i in range(3)]
    workspace.save_boxes(image_ids[0], [], reviewed=True)
    monkeypatch.setattr("ndt.detection.predict", lambda *args: [Detection("crack", (1, 2, 10, 20), 0.8)])
    assert workspace.auto_label(image_ids, object(), "best.pt") == 2
    assert workspace.get(image_ids[0])["boxes"] == "[]"
    assert workspace.get(image_ids[0])["reviewed"]
    assert not workspace.get(image_ids[1])["reviewed"]
    assert workspace.get(image_ids[1])["model"] == "best.pt"


def test_new_pages_and_persisted_annotation(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    workspace = Workspace("data/workbench")
    image_id, _ = workspace.import_image("weld.png", content(1)[0], "weld-1", "train")
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value == "1"
    # An empty annotation becomes a confirmed negative only after explicit save.
    next(b for b in app.button if b.label == "保存并确认标注").click().run()
    assert not app.exception
    assert workspace.get(image_id)["reviewed"]
    for page in ["训练中心", "批量自动标注"]:
        app.sidebar.selectbox[0].set_value(page).run()
        assert not app.exception
    next(b for b in app.button if b.label == "生成标注结果下载包").click().run()
    assert not app.exception
