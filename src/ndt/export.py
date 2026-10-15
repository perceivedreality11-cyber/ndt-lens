"""Portable image/label exports with review and prediction provenance."""

import io
import json
import zipfile

import yaml

from ndt.detection import CLASSES, annotate, as_json, report
from ndt.imaging import load_image
from ndt.workspace import detections


def yolo_labels(boxes, width, height):
    lines = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        lines.append(
            f"{CLASSES.index(box.label)} {(x1 + x2) / 2 / width:.9f} {(y1 + y2) / 2 / height:.9f} "
            f"{(x2 - x1) / width:.9f} {(y2 - y1) / height:.9f}"
        )
    return "\n".join(lines)


def export_workspace(workspace, *, reviewed_only=True):
    rows = [r for r in workspace.images() if r["reviewed"] or not reviewed_only]
    if not rows:
        raise ValueError("当前范围没有可导出的图片，请先确认标注或选择包含待复核结果。")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        manifest = []
        for row in rows:
            path = workspace.image_path(row["id"])
            image = load_image(path)
            boxes = detections(row)
            png = io.BytesIO()
            annotate(image, boxes).save(png, format="PNG")
            stem = f"{row['split']}/{row['id']}"
            archive.write(path, f"images/{stem}.png")
            archive.writestr(f"annotated/{stem}.png", png.getvalue())
            archive.writestr(f"labels/{stem}.txt", yolo_labels(boxes, image.width, image.height))
            payload = report(boxes, image, engine="yolo" if row["model"] else "manual", source=row["name"])
            payload.update(
                reviewed=bool(row["reviewed"]), model=row["model"], group=row["group_id"], split=row["split"]
            )
            archive.writestr(f"reports/{stem}.json", as_json(payload))
            manifest.append(
                {k: row[k] for k in ("id", "name", "group_id", "split", "reviewed", "model", "revision")}
            )
        config = dict(path=".", train="images/train", val="images/val", names=CLASSES)
        if any(r["split"] == "test" for r in rows):
            config["test"] = "images/test"
        archive.writestr("dataset.yaml", yaml.safe_dump(config))
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr("workspace.json", json.dumps(workspace.metadata(), ensure_ascii=False, indent=2))
        archive.writestr(
            "README.txt",
            (
                "NDT Lens annotation export\n"
                f"Reviewed only: {reviewed_only}\n"
                "images/ contains normalized images without annotation overlays.\n"
                "annotated/ contains visualization only; never use these overlays for training.\n"
                "Review manifest.json before training; unreviewed predictions are not ground truth.\n"
                "Training requires populated train/val splits and positive training boxes.\n"
                "Run: ndt check-data --data dataset.yaml\n"
            ),
        )
    return buffer.getvalue()
