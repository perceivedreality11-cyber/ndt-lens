"""Durable, local annotation workspace and reviewed-only YOLO snapshots."""

import hashlib
import io
import json
import math
import shutil
import sqlite3
import uuid
from dataclasses import asdict
from pathlib import Path

import yaml

from ndt.dataset import validate_dataset
from ndt.detection import CLASSES, Detection
from ndt.imaging import load_image


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, width INTEGER, height INTEGER,
                group_id TEXT NOT NULL, split TEXT NOT NULL, reviewed INTEGER DEFAULT 0,
                boxes TEXT DEFAULT '[]', model TEXT, revision INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")

    def connect(self):
        db = sqlite3.connect(self.root / "workspace.sqlite", timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def metadata(self):
        path = self.root / "workspace.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"synthetic": False}

    def images(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM images ORDER BY created_at, id")]

    def get(self, image_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
        if row is None:
            raise ValueError("图片不存在。")
        return dict(row)

    def image_path(self, image_id):
        self.get(image_id)
        return self.root / "images" / f"{image_id}.png"

    def import_image(self, name: str, content: bytes, group: str, split: str):
        if not group.strip() or split not in ("train", "val", "test"):
            raise ValueError("请填写工件/底片组号，并选择数据集。")
        image = load_image(io.BytesIO(content))
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM images WHERE id=?", (digest,)).fetchone():
                return digest, False
            conflict = db.execute(
                "SELECT 1 FROM images WHERE group_id=? AND split!=?", (group.strip(), split)
            ).fetchone()
            if conflict:
                raise ValueError("同一工件/底片组不能跨 train、val、test，请更换组号或数据集。")
            for folder in ("images", "originals"):
                (self.root / folder).mkdir(exist_ok=True)
            image.save(self.root / "images" / f"{digest}.png")
            # Uploaded names never become paths; preserve original bytes separately.
            (self.root / "originals" / f"{digest}.bin").write_bytes(content)
            db.execute(
                "INSERT INTO images(id,name,width,height,group_id,split) VALUES(?,?,?,?,?,?)",
                (digest, Path(name).name, image.width, image.height, group.strip(), split),
            )
        return digest, True

    def save_boxes(self, image_id, boxes, *, reviewed=False, model=None, expected_revision=None):
        row = self.get(image_id)
        cleaned = []
        for box in boxes:
            try:
                label = box["label"]
                coords = tuple(float(v) for v in box["xyxy"])
                confidence = box.get("confidence")
                confidence = None if confidence is None else float(confidence)
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError("请补全每行的缺陷类别和四个数值坐标，或删除空行。") from exc
            if label not in CLASSES or len(coords) != 4 or not all(math.isfinite(v) for v in coords):
                raise ValueError("标注类别或坐标无效。")
            x1, y1, x2, y2 = coords
            if not (0 <= x1 < x2 <= row["width"] and 0 <= y1 < y2 <= row["height"]):
                raise ValueError("标注框必须位于图内且具有正面积。")
            if confidence is not None and not (math.isfinite(confidence) and 0 <= confidence <= 1):
                raise ValueError("置信度无效。")
            cleaned.append({"label": label, "xyxy": list(coords), "confidence": confidence})
        revision = row["revision"] if expected_revision is None else expected_revision
        with self.connect() as db:
            changed = db.execute(
                "UPDATE images SET boxes=?,reviewed=?,model=?,revision=revision+1 WHERE id=? AND revision=?",
                (json.dumps(cleaned, allow_nan=False), int(reviewed), model, image_id, revision),
            ).rowcount
            if not changed:
                raise ValueError("标注已在另一页面更新，请刷新后重新编辑。")

    def auto_label(self, image_ids, model, model_path, confidence=0.25, progress=None):
        from ndt.detection import predict

        updated = 0
        for index, image_id in enumerate(image_ids):
            row = self.get(image_id)
            # Reviewed annotations, including confirmed negatives, are never overwritten.
            if not row["reviewed"]:
                boxes = predict(model, load_image(self.image_path(image_id)), confidence)
                self.save_boxes(
                    image_id,
                    [asdict(box) for box in boxes],
                    model=str(model_path),
                    expected_revision=row["revision"],
                )
                updated += 1
            if progress:
                progress((index + 1) / len(image_ids))
        return updated

    def snapshot(self):
        rows = [row for row in self.images() if row["reviewed"]]
        for split in ("train", "val"):
            if not any(row["split"] == split for row in rows):
                raise ValueError(f"请至少复核一张 {split} 图片；train 和 val 必须来自不同工件组。")
        destination = self.root / "snapshots" / uuid.uuid4().hex
        destination.mkdir(parents=True)
        try:
            for row in rows:
                split = row["split"]
                images, labels = destination / "images" / split, destination / "labels" / split
                images.mkdir(parents=True, exist_ok=True)
                labels.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.image_path(row["id"]), images / f"{row['id']}.png")
                lines = []
                for box in json.loads(row["boxes"]):
                    x1, y1, x2, y2 = box["xyxy"]
                    lines.append(
                        f"{CLASSES.index(box['label'])} {(x1 + x2) / 2 / row['width']:.9f} "
                        f"{(y1 + y2) / 2 / row['height']:.9f} {(x2 - x1) / row['width']:.9f} "
                        f"{(y2 - y1) / row['height']:.9f}"
                    )
                (labels / f"{row['id']}.txt").write_text("\n".join(lines), encoding="utf-8")
            config = {
                "path": str(destination),
                "train": "images/train",
                "val": "images/val",
                "names": CLASSES,
            }
            if any(row["split"] == "test" for row in rows):
                config["test"] = "images/test"
            path = destination / "dataset.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            (destination / "manifest.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (destination / "workspace.json").write_text(
                json.dumps(self.metadata(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            validate_dataset(path)
            return path
        except Exception:
            shutil.rmtree(destination)
            raise


def detections(row):
    return [
        Detection(item["label"], tuple(item["xyxy"]), item.get("confidence"))
        for item in json.loads(row["boxes"])
    ]
