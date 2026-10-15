"""Reproduce a synthetic-only baseline vs. YOLO experiment on a held-out split.

Run from the repository root after installing .[train]. No pretrained download.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ndt.dataset import validate_dataset
from ndt.detection import CLASSES, Detection, candidates, load_model, predict
from ndt.evaluation import evaluate_pairs
from ndt.imaging import load_image
from ndt.provenance import environment_report
from ndt.synthetic import make_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/experiment"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty; use a new directory to preserve previous experiments.")
    config = make_dataset(args.output / "dataset", count=args.count, seed=args.seed, size=args.size)
    counts = validate_dataset(config)
    from ultralytics import YOLO

    started = datetime.now(timezone.utc)
    model = YOLO("yolo11n.yaml")
    model.train(
        data=str(config.resolve()),
        epochs=args.epochs,
        imgsz=args.size,
        batch=8,
        workers=0,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        project=str((args.output / "training").resolve()),
        name="ndt",
        hsv_h=0,
        hsv_s=0,
        hsv_v=0.1,
        fliplr=0.5,
        flipud=0,
        degrees=3,
        mosaic=0,
        mixup=0,
    )
    weights = Path(model.trainer.best)
    learned = load_model(weights)
    official = learned.val(
        data=str(config.resolve()),
        split="test",
        imgsz=args.size,
        device=args.device,
        workers=0,
        project=str((args.output / "evaluation").resolve()),
        name="test",
    )
    heuristic_pairs, yolo_pairs = [], []
    for path in sorted((config.parent / "images/test").glob("*.png")):
        image = load_image(path)
        expected = []
        for row in (config.parent / "labels/test" / f"{path.stem}.txt").read_text().splitlines():
            class_id, cx, cy, w, h = map(float, row.split())
            expected.append(
                Detection(
                    CLASSES[int(class_id)],
                    (
                        (cx - w / 2) * image.width,
                        (cy - h / 2) * image.height,
                        (cx + w / 2) * image.width,
                        (cy + h / 2) * image.height,
                    ),
                )
            )
        heuristic_pairs.append((expected, candidates(image)))
        yolo_pairs.append((expected, predict(learned, image, confidence=0.25, imgsz=args.size)))
    payload = dict(
        synthetic_only=True,
        note="Procedural phantoms; results do not establish industrial defect performance.",
        started_utc=started.isoformat(),
        duration_seconds=(datetime.now(timezone.utc) - started).total_seconds(),
        seed=args.seed,
        initialization="yolo11n.yaml (random initialization, no pretrained download)",
        epochs=args.epochs,
        image_size=args.size,
        batch=8,
        device=args.device,
        split_counts=counts,
        environment=environment_report(),
        confidence_threshold=0.25,
        heuristic=evaluate_pairs(heuristic_pairs),
        yolo=evaluate_pairs(yolo_pairs),
        official_yolo_test_metrics={k: float(v) for k, v in official.results_dict.items()},
    )
    destination = args.output / "summary.json"
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
