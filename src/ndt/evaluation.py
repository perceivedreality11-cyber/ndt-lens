"""Class-aware, one-to-one matching for a fixed operating-point comparison."""

from ndt.detection import CLASSES


def intersection_over_union(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def evaluate_pairs(pairs, iou_threshold=0.5):
    """pairs: (ground-truth boxes, predictions) for each independent image.

    This reports precision/recall at one threshold, not AP or classification
    accuracy. Duplicate predictions count as false positives, including when
    there are no ground-truth boxes in a negative image.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1].")
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASSES}
    image_count = 0
    for expected, predicted in pairs:
        image_count += 1
        matched = set()
        predicted = sorted(
            predicted, key=lambda b: b.confidence if b.confidence is not None else 0, reverse=True
        )
        for prediction in predicted:
            eligible = [
                (intersection_over_union(prediction.xyxy, box.xyxy), index)
                for index, box in enumerate(expected)
                if box.label == prediction.label and index not in matched
            ]
            overlap, index = max(eligible, default=(0, -1))
            if index >= 0 and overlap >= iou_threshold:
                matched.add(index)
                counts[prediction.label]["tp"] += 1
            else:
                counts[prediction.label]["fp"] += 1
        for index, box in enumerate(expected):
            if index not in matched:
                counts[box.label]["fn"] += 1

    def metrics(values):
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        return {
            **values,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }

    total = {key: sum(item[key] for item in counts.values()) for key in ("tp", "fp", "fn")}
    return dict(
        images=image_count,
        iou_threshold=iou_threshold,
        micro=metrics(total),
        per_class={name: metrics(values) for name, values in counts.items()},
    )
