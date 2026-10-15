import pytest

from ndt.detection import Detection
from ndt.evaluation import evaluate_pairs, intersection_over_union


def test_iou_touching_edges_and_degenerate_boxes():
    assert intersection_over_union((0, 0, 10, 10), (10, 0, 20, 10)) == 0
    assert intersection_over_union((0, 0, 0, 0), (0, 0, 0, 0)) == 0
    assert intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1


def test_duplicate_predictions_wrong_classes_and_negative_images_are_counted():
    expected = Detection("crack", (0, 0, 10, 10))
    duplicate = Detection("crack", (0, 0, 10, 10), 0.4)
    correct = Detection("crack", (0, 0, 10, 10), 0.9)
    wrong_class = Detection("porosity", (0, 0, 10, 10), 0.8)
    result = evaluate_pairs(
        [
            ([expected], [duplicate, correct, wrong_class]),
            ([expected], []),
            ([], [correct]),
        ]
    )
    assert result["micro"] == dict(tp=1, fp=3, fn=1, precision=0.25, recall=0.5, f1=pytest.approx(1 / 3))
    assert result["per_class"]["porosity"]["fp"] == 1


def test_empty_evaluation_has_defined_zero_denominators():
    assert evaluate_pairs([])["micro"]["f1"] == 0
    with pytest.raises(ValueError):
        evaluate_pairs([], iou_threshold=0)
