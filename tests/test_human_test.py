import pytest

from human_test import compute_scores


def test_human_scores_keep_last_answer_and_split_by_source() -> None:
    manifest = {
        "items": [
            {"id": "a", "source": "gan", "class_id": 0},
            {"id": "b", "source": "gan", "class_id": 1},
            {"id": "c", "source": "real", "class_id": 2},
        ]
    }
    answers = [
        {"participant": "p1", "image_id": "a", "answer": "generated"},
        {"participant": "p1", "image_id": "a", "answer": "real"},  # corrected
        {"participant": "p1", "image_id": "b", "answer": "generated"},
        {"participant": "p1", "image_id": "c", "answer": "real"},
        {"participant": "p2", "image_id": "a", "answer": "real"},
        {"participant": "p2", "image_id": "unknown", "answer": "real"},
    ]

    scores = compute_scores(manifest, answers)

    assert scores["participants"] == 2
    assert scores["answers"] == 4
    assert scores["per_source"]["gan"]["judged_real_rate"] == pytest.approx(2 / 3)
    assert scores["per_source"]["gan"]["participants"] == 2
    assert scores["per_source"]["real"]["judged_real_rate"] == 1.0
    assert scores["human_accuracy"] == pytest.approx(2 / 4)
