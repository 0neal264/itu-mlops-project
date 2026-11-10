from pathlib import Path

import torch

from vial_classifier.config import DATA_DIR, MODELS_DIR, PROJ_ROOT
from vial_classifier.features import calculate_image_statistics, count_classes


def test_project_root():
    expected_root = Path(__file__).resolve().parents[1]
    assert PROJ_ROOT == expected_root


def test_project_paths():
    assert DATA_DIR == PROJ_ROOT / "data"
    assert MODELS_DIR == PROJ_ROOT / "models"


def test_count_classes():
    assert count_classes([0, 1, 1, 0, 1]) == {0: 2, 1: 3}


def test_calculate_image_statistics():
    images = torch.tensor([
        [[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]]],
        [[[4.0, 4.0]], [[5.0, 5.0]], [[6.0, 6.0]]],
    ])

    statistics = calculate_image_statistics([(images, torch.tensor([0, 1]))])

    assert statistics == {
        "mean": [2.5, 3.5, 4.5],
        "std": [0.0, 0.0, 0.0],
        "total_samples": 2,
    }