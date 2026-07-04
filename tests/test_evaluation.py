import numpy as np

from kaggle_s6_e7.evaluation import (
    classification_metrics,
    plot_multiclass_roc,
    predictions_from_probabilities,
)


def test_multiclass_metrics_and_roc_artifact(tmp_path):
    labels = ["at-risk", "fit", "unhealthy"]
    y_true = ["at-risk", "fit", "unhealthy", "at-risk", "fit", "unhealthy"]
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.1, 0.7],
        ]
    )
    predictions = predictions_from_probabilities(probabilities, labels)
    metrics = classification_metrics(y_true, predictions, probabilities, labels)
    output = plot_multiclass_roc(y_true, probabilities, labels, tmp_path / "roc.png")

    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["f1_weighted"] == 1.0
    assert metrics["class_recall"]["fit"] == 1.0
    assert output.is_file()


def test_class_multipliers_change_decision():
    probabilities = np.array([[0.45, 0.40, 0.15]])
    labels = ["at-risk", "fit", "unhealthy"]
    prediction = predictions_from_probabilities(probabilities, labels, [1.0, 1.2, 1.0])
    assert prediction.tolist() == ["fit"]
