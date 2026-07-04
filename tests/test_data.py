import pandas as pd
import pytest

from kaggle_s6_e7.data import infer_feature_columns, validate_schema


def frames():
    train = pd.DataFrame({"id": [1, 2, 3], "health_condition": ["fit", "at-risk", "unhealthy"],
                          "numeric": [1.0, 2.0, 3.0], "category": ["a", "b", "a"]})
    return train, train.drop(columns="health_condition")


def test_schema_and_column_inference():
    train, test = frames()
    validate_schema(train, test)
    assert infer_feature_columns(train) == (["category"], ["numeric"])


def test_schema_rejects_target_in_test():
    train, _ = frames()
    with pytest.raises(ValueError, match="must not contain"):
        validate_schema(train, train.copy())
