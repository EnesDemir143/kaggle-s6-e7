import numpy as np
import pandas as pd

from kaggle_s6_e7.preprocessing import (
    INTERACTION_COLS,
    RAW_CATEGORICAL_COLS,
    RAW_NUMERIC_COLS,
    RATIO_COLS,
    V2CorePreprocessor,
    build_one_hot_encoder,
)


def frame() -> pd.DataFrame:
    rows = 8
    data = {col: np.arange(1, rows + 1, dtype=float) for col in RAW_NUMERIC_COLS}
    data.update({col: ["a", "b"] * 4 for col in RAW_CATEGORICAL_COLS})
    result = pd.DataFrame(data)
    result.loc[0, "bmi"] = np.nan
    result.loc[1, "diet_type"] = None
    return result


def test_v2_core_contract_and_fit_only_statistics():
    train = frame()
    validation = frame()
    validation.loc[0, "bmi"] = 10_000
    processor = V2CorePreprocessor(lower_q=0.1, upper_q=0.9).fit(train)
    result = processor.transform(validation)

    assert result.loc[0, "bmi"] == 10_000
    assert result.loc[0, "bmi_outlier_high"] == 1
    assert result.loc[0, "bmi_is_missing"] == 0
    assert "gender_activity" not in result
    assert set(INTERACTION_COLS + RATIO_COLS).issubset(result.columns)
    assert not result.select_dtypes(include="number").isna().any().any()


def test_missing_values_and_unknown_categories_are_supported():
    train = frame()
    validation = frame()
    validation.loc[0, "gender"] = "unseen"
    processor = V2CorePreprocessor().fit(train)
    transformed_train = processor.transform(train)
    transformed_validation = processor.transform(validation)
    numeric = [
        col for col in transformed_train if col not in processor.categorical_features_
    ]
    encoder = build_one_hot_encoder(processor.categorical_features_, numeric)
    encoder.fit(transformed_train)
    matrix = encoder.transform(transformed_validation)

    assert transformed_train.loc[0, "bmi_is_missing"] == 1
    assert transformed_train.loc[1, "diet_type"] == "missing"
    assert matrix.shape[0] == len(validation)
