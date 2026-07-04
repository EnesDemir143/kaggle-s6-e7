import numpy as np
import pandas as pd

from kaggle_s6_e7.features import (add_categorical_interactions, add_missing_features,
                                    add_outlier_flags, add_ratio_features, fit_outlier_bounds)


def feature_frame():
    return pd.DataFrame({"calorie_expenditure": [100.0, 200.0, 300.0], "step_count": [0.0, np.nan, 10.0],
                         "exercise_duration": [0.0, 10.0, 20.0], "water_intake": [1.0, 2.0, np.nan],
                         "bmi": [20.0, 25.0, 30.0], "sleep_duration": [5.0, 7.0, 9.0],
                         "stress_level": ["low", None, "high"], "sleep_quality": ["good", "average", None],
                         "physical_activity_level": ["active", "moderate", "sedentary"],
                         "diet_type": ["veg", None, "non-veg"], "smoking_alcohol": ["no", "yes", None],
                         "gender": ["female", "male", None], "heart_rate": [55.0, 75.0, 105.0]})


def test_feature_builders_do_not_mutate_and_ratios_are_finite():
    original = feature_frame()
    result = add_ratio_features(original)
    assert set(result) > set(original)
    assert not np.isinf(result.select_dtypes(include="number").to_numpy()).any()
    assert list(original.columns) == list(feature_frame().columns)


def test_missing_and_interaction_features_are_stable():
    df = feature_frame()
    missing = add_missing_features(df, ["step_count", "water_intake"])
    interactions = add_categorical_interactions(df)
    assert missing.missing_count.tolist() == [0, 1, 1]
    assert "missing" in interactions.loc[1, "activity_diet"]


def test_outlier_bounds_are_reused_for_test():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    test = pd.DataFrame({"x": [-100.0, 100.0]})
    bounds = fit_outlier_bounds(train, ["x"], 0.25, 0.75)
    flagged = add_outlier_flags(test, bounds)
    assert bounds["x"] == (0.75, 2.25)
    assert flagged.outlier_count.tolist() == [1, 1]
