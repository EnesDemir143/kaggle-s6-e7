import json

import numpy as np
import pandas as pd
import pytest

from kaggle_s6_e7.cache import FoldFeatureCache, load_cached_csv
from kaggle_s6_e7.experiment_config import resolve_experiment
from kaggle_s6_e7.postprocess import tune_class_multipliers
from kaggle_s6_e7.preprocessing import FoldPreprocessor
from kaggle_s6_e7.training import normalize_probabilities, sample_weights


def feature_frame(rows: int = 12) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sleep_duration": np.linspace(4, 10, rows),
            "heart_rate": np.linspace(50, 110, rows),
            "bmi": np.linspace(17, 35, rows),
            "calorie_expenditure": np.linspace(100, 1000, rows),
            "step_count": np.linspace(1000, 15000, rows),
            "exercise_duration": np.linspace(0, 90, rows),
            "water_intake": np.linspace(1, 4, rows),
            "diet_type": ["balanced", "vegan"] * (rows // 2),
            "stress_level": ["low", "high"] * (rows // 2),
            "sleep_quality": ["good", "poor"] * (rows // 2),
            "physical_activity_level": ["active", "sedentary"] * (rows // 2),
            "smoking_alcohol": ["no", "yes"] * (rows // 2),
            "gender": ["female", "male"] * (rows // 2),
        }
    )
    frame.loc[0, "bmi"] = np.nan
    return frame


def test_config_inheritance_resolves_single_ablation():
    e002 = resolve_experiment("E002")
    e003 = resolve_experiment("E003")
    assert e002["features"]["gender_activity"] is False
    assert e003["features"]["gender_activity"] is True
    assert e003["features"]["ratios"] is True


def test_fold_preprocessor_handles_unknown_and_clipping():
    config = resolve_experiment("E005")["features"]
    train = feature_frame()
    valid = feature_frame()
    valid.loc[1, "gender"] = "unseen"
    valid.loc[1, "bmi"] = 10_000
    processor = FoldPreprocessor(config).fit(train)
    result = processor.transform(valid)
    assert str(result.loc[1, "gender"]) == "__UNKNOWN__"
    assert result.loc[1, "bmi_outlier_high"] == 1
    assert result.loc[1, "bmi"] < 10_000
    assert result.loc[0, "bmi_is_missing"] == 1


def test_parquet_cache_round_trip_and_manifest_rejection(tmp_path):
    cache = FoldFeatureCache(tmp_path)
    key = cache.key(data={"sha": "x"}, fold=0, config={"ratios": True})
    frame = pd.DataFrame({"a": [1, 2]})
    cache.save(key, frame, frame, frame)
    loaded = cache.load(key)
    assert loaded is not None and loaded[0].equals(frame)
    manifest = tmp_path / "folds" / key / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["frames"]["train"]["rows"] = 999
    manifest.write_text(json.dumps(payload))
    assert cache.load(key) is None


def test_raw_cache_and_probability_normalization(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(source, index=False)
    assert load_cached_csv(source, tmp_path / "raw").x.tolist() == [1, 2]
    normalized = normalize_probabilities(np.array([[0.2, 0.2, 0.2]]))
    assert normalized.sum(axis=1) == pytest.approx([1.0])


def test_weights_and_multiplier_search_are_deterministic():
    y = pd.Series(["at-risk"] * 6 + ["fit"] * 2 + ["unhealthy"])
    weights = sample_weights(y, "sqrt_balanced")
    assert weights is not None and weights[-1] > weights[0]
    proba = np.array([[0.6, 0.3, 0.1], [0.4, 0.5, 0.1], [0.4, 0.1, 0.5]])
    labels = np.array(["at-risk", "fit", "unhealthy"])
    first, _ = tune_class_multipliers(labels, proba, random_trials=10)
    second, _ = tune_class_multipliers(labels, proba, random_trials=10)
    assert first == pytest.approx(second)
