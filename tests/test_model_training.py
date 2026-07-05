import numpy as np
import pandas as pd
import pytest

from kaggle_s6_e7.model_adapters import categorical_columns, create_model_adapter


def frame(rows: int = 60, *, categorical: bool = True) -> pd.DataFrame:
    data: dict[str, object] = {
        "numeric": np.linspace(0, 1, rows),
        "numeric_2": np.arange(rows) % 7,
    }
    if categorical:
        data["category"] = pd.Categorical(["a", "b", "c"] * (rows // 3))
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    ("model", "params"),
    [
        (
            "xgboost",
            {
                "objective": "multi:softprob",
                "num_class": 3,
                "n_estimators": 8,
                "max_depth": 2,
                "learning_rate": 0.2,
                "early_stopping_rounds": 3,
                "n_jobs": 1,
                "random_state": 42,
            },
        ),
        (
            "catboost",
            {
                "loss_function": "MultiClass",
                "iterations": 8,
                "depth": 2,
                "learning_rate": 0.2,
                "od_wait": 3,
                "random_seed": 42,
                "allow_writing_files": False,
                "verbose": False,
            },
        ),
    ],
)
def test_diversity_adapters_fit_normalized_probabilities(model, params):
    X = frame()
    y = np.tile(np.arange(3), 20)
    adapter = create_model_adapter(model, params)
    adapter.fit(X.iloc[:45], y[:45], X.iloc[45:], y[45:], None)
    probabilities = adapter.predict_proba(X.iloc[45:])
    assert probabilities.shape == (15, 3)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(15), abs=1e-6)
    assert adapter.best_iteration() > 0


def test_catboost_input_mode_follows_actual_dtypes():
    params = {
        "loss_function": "MultiClass",
        "iterations": 3,
        "depth": 2,
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": False,
    }
    y = np.tile(np.arange(3), 20)
    native = create_model_adapter("catboost", params)
    native.fit(frame().iloc[:45], y[:45], frame().iloc[45:], y[45:], None)
    assert native.input_mode == "native_categorical"
    numeric = create_model_adapter("catboost", params)
    numeric.fit(
        frame(categorical=False).iloc[:45],
        y[:45],
        frame(categorical=False).iloc[45:],
        y[45:],
        None,
    )
    assert numeric.input_mode == "numeric_only"
    assert categorical_columns(frame(categorical=False)) == []


def test_model_registry_rejects_unknown_family():
    with pytest.raises(ValueError, match="Unsupported model"):
        create_model_adapter("unknown", {})
