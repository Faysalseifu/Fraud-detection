import joblib
import numpy as np
import shap
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def _make_binary_data(n_samples=40, n_features=5, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def test_model_save_load_predictions_match(tmp_path):
    X, y = _make_binary_data()
    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.3,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)

    out_path = tmp_path / "xgb.pkl"
    joblib.dump(model, out_path)
    loaded = joblib.load(out_path)

    proba_orig = model.predict_proba(X)
    proba_loaded = loaded.predict_proba(X)
    assert np.allclose(proba_orig, proba_loaded, atol=1e-6)


def test_shap_values_shape_matches_input():
    X, y = _make_binary_data(n_samples=30, n_features=4, seed=11)
    model = XGBClassifier(
        n_estimators=8,
        max_depth=2,
        learning_rate=0.4,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=11,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    assert shap_values.shape == X.shape


def test_unknown_country_category_pipeline_predicts():
    X_train = np.array(
        [
            [10.0, "US"],
            [12.0, "DE"],
            [9.0, "US"],
            [14.0, "FR"],
        ],
        dtype=object,
    )
    y_train = np.array([0, 1, 0, 1])

    num_features = [0]
    cat_features = [1]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    model = LogisticRegression(max_iter=200)
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)

    X_test = np.array([[11.0, "NG"]], dtype=object)  # Unknown country
    proba = pipe.predict_proba(X_test)
    assert proba.shape == (1, 2)


def test_extreme_values_do_not_break_prediction():
    X, y = _make_binary_data(n_samples=25, n_features=3, seed=3)
    model = XGBClassifier(
        n_estimators=6,
        max_depth=2,
        learning_rate=0.5,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=3,
    )
    model.fit(X, y)

    X_extreme = X * 1e6
    proba = model.predict_proba(X_extreme)
    assert np.isfinite(proba).all()


def test_empty_input_count_prev_within():
    from src.feature_utils import count_prev_within

    assert count_prev_within([], window_hours=2) == []
