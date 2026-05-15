"""
Module 6 Assignment: Otomoto Marketing Segmentation Model Optimization
Optimized ANN churn/marketing segmentation model using Telco customer data.

How to run:
    python otomoto_ann_model_optimization.py

Outputs:
    - otomoto_ann_optimizer_results.csv
    - otomoto_customer_segments.csv
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "teleconnect(3).csv"
if not DATA_PATH.exists():
    DATA_PATH = Path("/mnt/data/teleconnect(3).csv")
OUTPUT_RESULTS = SCRIPT_DIR / "otomoto_ann_optimizer_results.csv"
OUTPUT_SEGMENTS = SCRIPT_DIR / "otomoto_customer_segments.csv"


def load_and_prepare_data(path: Path):
    """Load the customer file, clean TotalCharges, and split predictors/target."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})
    return df, X, y


def build_preprocessor(X):
    """Create numeric scaling and categorical one-hot encoding pipeline."""
    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = X.select_dtypes(exclude=["object"]).columns.tolist()
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("numeric", StandardScaler(), numeric_columns),
        ("categorical", encoder, categorical_columns),
    ])


def best_threshold(y_true, probabilities):
    """Select the classification threshold that maximizes validation F1-score."""
    thresholds = np.arange(0.20, 0.71, 0.01)
    scores = []
    for t in thresholds:
        pred = (probabilities >= t).astype(int)
        scores.append(f1_score(y_true, pred, zero_division=0))
    idx = int(np.argmax(scores))
    return float(thresholds[idx]), float(scores[idx])


def evaluate(y_true, probabilities, threshold):
    """Evaluate a model from predicted probabilities and a decision threshold."""
    pred = (probabilities >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1-score": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "Log loss": log_loss(y_true, np.column_stack([1 - probabilities, probabilities])),
        "Confusion Matrix": confusion_matrix(y_true, pred).tolist(),
    }


def main():
    df, X, y = load_and_prepare_data(DATA_PATH)
    preprocessor = build_preprocessor(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, stratify=y_train_full, random_state=42
    )

    models = {
        "Original baseline ANN: Adam, threshold 0.50": {
            "classifier": MLPClassifier(
                hidden_layer_sizes=(16,), solver="adam", alpha=0.0001,
                learning_rate_init=0.001, batch_size=128, max_iter=120,
                early_stopping=False, random_state=42
            ),
            "optimize_threshold": False,
            "description": "Recreated existing simple ANN with Adam and default 0.50 classification threshold."
        },
        "Optimizer 1: SGD + momentum": {
            "classifier": MLPClassifier(
                hidden_layer_sizes=(16,), solver="sgd", alpha=0.0001,
                learning_rate="adaptive", learning_rate_init=0.01, momentum=0.9,
                nesterovs_momentum=True, batch_size=128, max_iter=160,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=15, random_state=42
            ),
            "optimize_threshold": True,
            "description": "Gradient descent with momentum and adaptive learning rate."
        },
        "Optimizer 2: Adam + regularization + early stopping": {
            "classifier": MLPClassifier(
                hidden_layer_sizes=(32, 16), solver="adam", alpha=0.001,
                learning_rate_init=0.001, batch_size=128, max_iter=160,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=15, random_state=42
            ),
            "optimize_threshold": True,
            "description": "Adaptive moment estimation with a deeper ANN, L2 regularization, and early stopping."
        },
        "Optimizer 3: L-BFGS quasi-Newton": {
            "classifier": MLPClassifier(
                hidden_layer_sizes=(16,), solver="lbfgs", alpha=0.001,
                max_iter=100, random_state=42
            ),
            "optimize_threshold": True,
            "description": "A quasi-Newton optimizer suitable for smaller dense tabular models."
        },
    }

    result_rows = []
    trained_pipelines = {}
    thresholds = {}

    for name, spec in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("ann", spec["classifier"])])
        pipe.fit(X_train, y_train)
        val_proba = pipe.predict_proba(X_val)[:, 1]
        if spec["optimize_threshold"]:
            threshold, val_f1 = best_threshold(y_val, val_proba)
        else:
            threshold, val_f1 = 0.50, f1_score(y_val, (val_proba >= 0.50).astype(int))
        test_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, test_proba, threshold)
        metrics.update({
            "Model": name,
            "Decision threshold": threshold,
            "Validation F1 at selected threshold": val_f1,
            "Iterations": getattr(pipe.named_steps["ann"], "n_iter_", None),
            "Training loss": getattr(pipe.named_steps["ann"], "loss_", None),
            "Documentation": spec["description"],
        })
        result_rows.append(metrics)
        trained_pipelines[name] = pipe
        thresholds[name] = threshold

    results = pd.DataFrame(result_rows)
    cols = ["Model", "Decision threshold", "Accuracy", "Precision", "Recall", "F1-score",
            "ROC-AUC", "Log loss", "Iterations", "Training loss", "Validation F1 at selected threshold",
            "Confusion Matrix", "Documentation"]
    results = results[cols]
    results.to_csv(OUTPUT_RESULTS, index=False)
    print(results.to_string(index=False))

    # Select the best operational model by F1-score first, then ROC-AUC.
    best_model_name = results.sort_values(["F1-score", "ROC-AUC"], ascending=False).iloc[0]["Model"]
    best_threshold_value = float(results.loc[results["Model"] == best_model_name, "Decision threshold"].iloc[0])

    final_model = Pipeline([("preprocess", preprocessor), ("ann", models[best_model_name]["classifier"])])
    final_model.fit(X_train_full, y_train_full)
    all_proba = final_model.predict_proba(X)[:, 1]

    segments = df[[
        "customerID", "gender", "SeniorCitizen", "tenure", "Contract", "InternetService",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
    ]].copy()
    segments["Churn_Risk_Probability"] = all_proba
    segments["Marketing_Segment"] = pd.cut(
        all_proba,
        bins=[-0.001, 0.30, 0.60, 1.0],
        labels=["Low risk: retain and upsell", "Medium risk: nurture", "High risk: retention/win-back"]
    )
    segments["Predicted_Churn_Flag"] = np.where(all_proba >= best_threshold_value, "Yes", "No")
    segments.to_csv(OUTPUT_SEGMENTS, index=False)

    print("\nBest operational model:", best_model_name)
    print("Selected decision threshold:", round(best_threshold_value, 2))
    print("\nMarketing segment counts:")
    print(segments["Marketing_Segment"].value_counts().to_string())


if __name__ == "__main__":
    main()
