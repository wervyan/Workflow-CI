import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns

from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, GridSearchCV


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "bank_customer_churn_preprocessing" / "bank_customer_churn_preprocessed.csv"

ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_EXPORT_DIR = BASE_DIR / "model_export"

EXPERIMENT_NAME = "Bank Customer Churn CI"


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "churn" not in df.columns:
        raise ValueError("Kolom target 'churn' tidak ditemukan.")

    X = df.drop(columns=["churn"])
    y = df["churn"]

    return X, y


def save_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Churn", "Churn"],
        yticklabels=["Not Churn", "Churn"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_feature_importance(model, feature_names, output_path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(
        range(len(feature_names)),
        [feature_names[i] for i in indices],
        rotation=45,
        ha="right",
    )
    plt.title("Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    # mlflow.set_experiment(EXPERIMENT_NAME)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_EXPORT_DIR.exists():
        shutil.rmtree(MODEL_EXPORT_DIR)

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    base_model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    with mlflow.start_run(run_name="ci_random_forest_gridsearch") as run:
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)[:, 1]

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        recall = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_test_proba)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_param("cv", 3)
        mlflow.log_param("scoring", "f1")
        mlflow.log_params(grid_search.best_params_)

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        classification_report_text = classification_report(
            y_test,
            y_test_pred,
            target_names=["Not Churn", "Churn"],
            zero_division=0,
        )

        report_path = ARTIFACT_DIR / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report_text)

        metric_info = {
            "run_id": run.info.run_id,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "best_params": grid_search.best_params_,
        }

        metric_info_path = ARTIFACT_DIR / "metric_info.json"
        with open(metric_info_path, "w") as f:
            json.dump(metric_info, f, indent=4)

        confusion_matrix_path = ARTIFACT_DIR / "training_confusion_matrix.png"
        save_confusion_matrix(y_test, y_test_pred, confusion_matrix_path)

        feature_importance_path = ARTIFACT_DIR / "feature_importance.png"
        save_feature_importance(best_model, X.columns.tolist(), feature_importance_path)

        feature_columns_path = ARTIFACT_DIR / "feature_columns.json"
        with open(feature_columns_path, "w") as f:
            json.dump(X.columns.tolist(), f, indent=4)

        input_example = X_test.head(5)
        signature = infer_signature(X_test, best_model.predict(X_test))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )

        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(metric_info_path))
        mlflow.log_artifact(str(confusion_matrix_path))
        mlflow.log_artifact(str(feature_importance_path))
        mlflow.log_artifact(str(feature_columns_path))

        mlflow.sklearn.save_model(
            sk_model=best_model,
            path=str(MODEL_EXPORT_DIR),
            input_example=input_example,
            signature=signature,
        )

        run_id_path = BASE_DIR / "run_id.txt"
        with open(run_id_path, "w") as f:
            f.write(run.info.run_id)

        model_uri_path = BASE_DIR / "model_uri.txt"
        with open(model_uri_path, "w") as f:
            f.write(str(MODEL_EXPORT_DIR))

        print("CI training completed.")
        print(f"Run ID         : {run.info.run_id}")
        print(f"Train Accuracy : {train_accuracy:.4f}")
        print(f"Test Accuracy  : {test_accuracy:.4f}")
        print(f"Precision      : {precision:.4f}")
        print(f"Recall         : {recall:.4f}")
        print(f"F1-score       : {f1:.4f}")
        print(f"ROC-AUC        : {roc_auc:.4f}")


if __name__ == "__main__":
    main()