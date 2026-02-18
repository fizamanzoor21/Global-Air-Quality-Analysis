import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report
)

warnings.filterwarnings("ignore")


DATA_PATH = "outputs/cleaned_data.csv"

FEATURES = [
    "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "Temperature", "Humidity", "Wind Speed",
    "Month", "DayOfWeek"
]

TARGET_REG = "AQI"
TARGET_CLS = "AQI_Category"


def ensure_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if "Month" not in df.columns:
            df["Month"] = df["Date"].dt.month
        if "DayOfWeek" not in df.columns:
            df["DayOfWeek"] = df["Date"].dt.dayofweek
    return df


def pick_existing_features(df: pd.DataFrame, features: list) -> list:
    existing = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"[WARN] Skipped missing features: {missing}")
    return existing


def run_regression(df: pd.DataFrame):
    print("\n=== REGRESSION MODELS (6) ===")

    features = pick_existing_features(df, FEATURES)
    X = df[features]
    y = df[TARGET_REG]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.05),
        "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.5),
        "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    results = []

    for name, model in models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)

        pred_train = pipe.predict(X_train)
        pred_test = pipe.predict(X_test)

        results.append({
            "Model": name,
            "Train_R2": r2_score(y_train, pred_train),
            "Test_R2": r2_score(y_test, pred_test),
            "Test_RMSE": np.sqrt(mean_squared_error(y_test, pred_test)),
            "Test_MAE": mean_absolute_error(y_test, pred_test),
            "CV_R2_Mean": cross_val_score(
                pipe, X_train, y_train, cv=5, scoring="r2"
            ).mean()
        })

    reg_results = pd.DataFrame(results).sort_values("Test_R2", ascending=False)
    reg_results.to_csv("outputs/regression_results.csv", index=False)

    print(reg_results)
    return reg_results


def run_classification(df: pd.DataFrame):
    print("\n=== CLASSIFICATION MODELS (2) ===")

    features = pick_existing_features(df, FEATURES)
    X = df[features]
    y = df[TARGET_CLS]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, random_state=42
        )
    }

    results = []

    for name, model in models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)

        print(f"\n{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, target_names=le.classes_))

        results.append({
            "Model": name,
            "Accuracy": acc
        })

    cls_results = pd.DataFrame(results)
    cls_results.to_csv("outputs/classification_results.csv", index=False)

    return cls_results


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Run preprocessing first to generate cleaned_data.csv")

    df = pd.read_csv(DATA_PATH)
    df = ensure_time_features(df)

    run_regression(df)
    run_classification(df)

    print("\n[OK] Modeling complete. Results saved in outputs/ folder.")


if __name__ == "__main__":
    main()
