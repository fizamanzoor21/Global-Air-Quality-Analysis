import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import os
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
)


DATA_PATH = "data/processed/cleaned_air_quality.csv"


BASE_FEATURES = [
    "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "Temperature", "Humidity", "Wind Speed"
]

TIME_FEATURES = ["Month", "DayOfWeek"]


def ensure_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        if "Month" not in df.columns:
            df["Month"] = df["Date"].dt.month

        if "DayOfWeek" not in df.columns:
            df["DayOfWeek"] = df["Date"].dt.dayofweek

    return df


def pick_existing_features(df: pd.DataFrame, features: list) -> list:
    existing = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    if missing:
        print(f"[WARN] Skipping missing features: {missing}")

    return existing

def run_regression(df: pd.DataFrame):
    print("\n=== REGRESSION MODELS (6) ===")

    features = pick_existing_features(df, BASE_FEATURES + TIME_FEATURES)
    X = df[features]
    y = df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.05),
        "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    results = []
    fitted_pipes = {}

    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("pre", preprocessor),
                ("model", model),
            ]
        )

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        results.append({
            "Model": name,
            "Test_R2": r2_score(y_test, preds),
            "Test_RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "Test_MAE": mean_absolute_error(y_test, preds),
        })

        fitted_pipes[name] = pipe

    results_df = pd.DataFrame(results).sort_values("Test_R2", ascending=False).reset_index(drop=True)

    # Save regression results table (optional but useful)
    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/regression_results.csv", index=False)

    print(results_df)

   
    best_model_name = results_df.loc[0, "Model"]
    best_pipe = fitted_pipes[best_model_name]

    imp_df = permutation_importance_rmse(
        best_pipe, X_test, y_test, n_repeats=5, random_state=42
    )

    imp_df.to_csv("outputs/feature_importance_permutation.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=imp_df.head(10),
        x="Importance_Delta_RMSE",
        y="Feature"
    )
    plt.title(f"Permutation Feature Importance (RMSE Increase) - {best_model_name}")
    plt.xlabel("Increase in RMSE after permuting feature")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("outputs/fig_feature_importance.png", dpi=200)
    plt.close()

    print(f"[OK] Feature importance saved: outputs/fig_feature_importance.png")
    print(f"[OK] Feature importance table saved: outputs/feature_importance_permutation.csv")

    return results_df



def run_classification(df: pd.DataFrame):
    print("\n=== CLASSIFICATION MODELS (2) ===")

    features = pick_existing_features(df, BASE_FEATURES + TIME_FEATURES)

    X = df[features]
    y = df["AQI_Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
    }

    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("pre", preprocessor),
                ("model", model),
            ]
        )

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        print(f"\n{name}")
        print("Accuracy:", accuracy_score(y_test, preds))
        print(classification_report(y_test, preds))

def permutation_importance_rmse(pipe, X_test, y_test, n_repeats=5, random_state=42):
    """
    Permutation importance for a fitted Pipeline (preprocessor + model).
    Importance = increase in RMSE when a feature column is randomly permuted.
    """
    rng = np.random.default_rng(random_state)

    # Baseline
    baseline_pred = pipe.predict(X_test)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

    importances = []
    for col in X_test.columns:
        deltas = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            pred = pipe.predict(Xp)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            deltas.append(rmse - baseline_rmse)

        importances.append({
            "Feature": col,
            "Importance_Delta_RMSE": float(np.mean(deltas))
        })

    imp_df = pd.DataFrame(importances).sort_values(
        "Importance_Delta_RMSE", ascending=False
    ).reset_index(drop=True)

    return imp_df


def main():
    df = pd.read_csv(DATA_PATH)
    df = ensure_time_features(df)

    run_regression(df)
    run_classification(df)


if __name__ == "__main__":
    main()
