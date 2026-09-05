import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import skops.io as sio


# ============================================================
# 1. Load Dataset
# ============================================================

housing = fetch_california_housing()

X, y = housing["data"], housing["target"]

Xtr, Xte, ytr, yte = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


# ============================================================
# 2. Build ML Model
# ============================================================

n_estimators = 100
max_depth = 12
min_samples_leaf = 5


# 1. We don't not to impute the data bcz there are no missing values in california dataset
# 2. We don't need to scale it becauze we are doing Random Forest Regressor
# so moral of the story make only model not pipeline

model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=1,
        )

# ============================================================
# 3. Configure MLflow
# ============================================================

mlflow.set_experiment("California-Housing-RandomForest")


# ============================================================
# 4. Train + Track Experiment
# ============================================================

with mlflow.start_run() as run:

    print("MLflow Run ID:", run.info.run_id)

    # --------------------------------------------------------
    # Log dataset information
    # --------------------------------------------------------

    mlflow.log_param("dataset", "California Housing")
    mlflow.log_param("n_samples", X.shape[0])
    mlflow.log_param("n_features", X.shape[1])

    # --------------------------------------------------------
    # Log train/test configuration
    # --------------------------------------------------------

    mlflow.log_param("test_size", 0.25)
    mlflow.log_param("random_state", 42)

    # --------------------------------------------------------
    # Log Random Forest hyperparameters
    # --------------------------------------------------------

    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("n_jobs", 1)

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print("Training model...")

    model.fit(Xtr, ytr)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(Xte)

    # --------------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------------

    r2 = r2_score(yte, predictions)
    rmse = np.sqrt(mean_squared_error(yte, predictions))
    mae = mean_absolute_error(yte, predictions)

    print(f"Test R2:   {r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")

    # --------------------------------------------------------
    # Log metrics to MLflow
    # --------------------------------------------------------

    mlflow.log_metric("test_r2", r2)
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)

    # --------------------------------------------------------
    # Log model to MLflow
    # --------------------------------------------------------

    mlflow.sklearn.log_model(
        model,
        name="random_forest_model"
    )

    # --------------------------------------------------------
    # Save model for FastAPI / skops deployment
    # --------------------------------------------------------

    os.makedirs("model", exist_ok=True)

    sio.dump(
        model,
        "model/model.skops"
    )

    # --------------------------------------------------------
    # Log the skops model as an artifact
    # --------------------------------------------------------

    mlflow.log_artifact(
        "model/model.skops",
        artifact_path="deployment_model"
    )

    print("\nModel saved successfully:")
    print("model/model.skops")

    print("\nMLflow tracking completed.")