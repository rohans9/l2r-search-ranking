import pytest
import numpy as np
import os
import mlflow
from src.experiment.runner import ExperimentRunner

@pytest.fixture
def tracking_uri(tmp_path):
    return f"file://{tmp_path}/mlruns"

@pytest.fixture
def mock_datasets():
    np.random.seed(42)
    # 3 queries, 4 documents per query = 12 samples total
    X_train = np.random.rand(12, 4)
    y_train = np.random.choice([0, 1, 2, 3], size=12)
    qids_train = np.array([1]*4 + [2]*4 + [3]*4)

    X_val = np.random.rand(12, 4)
    y_val = np.random.choice([0, 1, 2, 3], size=12)
    qids_val = np.array([1]*4 + [2]*4 + [3]*4)

    return (X_train, y_train, qids_train), (X_val, y_val, qids_val)

class TestExperimentRunner:

    def test_end_to_end_xgboost_run(self, tracking_uri, mock_datasets):
        """Verifies that the runner executes a complete XGBoost training and tracking iteration."""
        train_data, val_data = mock_datasets
        runner = ExperimentRunner(experiment_name="e2e_test_experiment", tracking_uri=tracking_uri)

        xgb_config = {
            "model_type": "xgboost",
            "hyperparameters": {
                "learning_rate": 0.1,
                "max_depth": 2,
                "n_estimators": 5,
                "random_state": 42
            },
            "evaluation": {
                "k_values": [5]
            }
        }

        metrics = runner.run_experiment(
            run_name="xgb_baseline",
            model_config=xgb_config,
            train_data=train_data,
            val_data=val_data
        )

        assert "ndcg_5" in metrics
        assert "map" in metrics
        
        # Confirm run entry recorded into MLflow storage
        experiment = mlflow.get_experiment_by_name("e2e_test_experiment")
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) == 1
        assert runs.iloc[0]["params.model_type"] == "xgboost"

    def test_end_to_end_lightgbm_run(self, tracking_uri, mock_datasets):
        """Verifies that the runner executes a complete LightGBM training and tracking iteration."""
        train_data, val_data = mock_datasets
        runner = ExperimentRunner(experiment_name="e2e_test_experiment", tracking_uri=tracking_uri)

        lgb_config = {
            "model_type": "lightgbm",
            "hyperparameters": {
                "learning_rate": 0.1,
                "max_depth": 2,
                "n_estimators": 5,
                "random_state": 42
            },
            "evaluation": {
                "k_values": [5]
            }
        }

        metrics = runner.run_experiment(
            run_name="lgb_baseline",
            model_config=lgb_config,
            train_data=train_data,
            val_data=val_data
        )

        assert "ndcg_5" in metrics
        
        # Confirm both run entries now reside in MLflow storage
        experiment = mlflow.get_experiment_by_name("e2e_test_experiment")
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) == 1