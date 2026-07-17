import pytest
import os
import mlflow
from src.experiment.logger import MLflowLogger

# Move the fixture out here at the module level
@pytest.fixture
def tracking_uri(tmp_path):
    """Use a temporary directory for MLflow tracking so tests don't pollute your real mlruns folder."""
    return f"file://{tmp_path}/mlruns"


class TestMLflowLogger:

    def test_logger_initialization(self, tracking_uri):
        """Test that the logger correctly sets up the experiment."""
        logger = MLflowLogger(experiment_name="test_init_exp", tracking_uri=tracking_uri)
        
        experiment = mlflow.get_experiment_by_name("test_init_exp")
        assert experiment is not None
        assert experiment.name == "test_init_exp"

    def test_log_run_execution(self, tracking_uri, tmp_path):
        """Test end-to-end logging of params, metrics, and artifacts."""
        logger = MLflowLogger(experiment_name="test_run_exp", tracking_uri=tracking_uri)
        
        # Mock Data
        config = {
            "model_type": "xgboost",
            "hyperparams": {
                "max_depth": 3,
                "learning_rate": 0.05,
                "k_values": [5, 10]
            }
        }
        metrics = {
            "ndcg_10": 0.88,
            "map": 0.76
        }
        
        # Create a dummy model artifact file
        dummy_model_path = tmp_path / "dummy_model.pkl"
        dummy_model_path.write_text("fake serialized model data")
        
        # Execute logging
        logger.log_run(
            run_name="baseline_test", 
            config=config, 
            metrics=metrics, 
            model_artifact_path=str(dummy_model_path)
        )
        
        # Verify via MLflow's Pandas API
        experiment = mlflow.get_experiment_by_name("test_run_exp")
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        # Assert exactly one run was logged
        assert len(runs) == 1
        
        # Assert metrics logged correctly
        assert runs.iloc[0]["metrics.ndcg_10"] == 0.88
        assert runs.iloc[0]["metrics.map"] == 0.76
        
        # Assert nested parameters were flattened and logged correctly
        assert runs.iloc[0]["params.model_type"] == "xgboost"
        assert runs.iloc[0]["params.hyperparams.max_depth"] == "3"
        assert runs.iloc[0]["params.hyperparams.k_values"] == "[5, 10]"