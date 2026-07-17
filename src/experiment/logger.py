import os
import mlflow
from typing import Dict, Any, Optional

class MLflowLogger:
    """
    Wrapper around MLflow to standardize experiment tracking for L2R models.
    """
    def __init__(self, experiment_name: str, tracking_uri: str = "./mlruns"):
        """
        Initializes the MLflow tracking environment.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        # Ensure tracking directory exists if using local file system
        if tracking_uri.startswith("./"):
            os.makedirs(tracking_uri, exist_ok=True)
            
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def log_run(self, run_name: str, config: Dict[str, Any], metrics: Dict[str, float], model_artifact_path: Optional[str] = None):
        """
        Executes a logged run, recording parameters, metrics, and saving the model file.
        """
        with mlflow.start_run(run_name=run_name):
            # 1. Log Configuration Parameters
            self._log_params_flat(config)
            
            # 2. Log IR Evaluation Metrics
            mlflow.log_metrics(metrics)
            
            # 3. Log the Serialized Model (if provided)
            if model_artifact_path and os.path.exists(model_artifact_path):
                mlflow.log_artifact(model_artifact_path, artifact_path="models")

    def _log_params_flat(self, params: Dict[str, Any], parent_key: str = ''):
        """
        Recursively flattens nested YAML configuration dictionaries (e.g., config['model']['hyperparameters']) 
        so they can be tracked cleanly as 'model.hyperparameters.learning_rate'.
        """
        for k, v in params.items():
            key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                self._log_params_flat(v, key)
            else:
                # MLflow expects parameters to be logged as strings, floats, or ints.
                # Converting lists (like k_values: [5, 10]) to strings prevents crashing.
                mlflow.log_param(key, str(v))