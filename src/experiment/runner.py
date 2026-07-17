import os
import numpy as np
from typing import Dict, Any
from src.models.xgboost_ranker import XGBoostRanker
from src.models.lightgbm_ranker import LightGBMLambdaMART 
from src.evaluation.metrics import IREvaluator         
from src.experiment.logger import MLflowLogger       

class ExperimentRunner:
    """
    Orchestrates the training, evaluation, and tracking pipeline 
    for comparative Learning-to-Rank experiments.
    """
    def __init__(self, experiment_name: str, tracking_uri: str = "./mlruns"):
        self.logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=tracking_uri)

    def _to_group_sizes(self, qids: np.ndarray) -> np.ndarray:
        """Converts raw sorted query IDs into a list of group sizes for LightGBM."""
        _, counts = np.unique(qids, return_counts=True)
        return counts

    def run_experiment(self, run_name: str, model_config: Dict[str, Any], 
                       train_data: tuple, val_data: tuple) -> Dict[str, float]:
        """
        Executes a single model's training and evaluation run, logging everything to MLflow.
        """
        X_train, y_train, qids_train = train_data
        X_val, y_val, qids_val = val_data

        model_type = model_config.get("model_type", "").lower()
        
        # 1. Framework Routing and Parameter Adaptation
        if model_type == "xgboost":
            model = XGBoostRanker(**model_config)
            model.fit(X_train, y_train, qids_train, eval_set=(X_val, y_val, qids_val))
            
        elif model_type == "lightgbm":
            model = LightGBMLambdaMART(**model_config)
            
            # Convert raw qids to group boundaries for LightGBM's C++ engine
            train_groups = self._to_group_sizes(qids_train)
            val_groups = self._to_group_sizes(qids_val)
            
            model.fit(
                X_train, y_train, groups=train_groups,
                eval_set=[(X_val, y_val)],
                eval_group=[val_groups]
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # 2. Score Generation
        val_scores = model.predict(X_val)

        # 3. Compute Metrics
        k_values = model_config.get("evaluation", {}).get("k_values", [5, 10])
        evaluator = IREvaluator(k_values=k_values)     
        metrics = evaluator.evaluate(y_val, val_scores, qids_val)

        # 4. Sanitize Metric Names for MLflow (Converts 'ndcg@5' -> 'ndcg_5')
        sanitized_metrics = {k.replace("@", "_"): v for k, v in metrics.items()}

        # 5. Serialization and Tracking Execution
        artifact_dir = "./tmp_artifacts"
        os.makedirs(artifact_dir, exist_ok=True)
        model_path = os.path.join(artifact_dir, f"{run_name}_model.pkl")
        model.save(model_path)

        self.logger.log_run(
            run_name=run_name,
            config=model_config,
            metrics=sanitized_metrics,
            model_artifact_path=model_path
        )

        if os.path.exists(model_path):
            os.remove(model_path)

        return sanitized_metrics