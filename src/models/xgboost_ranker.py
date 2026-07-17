import os
import pickle
import numpy as np
import xgboost as xgb
from typing import Dict, Any, Optional
from src.models.base_ranker import BaseRanker

class XGBoostRanker(BaseRanker):
    """
    XGBoost LambdaMART implementation for Learning-to-Rank tasks.
    Wraps xgb.XGBRanker to conform to the project's BaseRanker interface.
    """
    def __init__(self, **kwargs):
        """
        Initializes the XGBoost Ranker with hyperparameter configurations.
        """
        super().__init__(**kwargs)
        
        # Safely extract hyperparameters whether nested under 'hyperparameters' or passed flat
        self.hyperparams = kwargs.get("hyperparameters", kwargs)
        
        # Extract framework-specific parameters with sensible defaults
        self.model = xgb.XGBRanker(
            objective="rank:ndcg",
            learning_rate=self.hyperparams.get("learning_rate", 0.05),
            max_depth=self.hyperparams.get("max_depth", 5),
            n_estimators=self.hyperparams.get("n_estimators", 100),
            subsample=self.hyperparams.get("subsample", 1.0),
            colsample_bytree=self.hyperparams.get("colsample_bytree", 1.0),
            random_state=self.hyperparams.get("random_state", 42),
            tree_method=self.hyperparams.get("tree_method", "hist")
        )

    def fit(self, X: np.ndarray, y: np.ndarray, qids: np.ndarray, 
            eval_set: Optional[tuple] = None) -> None:
        """
        Trains the XGBoost ranking model.
        
        XGBoost natively accepts explicit 'qid' arrays in its modern API.
        The arrays must be sorted by query ID for ranking algorithms to process correctly.
        """
        # Sort data by query ID to ensure contiguous query groups
        sort_idx = np.argsort(qids)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        qids_sorted = qids[sort_idx]

        fit_params = {
            "X": X_sorted,
            "y": y_sorted,
            "qid": qids_sorted
        }

        # Format and append evaluation set if provided
        if eval_set:
            X_val, y_val, qids_val = eval_set
            val_sort_idx = np.argsort(qids_val)
            fit_params["eval_set"] = [(X_val[val_sort_idx], y_val[val_sort_idx])]
            fit_params["eval_qid"] = [qids_val[val_sort_idx]]
            fit_params["verbose"] = False

        self.model.fit(**fit_params)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates continuous relevance scores for a matrix of document features.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
        return self.model.predict(X)

    def save(self, filepath: str) -> None:
        """
        Serializes and saves the trained XGBoost model wrapper to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "XGBoostRanker":
        """
        Loads a serialized XGBoost model wrapper from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at file path: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.load(f)