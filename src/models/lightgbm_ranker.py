import lightgbm as lgb
import numpy as np
import joblib
from typing import Optional
from src.models.base_ranker import BaseRanker

class LightGBMLambdaMART(BaseRanker):
    """
    LightGBM implementation of LambdaMART.
    Optimizes for NDCG using pairwise/listwise loss.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Pull out the flat hyperparams block from kwargs
        self.hyperparams = kwargs.get("hyperparameters", kwargs).copy()
        
        if 'objective' not in self.hyperparams:
            self.hyperparams['objective'] = 'lambdarank'

        # Keep the normalized, effective configuration available for callers/tests.
        self.params = self.hyperparams.copy()
            
        # Initialize LGBMRanker using the flat hyperparams dictionary only
        self.model = lgb.LGBMRanker(**self.hyperparams)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
            eval_set: Optional[list] = None, eval_group: Optional[list] = None, **fit_params) -> None:
        """
        Fits the LambdaMART model. 
        The 'groups' array is critical here to define query boundaries for the loss function.
        """
        # LightGBM requires early stopping to be handled via callbacks in newer versions
        # but the scikit-learn API still accepts standard evaluation sets.
        self.model.fit(
            X, y,
            group=groups,
            eval_set=eval_set,
            eval_group=eval_group,
            **fit_params
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns raw ranking scores. Higher score = higher predicted relevance."""
        if self.model is None or not hasattr(self.model, 'booster_'):
            raise ValueError("Model has not been fitted yet. Call .fit() first.")
        
        return self.model.predict(X)

    def save(self, filepath: str) -> None:
        """Saves the scikit-learn wrapper using joblib."""
        if self.model is None or not hasattr(self.model, 'booster_'):
            raise ValueError("No fitted model to save.")
        joblib.dump(self.model, filepath)

    def load(self, filepath: str) -> None:
        """Loads the scikit-learn wrapper."""
        self.model = joblib.load(filepath)