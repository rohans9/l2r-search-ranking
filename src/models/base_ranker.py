from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional

class BaseRanker(ABC):
    """
    Abstract Base Class for all Learning-to-Rank models in the framework.
    Enforces a standard Scikit-Learn-like API for seamless experimentation.
    """
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
            eval_set: Optional[list] = None, eval_group: Optional[list] = None) -> None:
        """Trains the ranking model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates ranking scores for candidate documents."""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Serializes the model to disk."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Loads a serialized model from disk."""
        pass