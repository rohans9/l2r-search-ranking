import pytest
import numpy as np
import os
from src.models.xgboost_ranker import XGBoostRanker

@pytest.fixture
def dummy_rank_data():
    """Generates synthetic data representing 3 queries with 4 documents each."""
    np.random.seed(42)
    X = np.random.rand(12, 5)
    y = np.random.choice([0, 1, 2, 3], size=12)
    qids = np.array([101]*4 + [102]*4 + [103]*4)
    return X, y, qids

@pytest.fixture
def ranker_config():
    """Returns a baseline configuration dictionary for XGBoost."""
    return {
        "model_type": "xgboost",
        "hyperparameters": {
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 10,
            "random_state": 42
        }
    }

class TestXGBoostRanker:

    def test_model_initialization(self, ranker_config):
        """Verifies hyperparameters map correctly to the internal XGBRanker object."""
        ranker = XGBoostRanker(**ranker_config)
        assert ranker.model.learning_rate == 0.1
        assert ranker.model.max_depth == 3
        assert ranker.model.n_estimators == 10

    def test_fit_and_predict_shapes(self, ranker_config, dummy_rank_data):
        """Verifies fitting works seamlessly and predictions match dataset row counts."""
        X, y, qids = dummy_rank_data
        ranker = XGBoostRanker(**ranker_config)
        
        ranker.fit(X, y, qids)
        scores = ranker.predict(X)
        
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (12,)

    def test_fit_with_validation_set(self, ranker_config, dummy_rank_data):
        """Ensures training successfully handles optional verification sets during training logs."""
        X, y, qids = dummy_rank_data
        ranker = XGBoostRanker(**ranker_config)
        
        # Re-use data as validation set for structural verification
        ranker.fit(X, y, qids, eval_set=(X, y, qids))
        scores = ranker.predict(X)
        assert scores.shape == (12,)

    def test_serialization_integrity(self, ranker_config, dummy_rank_data, tmp_path):
        """Verifies trained models can be written and read from disk without losing performance values."""
        X, y, qids = dummy_rank_data
        ranker = XGBoostRanker(**ranker_config)
        ranker.fit(X, y, qids)
        
        original_scores = ranker.predict(X)
        model_file = tmp_path / "xgb_model.pkl"
        
        # Save model
        ranker.save(str(model_file))
        assert os.path.exists(model_file)
        
        # Load model and re-verify prediction output parity
        loaded_ranker = XGBoostRanker.load(str(model_file))
        loaded_scores = loaded_ranker.predict(X)
        
        np.testing.assert_array_almost_equal(original_scores, loaded_scores)