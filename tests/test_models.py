import pytest
import numpy as np
import os
from src.models.lightgbm_ranker import LightGBMLambdaMART

class TestLightGBMLambdaMART:

    @pytest.fixture
    def ltr_dummy_data(self):
        """Creates dummy L2R data: 100 docs across 5 queries."""
        np.random.seed(42)
        X = np.random.rand(100, 10) # 100 docs, 10 features
        y = np.random.randint(0, 5, size=100) # Relevance 0-4
        groups = np.array([20, 20, 20, 20, 20]) # 5 queries, 20 docs each
        return X, y, groups

    def test_initialization(self):
        """Test that default objective is forced to lambdarank."""
        model = LightGBMLambdaMART(n_estimators=50, learning_rate=0.1)
        assert model.params['objective'] == 'lambdarank'
        assert model.params['n_estimators'] == 50

    def test_fit_and_predict(self, ltr_dummy_data):
        """Test that the model trains and predicts correctly formatted arrays."""
        X, y, groups = ltr_dummy_data
        
        model = LightGBMLambdaMART(n_estimators=10, num_leaves=7, verbose=-1)
        model.fit(X, y, groups)
        
        preds = model.predict(X)
        
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (100,)
        # Ranking models predict continuous scores, not discrete classes
        assert preds.dtype == np.float64 or preds.dtype == np.float32

    def test_predict_without_fit_raises_error(self, ltr_dummy_data):
        """Ensure predicting with an untrained model safely crashes."""
        X, _, _ = ltr_dummy_data
        model = LightGBMLambdaMART()
        
        with pytest.raises(ValueError, match="Model has not been fitted yet"):
            model.predict(X)

    def test_save_and_load(self, ltr_dummy_data, tmp_path):
        """Test model serialization using joblib."""
        X, y, groups = ltr_dummy_data
        filepath = tmp_path / "lgbm_model.pkl"
        
        # Train and save
        model1 = LightGBMLambdaMART(n_estimators=5, verbose=-1)
        model1.fit(X, y, groups)
        preds1 = model1.predict(X)
        model1.save(str(filepath))
        
        # Load into new instance
        model2 = LightGBMLambdaMART()
        model2.load(str(filepath))
        preds2 = model2.predict(X)
        
        # Predictions from loaded model must perfectly match original
        np.testing.assert_array_almost_equal(preds1, preds2)