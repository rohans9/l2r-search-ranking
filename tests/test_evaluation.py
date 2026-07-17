import pytest
import numpy as np
from src.evaluation.metrics import IREvaluator

class TestIREvaluator:
    
    def setup_method(self):
        # We will test @2 and @5, treating any label >= 1 as relevant
        self.evaluator = IREvaluator(k_values=[2, 5], relevance_threshold=1)

    def test_perfect_ranking(self):
        """Test metrics when the model perfectly ranks the documents."""
        y_true = np.array([3, 2, 1, 0])
        y_pred = np.array([0.9, 0.8, 0.7, 0.1])
        groups = np.array([4]) # Single query with 4 docs
        
        metrics = self.evaluator.evaluate(y_true, y_pred, groups)
        
        # Perfect ordering means NDCG = 1.0
        assert np.isclose(metrics["ndcg@2"], 1.0)
        assert np.isclose(metrics["ndcg@5"], 1.0)
        
        # First item is relevant (label 3), so MRR = 1/1 = 1.0
        assert np.isclose(metrics["mrr"], 1.0)
        
        # MAP should be 1.0 for perfect binary ranking
        assert np.isclose(metrics["map"], 1.0)

    def test_worst_ranking(self):
        """Test metrics when the model ranks the best documents last."""
        y_true = np.array([3, 2, 1, 0])
        # Reversed predictions
        y_pred = np.array([0.1, 0.7, 0.8, 0.9]) 
        groups = np.array([4])
        
        metrics = self.evaluator.evaluate(y_true, y_pred, groups)
        
        # The model put the label 0 at rank 1, and label 1 at rank 2
        # y_true_sorted = [0, 1, 2, 3]
        # MRR: first relevant (label 1) is at index 1 (rank 2). MRR = 1/2 = 0.5
        assert np.isclose(metrics["mrr"], 0.5)
        
        # Precision_2: Top 2 are [0, 1]. Only one is >= threshold (1). P_2 = 1/2 = 0.5
        assert np.isclose(metrics["precision@2"], 0.5)

    def test_multi_query_aggregation(self):
        """Test that the metrics average correctly across multiple queries."""
        y_true = np.array([
            2, 1, 0,  # Query 1
            0, 0, 1   # Query 2
        ])
        y_pred = np.array([
            0.9, 0.5, 0.1,  # Query 1 (Perfect) -> MRR = 1.0
            0.9, 0.8, 0.7   # Query 2 (Worst)   -> MRR = 1/3 (0.333...)
        ])
        groups = np.array([3, 3])
        
        metrics = self.evaluator.evaluate(y_true, y_pred, groups)
        
        # Average MRR = (1.0 + 0.333333) / 2 = 0.666666
        assert np.isclose(metrics["mrr"], (1.0 + 1/3) / 2)

    def test_no_relevant_documents(self):
        """Test handling of queries with zero relevant documents."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0.5, 0.6, 0.1])
        groups = np.array([3])
        
        metrics = self.evaluator.evaluate(y_true, y_pred, groups)
        
        # If no documents are relevant, MRR, MAP, NDCG, and Recall should handle DivByZero gracefully
        assert metrics["mrr"] == 0.0
        assert metrics["map"] == 0.0
        assert metrics["ndcg@2"] == 0.0
        assert metrics["recall@2"] == 0.0