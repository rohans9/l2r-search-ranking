import numpy as np
from typing import Dict, List, Tuple

class IREvaluator:
    """
    Computes query-level Information Retrieval metrics for Learning-to-Rank.
    Metrics are calculated per query and then averaged across the dataset.
    """
    def __init__(self, k_values: List[int] = [5, 10], relevance_threshold: int = 1):
        """
        Args:
            k_values: List of cut-off values for @K metrics (e.g., NDCG@5, NDCG@10).
            relevance_threshold: Minimum label value to be considered "relevant" 
                                 (used for Precision, Recall, MAP, and MRR).
        """
        self.k_values = k_values
        self.relevance_threshold = relevance_threshold

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
        """
        Evaluates the predicted ranking scores against the true labels.
        """
        results = {f"ndcg@{k}": [] for k in self.k_values}
        results.update({f"precision@{k}": [] for k in self.k_values})
        results.update({f"recall@{k}": [] for k in self.k_values})
        results["mrr"] = []
        results["map"] = []

        for y_t, y_p in self._split_queries(y_true, y_pred, groups):
            if len(y_t) == 0:
                continue
                
            # Sort documents in this query by predicted score descending
            sort_indices = np.argsort(y_p)[::-1]
            y_true_sorted = y_t[sort_indices]
            
            # Ideal sort for this query (by true relevance descending)
            y_true_ideal = np.sort(y_t)[::-1]

            # Calculate rank-aware metrics
            results["mrr"].append(self._get_mrr(y_true_sorted))
            results["map"].append(self._get_ap(y_true_sorted))

            # Calculate @K metrics
            for k in self.k_values:
                results[f"ndcg@{k}"].append(self._get_ndcg(y_true_sorted, y_true_ideal, k))
                p, r = self._get_precision_recall(y_true_sorted, y_true_ideal, k)
                results[f"precision@{k}"].append(p)
                results[f"recall@{k}"].append(r)

        # Return the mean of each metric across all queries
        return {metric: float(np.mean(values)) for metric, values in results.items()}

    def _split_queries(self, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray):
        """Yields true and predicted labels segmented by query group."""
        indices = np.cumsum(groups)[:-1]
        y_true_splits = np.split(y_true, indices)
        y_pred_splits = np.split(y_pred, indices)
        return zip(y_true_splits, y_pred_splits)

    def _get_dcg(self, y_true_sorted: np.ndarray, k: int) -> float:
        """Calculates Discounted Cumulative Gain."""
        y_top_k = y_true_sorted[:k]
        # FORCE cast the exponent array to float64 to completely eliminate integer bit-overflow
        gains = np.power(2.0, y_top_k.astype(np.float64)) - 1.0
        discounts = np.log2(np.arange(2, len(y_top_k) + 2))
        return np.sum(gains / discounts)

    def _get_ndcg(self, y_true_sorted: np.ndarray, y_true_ideal: np.ndarray, k: int) -> float:
        """Calculates Normalized Discounted Cumulative Gain."""
        dcg = self._get_dcg(y_true_sorted, k)
        idcg = self._get_dcg(y_true_ideal, k)
        return dcg / idcg if idcg > 0 else 0.0

    def _get_mrr(self, y_true_sorted: np.ndarray) -> float:
        """Calculates Mean Reciprocal Rank."""
        relevant_indices = np.where(y_true_sorted >= self.relevance_threshold)[0]
        if len(relevant_indices) == 0:
            return 0.0
        # Rank is 1-indexed, so we add 1 to the 0-indexed position
        return 1.0 / (relevant_indices[0] + 1)

    def _get_ap(self, y_true_sorted: np.ndarray) -> float:
        """Calculates Average Precision for a single query."""
        binary_hits = (y_true_sorted >= self.relevance_threshold).astype(int)
        total_relevant = np.sum(binary_hits)
        if total_relevant == 0:
            return 0.0
        
        # Calculate precision at each cutoff where a relevant document is retrieved
        precisions = np.cumsum(binary_hits) / np.arange(1, len(binary_hits) + 1)
        return np.sum(precisions * binary_hits) / total_relevant

    def _get_precision_recall(self, y_true_sorted: np.ndarray, y_true_ideal: np.ndarray, k: int) -> Tuple[float, float]:
        """Calculates Precision@K and Recall@K."""
        relevant_retrieved = np.sum(y_true_sorted[:k] >= self.relevance_threshold)
        total_relevant = np.sum(y_true_ideal >= self.relevance_threshold)
        
        precision = relevant_retrieved / k
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
        return precision, recall