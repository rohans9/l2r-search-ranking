import numpy as np

class QueryGroupScaler:
    """
    Standardizes features locally within each query group.
    Prevents global feature scale mismatches from degrading ranking pairs.
    """
    def __init__(self, strategy: str = "zscore"):
        self.strategy = strategy

    def fit_transform(self, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """
        Standardizes features per query group.
        groups: Array of group sizes (e.g., [4, 3, 5]) indicating how many rows belong to each query.
        """
        X_scaled = X.copy()
        
        # Compute the boundary indices for slicing groups
        indices = np.cumsum(groups)[:-1]
        splits = np.split(X_scaled, indices, axis=0)
        
        processed_splits = []
        for split in splits:
            if len(split) <= 1:
                processed_splits.append(split)
                continue
                
            if self.strategy == "zscore":
                mean = np.mean(split, axis=0)
                std = np.std(split, axis=0)
                # Avoid division by zero for static features
                std[std == 0] = 1.0
                normalized = (split - mean) / std
                
            elif self.strategy == "minmax":
                min_val = np.min(split, axis=0)
                max_val = np.max(split, axis=0)
                range_val = max_val - min_val
                range_val[range_val == 0] = 1.0
                normalized = (split - min_val) / range_val
                
            processed_splits.append(normalized)
            
        return np.vstack(processed_splits)