import numpy as np

class DatasetValidator:
    @staticmethod
    def validate_ltr_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> bool:
        """
        Runs programmatic sanity checks on extracted matrices to catch pipeline corruption early.
        """
        # 1. Coordinate check
        if X.shape[0] != len(y):
            raise ValueError(f"Shape mismatch: X rows ({X.shape[0]}) do not match y dimensions ({len(y)}).")
            
        # 2. Group Alignment check
        total_grouped_records = np.sum(groups)
        if total_grouped_records != X.shape[0]:
            raise ValueError(
                f"Group validation failed: Sum of items across groups ({total_grouped_records}) "
                f"differs from absolute array row-count ({X.shape[0]})."
            )
            
        # 3. Label Value boundary check (MSLR ranges from 0 to 4)
        unique_labels = np.unique(y)
        if not all(0 <= label <= 4 for label in unique_labels):
            print(f"[Warning]: Out-of-bounds metrics or unexpected target classes detected: {unique_labels}")
            
        return True