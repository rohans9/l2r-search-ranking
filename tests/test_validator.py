import pytest
import numpy as np
from src.dataset.validator import DatasetValidator

class TestDatasetValidator:
    
    def test_validate_ltr_split_success(self):
        """Test validator passes on perfectly formed matrices."""
        X = np.zeros((5, 136))
        y = np.array([0, 1, 2, 3, 4])
        groups = np.array([2, 3])
        
        assert DatasetValidator.validate_ltr_split(X, y, groups) is True

    def test_validate_shape_mismatch(self):
        """Test validator catches mismatch between features and labels."""
        X = np.zeros((5, 136))
        y = np.array([0, 1, 2, 3])  # Only 4 labels for 5 rows
        groups = np.array([2, 3])
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            DatasetValidator.validate_ltr_split(X, y, groups)

    def test_validate_group_sum_mismatch(self):
        """Test validator catches when group sizes don't add up to total rows."""
        X = np.zeros((5, 136))
        y = np.array([0, 1, 2, 3, 4])
        groups = np.array([2, 2])  # Sum is 4, but we have 5 rows
        
        with pytest.raises(ValueError, match="Group validation failed"):
            DatasetValidator.validate_ltr_split(X, y, groups)