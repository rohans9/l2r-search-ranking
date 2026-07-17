import pytest
import numpy as np
from src.dataset.loader import MSLRDataLoader

class TestMSLRDataLoader:
    
    def setup_method(self):
        self.loader = MSLRDataLoader(num_features=136)

    def test_parse_libsvm_line_valid(self):
        """Test parsing of a standard MSLR-WEB30K line."""
        line = "2 qid:10 1:0.5 3:1.2"
        label, qid, features = self.loader.parse_libsvm_line(line)
        
        assert label == 2
        assert qid == 10
        assert len(features) == 136
        # Note: LibSVM is 1-indexed, Python arrays are 0-indexed
        assert features[0] == 0.5  # Feature 1
        assert features[1] == 0.0  # Feature 2 (missing, should be 0)
        assert features[2] == 1.2  # Feature 3

    def test_parse_libsvm_line_empty(self):
        """Test that empty lines raise appropriate errors."""
        with pytest.raises(ValueError, match="Empty line encountered."):
            self.loader.parse_libsvm_line("   \n")

    def test_compute_query_groups(self):
        """Test that query group sizes are calculated correctly."""
        # 2 docs for qid 1, 3 docs for qid 2, 1 doc for qid 3
        qids = np.array([1, 1, 2, 2, 2, 3])
        groups = self.loader._compute_query_groups(qids)
        
        np.testing.assert_array_equal(groups, np.array([2, 3, 1]))

    def test_load_dataset_integration(self, mock_libsvm_file):
        """Test end-to-end loading of a dataset file."""
        X, y, groups = self.loader.load_dataset(mock_libsvm_file)
        
        # 1. Check Dimensions
        assert X.shape == (6, 136)
        assert len(y) == 6
        
        # 2. Check Labels
        np.testing.assert_array_equal(y, np.array([2, 0, 4, 1, 3, 0]))
        
        # 3. Check Groups (2 docs in qid 10, 3 docs in 20, 1 doc in 30)
        np.testing.assert_array_equal(groups, np.array([2, 3, 1]))