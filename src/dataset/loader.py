import numpy as np
from typing import Tuple, List

class MSLRDataLoader:
    def __init__(self, num_features: int = 136):
        """
        DataLoader optimized for the MSLR-WEB30K dataset format.
        MSLR-WEB30K uses 136 features per query-document pair.
        """
        self.num_features = num_features

    def parse_libsvm_line(self, line: str) -> Tuple[int, int, List[float]]:
        """
        Parses a single line of MSLR format: label qid:id 1:val 2:val ...
        """
        parts = line.strip().split()
        if not parts:
            raise ValueError("Empty line encountered.")
            
        label = int(parts[0])
        qid = int(parts[1].split(':')[1])
        
        # Initialize dense feature vector
        features = [0.0] * self.num_features
        for item in parts[2:]:
            feat_idx, feat_val = item.split(':')
            idx = int(feat_idx) - 1  # 1-indexed to 0-indexed mapping
            if 0 <= idx < self.num_features:
                features[idx] = float(feat_val)
                
        return label, qid, features

    def load_dataset(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the entire file into memory and extracts X, y, and query groups.
        
        Returns:
            X (np.ndarray): Feature array of shape (N, num_features)
            y (np.ndarray): Target labels array of shape (N,)
            groups (np.ndarray): 1D array describing the size of each sequential query group
        """
        labels = []
        qids = []
        features_list = []
        
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip() or line.startswith('#'):
                    continue
                label, qid, features = self.parse_libsvm_line(line)
                labels.append(label)
                qids.append(qid)
                features_list.append(features)
                
        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        qids_array = np.array(qids, dtype=np.int32)
        
        # Enforce and extract sequence blocks by qid
        groups = self._compute_query_groups(qids_array)
        
        return X, y, groups

    def _compute_query_groups(self, qids: np.ndarray) -> np.ndarray:
        """
        Computes group size counts for adjacent matching query IDs.
        Assumes data arrives grouped by qid sequentially.
        """
        # Find index switches where qid shifts
        shifts = qids[:-1] != qids[1:]
        shift_indices = np.where(shifts)[0] + 1
        
        # Split tracking array to find run lengths
        group_boundaries = np.insert(shift_indices, 0, 0)
        group_boundaries = np.append(group_boundaries, len(qids))
        groups = np.diff(group_boundaries)
        
        return groups