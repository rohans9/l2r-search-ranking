import os
import pytest
import numpy as np

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

@pytest.fixture
def mock_libsvm_file(tmp_path):
    """
    Creates a temporary mock MSLR-WEB30K format file for testing.
    Format: label qid:id 1:val 2:val ...
    """
    file_path = tmp_path / "mock_mslr.txt"
    
    # 3 queries: qid 10 (2 docs), qid 20 (3 docs), qid 30 (1 doc)
    lines = [
        "2 qid:10 1:0.5 3:1.2\n",
        "0 qid:10 2:0.1 136:0.9\n",
        "4 qid:20 1:0.8\n",
        "1 qid:20 5:0.5 10:0.1\n",
        "3 qid:20 1:0.2 2:0.3 3:0.4\n",
        "0 qid:30 136:1.0\n"
    ]
    
    with open(file_path, "w") as f:
        f.writelines(lines)
        
    return str(file_path)