import os
import numpy as np
import time

def compile_fold_to_numpy(raw_fold_dir: str, processed_fold_dir: str, split: str, num_features: int = 136):
    """
    Converts raw LETOR formatted text files into highly optimized, binary NumPy matrices.
    
    Data Schema Transformation:
    ---------------------------
    Raw Line Format:  <label> qid:<query_id> 1:<feat_val> 2:<feat_val> ... 136:<feat_val>
    
    Compiled Outputs:
    - X_{split}.npy     --> Float32 matrix of shape (num_lines, 136) storing continuous feature variables.
    - y_{split}.npy     --> Int8 vector of shape (num_lines,) storing relevance labels [0 to 4].
    - qids_{split}.npy  --> Int32 vector of shape (num_lines,) storing unique Query Identifiers.
    
    Enables low-latency disk-streaming via memory mapping (mmap) during training loops.
    """
    txt_path = os.path.join(raw_fold_dir, f"{split}.txt")
    
    # Clean output pathing directed to data/processed/FoldX/
    out_x = os.path.join(processed_fold_dir, f"X_{split}.npy")
    out_y = os.path.join(processed_fold_dir, f"y_{split}.npy")
    out_q = os.path.join(processed_fold_dir, f"qids_{split}.npy")
    
    if not os.path.exists(txt_path):
        print(f"--> [Warning] Optional split '{split}' not found in {raw_fold_dir}. Skipping.")
        return

    # Check if binaries already exist in processed to prevent re-runs
    if os.path.exists(out_x) and os.path.exists(out_y) and os.path.exists(out_q):
        print(f"--> Binaries for {split.upper()} already exist in processed/{os.path.basename(processed_fold_dir)}. Skipping.")
        return

    print(f"=== Compiling {split.upper()} Data ===")
    print(f"--> [1/3] Counting lines in raw {split}.txt to safely pre-allocate RAM...")
    with open(txt_path, 'r') as f:
        num_lines = sum(1 for _ in f)
        
    print(f"--> [2/3] Allocating strict float32 bounds for {num_lines:,} rows...")
    X = np.zeros((num_lines, num_features), dtype=np.float32)
    y = np.zeros(num_lines, dtype=np.int8)
    qids = np.zeros(num_lines, dtype=np.int32)
    
    print(f"--> [3/3] Parsing text directly into allocated memory block...")
    start_time = time.time()
    
    with open(txt_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            
            # Extract Label and Query ID
            y[i] = int(parts[0])
            qids[i] = int(parts[1].split(':')[1])
            
            # Extract 136 Features
            for feat in parts[2:]:
                idx, val = feat.split(':')
                X[i, int(idx) - 1] = float(val)
                
            if (i + 1) % 250000 == 0:
                print(f"    ...parsed {i + 1:,} / {num_lines:,} rows")

    print(f"--> Saving compiled binaries to {processed_fold_dir} (Parse time: {time.time()-start_time:.1f}s)...")
    np.save(out_x, X)
    np.save(out_y, y)
    np.save(out_q, qids)
    print(f"--> {split.upper()} compilation complete!\n")

if __name__ == "__main__":
    raw_base_dir = "data/raw/MSLR-WEB30K"
    processed_base_dir = "data/processed"
    
    print("=== Starting Structured Data Compilation Pipeline (Train, Vali, and Test) ===")
    compiled_count = 0
    
    for fold in range(1, 6):
        raw_fold_dir = os.path.join(raw_base_dir, f"Fold{fold}")
        processed_fold_dir = os.path.join(processed_base_dir, f"Fold{fold}")
        
        if os.path.exists(raw_fold_dir):
            print(f"\nProcessing: {os.path.basename(raw_fold_dir)} -> {processed_fold_dir}")
            os.makedirs(processed_fold_dir, exist_ok=True)
            
            try:
                # Compile all 3 critical splits safely
                compile_fold_to_numpy(raw_fold_dir, processed_fold_dir, "train")
                compile_fold_to_numpy(raw_fold_dir, processed_fold_dir, "vali")
                compile_fold_to_numpy(raw_fold_dir, processed_fold_dir, "test")
                compiled_count += 1
            except Exception as e:
                print(f"[Fold Error] Failed processing Fold {fold}: {e}")
        else:
            print(f"\n[Warning] Raw directory {raw_fold_dir} not found on disk. Skipping.")
            
    print(f"\n=== Pipeline Completed! Verified/Compiled {compiled_count}/5 folds inside data/processed/ ===")