import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import numpy as np
from src.experiment.runner import ExperimentRunner
from src.features.processors import QueryGroupScaler

def load_mslr_fold(fold_dir: str):
    """Loads 100% of pre-compiled NumPy arrays using high-performance disk memory mapping."""
    print("--> Instantly memory-mapping 100% of Train binaries...")
    X_train = np.load(os.path.join(fold_dir, "X_train.npy"), mmap_mode='r')
    y_train = np.load(os.path.join(fold_dir, "y_train.npy"))
    qids_train = np.load(os.path.join(fold_dir, "qids_train.npy"))
    
    print("--> Instantly memory-mapping 100% of Validation binaries...")
    X_val = np.load(os.path.join(fold_dir, "X_vali.npy"), mmap_mode='r')
    y_val = np.load(os.path.join(fold_dir, "y_vali.npy"))
    qids_val = np.load(os.path.join(fold_dir, "qids_vali.npy"))
    
    return (X_train, y_train.astype(int), qids_train.astype(int)), (X_val, y_val.astype(int), qids_val.astype(int))

if __name__ == "__main__":
    runner = ExperimentRunner(experiment_name="MSLR_Web30K_Benchmark")
    fold_1_path = "data/processed/Fold1"
    
    try:
        print("=== Step 1: Loading 100% Real MSLR-WEB30K Data ===")
        (X_train, y_train, qids_train), (X_val, y_val, qids_val) = load_mslr_fold(fold_1_path)
        
        print("--> Applying Query-Grouped Z-Score Feature Scaling across full matrices...")
        _, train_groups = np.unique(qids_train, return_counts=True)
        _, val_groups = np.unique(qids_val, return_counts=True)
        
        scaler = QueryGroupScaler(strategy="zscore")
        X_train_scaled = scaler.fit_transform(X_train, train_groups)
        X_val_scaled = scaler.fit_transform(X_val, val_groups)
        
        train_data = (X_train_scaled, y_train, qids_train)
        val_data = (X_val_scaled, y_val, qids_val)
        
        # =====================================================================
        # CONFIGURATION ENGINE MATRIX
        # =====================================================================
        configs = {
            "xgb_pairwise_baseline": {
                "model_type": "xgboost",
                "hyperparameters": {"learning_rate": 0.1, "max_depth": 6, "n_estimators": 50, "tree_method": "hist"},
                "evaluation": {"k_values": [5, 10]}
            },
            "lgb_pairwise_baseline": {
                "model_type": "lightgbm",
                "hyperparameters": {"learning_rate": 0.1, "max_depth": 6, "n_estimators": 50, "verbose": -1},
                "evaluation": {"k_values": [5, 10]}
            },
            "xgb_listwise_upgraded": {
                "model_type": "xgboost",
                "hyperparameters": {
                    "objective": "rank:ndcg", "learning_rate": 0.05, "max_depth": 6, 
                    "n_estimators": 150, "tree_method": "hist", "subsample": 0.8, "colsample_bytree": 0.9
                },
                "evaluation": {"k_values": [5, 10]}
            },
            "lgb_listwise_upgraded": {
                "model_type": "lightgbm",
                "hyperparameters": {
                    "objective": "lambdarank", "learning_rate": 0.05, "num_leaves": 63, 
                    "max_depth": -1, "n_estimators": 150, "min_child_samples": 20, "verbose": -1
                },
                "evaluation": {"k_values": [5, 10]}
            }
        }

        # =====================================================================
        # SEQUENTIAL RUN EXECUTION LOOP
        # =====================================================================
        final_summary = {}
        for run_name, config in configs.items():
            print(f"\n=== Running Execution Suite: {run_name} ===")
            metrics = runner.run_experiment(
                run_name=run_name,
                model_config=config,
                train_data=train_data,
                val_data=val_data
            )
            final_summary[run_name] = {
                "NDCG@5": metrics.get("ndcg_5"),
                "NDCG@10": metrics.get("ndcg_10"),
                "MRR": metrics.get("mrr")
            }

        # Print clean baseline validation summary directly to console
        print("\n" + "="*60 + "\n FINAL Side-by-Side Performance Comparison\n" + "="*60)
        for model, scores in final_summary.items():
            print(f"{model.upper():<30} | NDCG@5: {scores['NDCG@5']:.4f} | NDCG@10: {scores['NDCG@10']:.4f} | MRR: {scores['MRR']:.4f}")
        print("="*60)

    except Exception as e:
        print(f"\n[Pipeline Crash] Error encountered: {str(e)}")