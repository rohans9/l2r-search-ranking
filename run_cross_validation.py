import os
import numpy as np
import lightgbm as lgb
from src.evaluation.metrics import IREvaluator
from src.features.processors import QueryGroupScaler

def evaluate_fold(fold_idx: int):
    # Read directly from the newly organized processed path
    fold_dir = f"data/processed/Fold{fold_idx}"
    
    if not os.path.exists(fold_dir):
        raise FileNotFoundError(f"Processed binaries not found at {fold_dir}. Run compiler first.")
    
    # 1. Load memory-mapped features and full arrays
    X_train = np.load(os.path.join(fold_dir, "X_train.npy"), mmap_mode='r')
    y_train = np.load(os.path.join(fold_dir, "y_train.npy"))
    qids_train = np.load(os.path.join(fold_dir, "qids_train.npy"))
    
    X_val = np.load(os.path.join(fold_dir, "X_vali.npy"), mmap_mode='r')
    y_val = np.load(os.path.join(fold_dir, "y_vali.npy"))
    qids_val = np.load(os.path.join(fold_dir, "qids_vali.npy"))
    
    # 2. Extract group constraints per query block
    _, train_groups = np.unique(qids_train, return_counts=True)
    _, val_groups = np.unique(qids_val, return_counts=True)
    
    # 3. Apply Local Group Scale Normalization
    scaler = QueryGroupScaler(strategy="zscore")
    X_train_scaled = scaler.fit_transform(X_train, train_groups)
    X_val_scaled = scaler.fit_transform(X_val, val_groups)
    
    # 4. Train LightGBM Listwise LambdaMART Champion
    model = lgb.LGBMRanker(
        objective="lambdarank", 
        learning_rate=0.05, 
        num_leaves=63,
        max_depth=-1, 
        n_estimators=150, 
        min_child_samples=20, 
        verbose=-1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train, group=train_groups)
    
    # 5. Evaluate Performance 
    val_scores = model.predict(X_val_scaled)
    evaluator = IREvaluator(k_values=[5, 10])
    metrics = evaluator.evaluate(y_val, val_scores, qids_val)
    
    # Standardize keys to lowercase without symbols to prevent KeyErrors
    clean_metrics = {str(k).lower().replace("@", "_"): v for k, v in metrics.items()}
    
    # Safe lookups with fallbacks
    ndcg_5 = clean_metrics.get("ndcg_5") or clean_metrics.get("ndcg_5", 0.0)
    ndcg_10 = clean_metrics.get("ndcg_10") or clean_metrics.get("ndcg_10", 0.0)
    mrr = clean_metrics.get("mrr") or clean_metrics.get("mrr", 0.0)
    
    return ndcg_5, ndcg_10, mrr

if __name__ == "__main__":
    ndcg5_scores = []
    ndcg10_scores = []
    mrr_scores = []
    
    print("=== Starting Production-Grade 5-Fold Cross-Validation Matrix ===")
    for fold in range(1, 6):
        print(f"\n--> Crunching Evaluation Loop for Fold {fold}/5...")
        try:
            n5, n10, mrr = evaluate_fold(fold)
            ndcg5_scores.append(n5)
            ndcg10_scores.append(n10)
            mrr_scores.append(mrr)
            print(f"    Fold {fold} Complete -> NDCG@5: {n5:.4f} | NDCG@10: {n10:.4f} | MRR: {mrr:.4f}")
        except Exception as e:
            print(f"    [Error] Failed evaluating fold {fold}: {e}")
        
    print("\n" + "="*60 + "\n FINAL CROSS-VALIDATION ROBUSTNESS REPORT\n" + "="*60)
    print(f"Mean NDCG@5:  {np.mean(ndcg5_scores):.4f} ± {np.std(ndcg5_scores):.4f}")
    print(f"Mean NDCG@10: {np.mean(ndcg10_scores):.4f} ± {np.std(ndcg10_scores):.4f}")
    print(f"Mean MRR:     {np.mean(mrr_scores):.4f} ± {np.std(mrr_scores):.4f}")
    print("="*60)
    print("This cross-fold baseline is ready for portfolio architectural display.")