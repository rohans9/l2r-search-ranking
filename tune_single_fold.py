import gc
import time
import yaml
import logging
import itertools
import numpy as np
import lightgbm as lgb

# Import your custom modular pipeline components
from src.features.processors import QueryGroupScaler
from src.evaluation.metrics import IREvaluator

# ==========================================
# Configure the Terminal Logger
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_fold1_data():
    """Loads the pre-split binary files specifically from Fold1."""
    logger.info("Loading Fold1 training and validation binary chunks (mmap_mode='r')...")
    
    # Paths matching your exact directory structure
    X_train = np.load("data/processed/Fold1/X_train.npy", mmap_mode='r')
    y_train = np.load("data/processed/Fold1/y_train.npy", mmap_mode='r')
    q_train = np.load("data/processed/Fold1/qids_train.npy", mmap_mode='r')
    
    X_val = np.load("data/processed/Fold1/X_vali.npy", mmap_mode='r')
    y_val = np.load("data/processed/Fold1/y_vali.npy", mmap_mode='r')
    q_val = np.load("data/processed/Fold1/qids_vali.npy", mmap_mode='r')
    
    return X_train, y_train, q_train, X_val, y_val, q_val

def main():
    logger.info("=== Starting Exhaustive Single-Fold Grid Search ===")
    
    # Define the Exhaustive Grid Space
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
        'min_child_samples': [20, 50]
    }
    
    static_params = {
        'objective': 'lambdarank',
        'max_depth': -1,
        'n_estimators': 150,
        'verbose': -1,
        'n_jobs': 4  # Explicitly caps threads to prevent RAM overflow on your Air
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    logger.info(f"Total hyperparameter combinations mapped: {len(combinations)}")

    # Load data directly from pre-split Fold1 files
    X_train, y_train, q_train, X_val, y_val, q_val = load_fold1_data()
    _, train_group_sizes = np.unique(q_train, return_counts=True)
    _, val_group_sizes = np.unique(q_val, return_counts=True)

    best_score = -1.0
    best_params = None
    total_start_time = time.time()

    # Execute the Grid Search on Fold 1
    for idx, combo in enumerate(combinations, 1):
        combo_start_time = time.time()
        logger.info(f"--- [Combo {idx}/{len(combinations)}] Training Started on Fold1 (validation) ---")
        logger.info(f"Parameters: {combo}")
        
        current_params = {**static_params, **combo}

        # Context-Aware Scaling (Fitted on train, applied to val)
        scaler = QueryGroupScaler(strategy="zscore")
        X_train_scaled = scaler.fit_transform(X_train, train_group_sizes)
        X_val_scaled = scaler.fit_transform(X_val, val_group_sizes)

        # Initialize and Train
        ranker = lgb.LGBMRanker(**current_params, random_state=42)
        
        ranker.fit(X_train_scaled, y_train, group=train_group_sizes)

        # Predict and Evaluate
        predictions = ranker.predict(X_val_scaled)
        evaluator = IREvaluator()

        # Determine the exact method name
        eval_method = None
        if hasattr(evaluator, 'evaluate'):
            eval_method = evaluator.evaluate
        elif hasattr(evaluator, 'calculate_metrics'):
            eval_method = evaluator.calculate_metrics
            
        if eval_method is None:
            available_methods = [m for m in dir(evaluator) if not m.startswith('__')]
            logger.error(f"Available methods on IREvaluator: {available_methods}")
            raise AttributeError("Could not find 'evaluate' or 'calculate_metrics' on IREvaluator.")

        # Inspect the method parameters to find what you named the query ID argument
        import inspect
        sig = inspect.signature(eval_method)
        param_names = list(sig.parameters.keys())
        
        # Build the exact keyword argument dictionary your method expects
        eval_kwargs = {}
        
        # Assign y_true
        if 'y_true' in param_names: eval_kwargs['y_true'] = y_val
        elif 'labels' in param_names: eval_kwargs['labels'] = y_val
        
        # Assign y_pred
        if 'y_pred' in param_names: eval_kwargs['y_pred'] = predictions
        elif 'preds' in param_names: eval_kwargs['preds'] = predictions
        
        # Assign the query IDs dynamically by looking for common naming patterns
        qid_param = next((p for p in param_names if p in ['qids', 'query_ids', 'groups', 'query_group_ids']), None)
        if qid_param:
            eval_kwargs[qid_param] = q_val
        else:
            # Fallback if you used an unconventional parameter order or name
            logger.warning(f"Could not automatically map query ID parameter. Found signature: {param_names}")
            # Try passing positional arguments if keyword mapping completely misses
            eval_kwargs = dict(zip(param_names[1:], [y_val, predictions, q_val])) # skips 'self'

        # Execute the method safely with the dynamically built arguments
        metrics = eval_method(**eval_kwargs)
        
        # Robustly extract NDCG@10 from the metrics result
        ndcg_10 = None
        if isinstance(metrics, dict):
            # Normalizing keys to lowercase and stripping characters like '@' or '_'
            normalized_metrics = {str(k).lower().replace('@', '').replace('_', ''): v for k, v in metrics.items()}
            
            if 'ndcg10' in normalized_metrics:
                ndcg_10 = normalized_metrics['ndcg10']
            elif 'ndcg_10' in metrics: # fallback to exact match
                ndcg_10 = metrics['ndcg_10']
            else:
                # If they used a completely different naming convention, find any key with '10' and 'ndcg'
                for norm_k, original_v in normalized_metrics.items():
                    if 'ndcg' in norm_k and '10' in norm_k:
                        ndcg_10 = original_v
                        break
        elif isinstance(metrics, (int, float)):
            # If your evaluator just returns a single flat score directly
            ndcg_10 = metrics

        if ndcg_10 is None:
            logger.error(f"Could not automatically locate NDCG@10 in the evaluation output. Returned object: {metrics}")
            raise KeyError("Could not parse tracking metric from IREvaluator output.")

        combo_duration = time.time() - combo_start_time
        logger.info(f"Result [Combo {idx}]: Normalized NDCG@10 = {ndcg_10:.4f} (Took {combo_duration:.1f} seconds)")

        if ndcg_10 > best_score:
            logger.info(f"🌟 New Peak Reached! {ndcg_10:.4f} > {best_score:.4f}")
            best_score = ndcg_10
            best_params = combo

        # Explicit memory cleanup after each validation run
        del ranker
        gc.collect()
        logger.info(f"Memory cleared for Combo {idx}")

    # Final Output and Auto-Save
    total_duration = (time.time() - total_start_time) / 60
    logger.info("=============================================")
    logger.info("🏆 SINGLE-FOLD TUNING COMPLETE 🏆")
    logger.info(f"Total Execution Time: {total_duration:.2f} minutes")
    logger.info(f"Peak Validation NDCG@10: {best_score:.4f}")
    logger.info(f"Optimal Parameters: {best_params}")
    logger.info("=============================================")

    with open("config/lightgbm_lambda_mart.yaml", "w") as file:
        yaml.dump(best_params, file, default_flow_style=False)
        logger.info("✅ Frozen winning parameters to config/lightgbm_lambda_mart.yaml")

if __name__ == "__main__":
    main()