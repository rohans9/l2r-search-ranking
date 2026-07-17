# L2R Search Ranking

A compact, experiment-focused Learning-to-Rank (LTR) project for the MSLR-WEB30K dataset.

The repository includes:
- Data compilation from LETOR-style text to NumPy binaries
- Query-aware feature scaling
- Ranking models built on LightGBM and XGBoost
- IR evaluation metrics (NDCG, Precision, Recall, MRR, MAP)
- MLflow experiment tracking
- Unit and integration tests

## Project Structure

- `src/dataset/`: MSLR parsing and split validation
- `src/features/`: query-group feature processors
- `src/models/`: ranker interfaces and implementations
- `src/evaluation/`: IR metric computation
- `src/experiment/`: orchestration and MLflow logging
- `tests/`: test coverage for data/model/evaluation/runner components
- `config/`: model and base experiment settings
- `compile_dataset.py`: convert raw folds into binary arrays
- `run_cross_validation.py`: 5-fold LightGBM baseline evaluation
- `run_mslr_pipeline.py`: single-fold multi-model benchmark run
- `tune_single_fold.py`: fold-level LightGBM hyperparameter search

## Requirements

- Python 3.10+
- Dependencies from `requirements.txt`

Install:

```bash
python -m pip install -r requirements.txt
```

## Data Preparation

Place raw MSLR-WEB30K fold files under:

- `data/raw/MSLR-WEB30K/Fold1/`
- `data/raw/MSLR-WEB30K/Fold2/`
- `data/raw/MSLR-WEB30K/Fold3/`
- `data/raw/MSLR-WEB30K/Fold4/`
- `data/raw/MSLR-WEB30K/Fold5/`

Each fold should contain `train.txt`, `vali.txt`, and `test.txt`.

Compile raw text into `*.npy` arrays:

```bash
python compile_dataset.py
```

## Running Experiments

Run single-fold benchmark pipeline:

```bash
python run_mslr_pipeline.py
```

Run 5-fold LightGBM baseline:

```bash
python run_cross_validation.py
```

Run single-fold LightGBM tuning:

```bash
python tune_single_fold.py
```

## Testing

Run all tests:

```bash
python -m pytest -q
```

## Notes

- MLflow outputs are written to `mlruns/` by default.
- Temporary model artifacts are generated under `tmp_artifacts/` and cleaned after runs.
- Large raw/processed dataset files are intentionally excluded from version control.
