import os
import time
import traceback
from joblib import Parallel, delayed

from .datasets import DATASETS
from .algorithms import ALGORITHMS
from .metrics import compute_all_metrics
from . import storage


def run_pair(dataset_name: str, algo_name: str, results_dir: str, rerun: bool):
    if not rerun and storage.metrics_exist(dataset_name, algo_name, results_dir):
        print(f"  [skip] {dataset_name} / {algo_name} (cached)")
        return

    print(f"  [run]  {dataset_name} / {algo_name}")
    t0 = time.time()

    X, y, meta = DATASETS[dataset_name]()
    meta.update({"n_samples": int(X.shape[0]), "n_features": int(X.shape[1])})
    storage.save_dataset_meta(dataset_name, meta, results_dir)

    embedding = ALGORITHMS[algo_name](X)
    storage.save_embedding(dataset_name, algo_name, embedding, results_dir)

    metrics = compute_all_metrics(X, embedding, y)
    storage.save_metrics(dataset_name, algo_name, metrics, results_dir)

    elapsed = time.time() - t0
    print(f"  [done] {dataset_name} / {algo_name} ({elapsed:.1f}s, {len(metrics)} metrics)")


def run_experiment(
    datasets: list[str] | None = None,
    algorithms: list[str] | None = None,
    results_dir: str = "results",
    rerun: bool = False,
    n_jobs: int = 1,
):
    dataset_names = datasets or list(DATASETS.keys())
    algo_names = algorithms or list(ALGORITHMS.keys())

    missing_datasets = [d for d in dataset_names if d not in DATASETS]
    missing_algos = [a for a in algo_names if a not in ALGORITHMS]
    if missing_datasets:
        raise ValueError(f"Unknown datasets: {missing_datasets}. Available: {list(DATASETS.keys())}")
    if missing_algos:
        raise ValueError(f"Unknown algorithms: {missing_algos}. Available: {list(ALGORITHMS.keys())}")

    pairs = [(d, a) for d in dataset_names for a in algo_names]
    print(f"Running {len(pairs)} pairs ({len(dataset_names)} datasets x {len(algo_names)} algorithms)")
    os.makedirs(results_dir, exist_ok=True)

    def _safe_run(d, a):
        try:
            run_pair(d, a, results_dir, rerun)
        except Exception:
            print(f"  [ERROR] {d} / {a}")
            traceback.print_exc()

    if n_jobs == 1:
        for d, a in pairs:
            _safe_run(d, a)
    else:
        Parallel(n_jobs=n_jobs)(delayed(_safe_run)(d, a) for d, a in pairs)
