import json
import os
import numpy as np

RESULTS_DIR = "results"


def _algo_dir(dataset: str, algo: str, results_dir: str) -> str:
    return os.path.join(results_dir, dataset, algo)


def save_embedding(dataset: str, algo: str, embedding: np.ndarray, results_dir: str = RESULTS_DIR):
    path = _algo_dir(dataset, algo, results_dir)
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "embedding.npy"), embedding)


def load_embedding(dataset: str, algo: str, results_dir: str = RESULTS_DIR) -> np.ndarray:
    return np.load(os.path.join(_algo_dir(dataset, algo, results_dir), "embedding.npy"))


def save_metrics(dataset: str, algo: str, metrics: dict, results_dir: str = RESULTS_DIR):
    path = _algo_dir(dataset, algo, results_dir)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(dataset: str, algo: str, results_dir: str = RESULTS_DIR) -> dict:
    with open(os.path.join(_algo_dir(dataset, algo, results_dir), "metrics.json")) as f:
        return json.load(f)


def metrics_exist(dataset: str, algo: str, results_dir: str = RESULTS_DIR) -> bool:
    return os.path.exists(os.path.join(_algo_dir(dataset, algo, results_dir), "metrics.json"))


def save_dataset_meta(dataset: str, meta: dict, results_dir: str = RESULTS_DIR):
    path = os.path.join(results_dir, dataset)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def results_to_dataframe(results_dir: str = RESULTS_DIR):
    import pandas as pd
    rows = []
    if not os.path.isdir(results_dir):
        return pd.DataFrame(columns=["dataset", "algorithm", "metric", "value"])
    for dataset in sorted(os.listdir(results_dir)):
        d_path = os.path.join(results_dir, dataset)
        if not os.path.isdir(d_path):
            continue
        for algo in sorted(os.listdir(d_path)):
            metrics_path = os.path.join(d_path, algo, "metrics.json")
            if not os.path.isfile(metrics_path):
                continue
            with open(metrics_path) as f:
                metrics = json.load(f)
            for metric, value in metrics.items():
                rows.append({"dataset": dataset, "algorithm": algo, "metric": metric, "value": value})
    return pd.DataFrame(rows)
