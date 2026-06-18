import numpy as np
from .registry import register_dataset

MAX_SAMPLES = 5000


def _stratified_subsample(X, y, max_samples, seed=42):
    if len(X) <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    per_class = max(1, max_samples // len(classes))
    indices = []
    for c in classes:
        idx = np.where(y == c)[0]
        n = min(per_class, len(idx))
        indices.append(rng.choice(idx, size=n, replace=False))
    indices = np.concatenate(indices)
    # top up to max_samples if rounding left us short
    if len(indices) < max_samples:
        remaining = np.setdiff1d(np.arange(len(X)), indices)
        extra = rng.choice(remaining, size=min(max_samples - len(indices), len(remaining)), replace=False)
        indices = np.concatenate([indices, extra])
    return X[indices], y[indices]


@register_dataset("mnist")
def load_mnist():
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = np.concatenate([X_train, X_test]).reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y = np.concatenate([y_train, y_test])
    X, y = _stratified_subsample(X, y, MAX_SAMPLES)
    return X, y, {"name": "mnist", "n_classes": 10}


@register_dataset("fashion_mnist")
def load_fashion_mnist():
    from tensorflow.keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X = np.concatenate([X_train, X_test]).reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y = np.concatenate([y_train, y_test])
    X, y = _stratified_subsample(X, y, MAX_SAMPLES)
    return X, y, {"name": "fashion_mnist", "n_classes": 10}
