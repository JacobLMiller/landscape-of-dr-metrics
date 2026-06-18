import numpy as np
import tensorflow as tf
import tensorflow_projection_qm.metrics as pqm


def compute_all_metrics(X: np.ndarray, X_2d: np.ndarray, y: np.ndarray, k: int = 7) -> dict[str, float]:
    """Run every metric in tensorflow-projection-qm and return a flat dict of floats."""
    n_classes = int(y.max()) + 1 if y is not None else None

    X_tf = tf.cast(X, tf.float32)
    X_2d_tf = tf.cast(X_2d, tf.float32)
    y_tf = tf.cast(y, tf.int32) if y is not None else None

    results = pqm.run_all_metrics(X_tf, X_2d_tf, y_tf, k=k, n_classes=n_classes, as_numpy=True)

    # Flatten any array-valued metrics to their mean
    flat: dict[str, float] = {}
    for name, val in results.items():
        if isinstance(val, np.ndarray):
            flat[name] = float(np.mean(val))
        else:
            flat[name] = float(val)
    return flat
