import numpy as np
from sklearn import datasets as skds
from .registry import register_dataset


@register_dataset("moons")
def load_moons():
    X, y = skds.make_moons(n_samples=2000, noise=0.05, random_state=42)
    return X, y, {"name": "moons", "n_classes": 2}


@register_dataset("circles")
def load_circles():
    X, y = skds.make_circles(n_samples=2000, noise=0.05, factor=0.5, random_state=42)
    return X, y, {"name": "circles", "n_classes": 2}


@register_dataset("blobs")
def load_blobs():
    X, y = skds.make_blobs(n_samples=3000, centers=5, n_features=10, random_state=42)
    return X, y, {"name": "blobs", "n_classes": 5}


@register_dataset("swiss_roll")
def load_swiss_roll():
    X, t = skds.make_swiss_roll(n_samples=3000, noise=0.1, random_state=42)
    # Bin continuous t into 5 classes for label-based metrics
    y = np.digitize(t, np.percentile(t, [20, 40, 60, 80])).astype(int)
    return X, y, {"name": "swiss_roll", "n_classes": 5}


@register_dataset("s_curve")
def load_s_curve():
    X, t = skds.make_s_curve(n_samples=3000, noise=0.1, random_state=42)
    y = np.digitize(t, np.percentile(t, [20, 40, 60, 80])).astype(int)
    return X, y, {"name": "s_curve", "n_classes": 5}
