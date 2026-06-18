from sklearn import datasets as skds
from .registry import register_dataset


@register_dataset("iris")
def load_iris():
    d = skds.load_iris()
    return d.data, d.target, {"name": "iris", "n_classes": 3}


@register_dataset("wine")
def load_wine():
    d = skds.load_wine()
    return d.data, d.target, {"name": "wine", "n_classes": 3}


@register_dataset("breast_cancer")
def load_breast_cancer():
    d = skds.load_breast_cancer()
    return d.data, d.target, {"name": "breast_cancer", "n_classes": 2}


@register_dataset("digits")
def load_digits():
    d = skds.load_digits()
    return d.data, d.target, {"name": "digits", "n_classes": 10}
