from .registry import DATASETS, register_dataset
from . import synthetic, tabular, benchmarks

__all__ = ["DATASETS", "register_dataset"]
