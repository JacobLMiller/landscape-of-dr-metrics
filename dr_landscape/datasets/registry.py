from typing import Callable

DATASETS: dict[str, Callable] = {}


def register_dataset(name: str):
    def decorator(fn: Callable) -> Callable:
        DATASETS[name] = fn
        return fn
    return decorator
