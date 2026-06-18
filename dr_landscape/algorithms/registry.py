from typing import Callable

ALGORITHMS: dict[str, Callable] = {}


def register_algorithm(name: str):
    def decorator(fn: Callable) -> Callable:
        ALGORITHMS[name] = fn
        return fn
    return decorator
