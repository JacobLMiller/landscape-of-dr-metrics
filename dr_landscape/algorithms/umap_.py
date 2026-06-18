from umap import UMAP
from .registry import register_algorithm


@register_algorithm("umap")
def run_umap(X):
    return UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)
