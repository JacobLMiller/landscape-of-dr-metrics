from sklearn.decomposition import PCA
from .registry import register_algorithm


@register_algorithm("pca")
def run_pca(X):
    return PCA(n_components=2, random_state=42).fit_transform(X)
