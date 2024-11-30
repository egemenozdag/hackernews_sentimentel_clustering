import pytest
from clustering import train_kmeans

def test_train_kmeans_valid_data():
    data = [[1, 2], [2, 3], [3, 4]]
    n_clusters = 2
    model = train_kmeans(data, n_clusters)
    assert model.n_clusters == n_clusters
    assert len(model.cluster_centers_) == n_clusters
