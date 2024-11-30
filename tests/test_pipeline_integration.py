from clustering import train_kmeans
from data_loader import load_data

def test_pipeline_integration():
    mock_data = [[1, 2], [2, 3], [3, 4]]
    model = train_kmeans(mock_data, n_clusters=2)
    assert model.n_clusters == 2
