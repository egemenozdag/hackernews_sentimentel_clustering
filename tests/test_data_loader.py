from unittest.mock import Mock
from data_loader import load_data

def test_load_data_mocking():
    mock_db = Mock()
    mock_db.query.return_value = [{"text": "Sample comment", "sentiment": "positive"}]
    data = load_data(mock_db)
    assert len(data) == 1
    assert data[0]["sentiment"] == "positive"

