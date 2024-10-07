
import pytest
import pandas as pd
from models.isolation_forest_model import AnomalyDetectionModel

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'feature1': [1, 2, 1.5, 100, 1.2],
        'feature2': [3, 3.5, 2.9, 110, 3.1]
    })

def test_isolation_forest(sample_data):
    model = AnomalyDetectionModel(contamination=0.2)
    model.train(sample_data)
    preds = model.predict(sample_data)
    assert len(preds) == len(sample_data)
