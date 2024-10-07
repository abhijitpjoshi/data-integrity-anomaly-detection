
from sklearn.ensemble import IsolationForest

class AnomalyDetectionModel:
    def __init__(self, contamination=0.05, random_state=42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
    
    def train(self, X):
        return self.model.fit(X)
    
    def predict(self, X):
        return self.model.predict(X)
