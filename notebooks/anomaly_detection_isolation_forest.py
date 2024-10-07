
import pandas as pd
from sklearn.ensemble import IsolationForest
from utils.data_loader import load_data
from utils.data_cleaning import preprocess_data

# Load and preprocess data
df = load_data('data/sample_data.csv')
df_cleaned = preprocess_data(df)

# Isolation Forest model for anomaly detection
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = iso_forest.fit_predict(df_cleaned[['feature1', 'feature2']])

# Output the anomalies
anomalies = df[df['anomaly'] == -1]
print(anomalies)
