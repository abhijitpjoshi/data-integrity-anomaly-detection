
import pandas as pd
from sklearn.impute import KNNImputer
from utils.data_loader import load_data

# Load the dataset with missing values
df = load_data('data/sample_data.csv')

# Initialize the KNN Imputer
imputer = KNNImputer(n_neighbors=3)

# Impute missing values
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Show the result
print(df_imputed)
