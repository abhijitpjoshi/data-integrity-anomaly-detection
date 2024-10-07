
# Data Integrity and Anomaly Detection Using Machine Learning

This repository implements various machine learning models to address data integrity issues, including anomaly detection, missing value imputation, rule-based validation, and duplicate detection. The models and techniques are geared toward large-scale datasets and complex environments where data consistency is critical.

## Overview

### Key Features:
- **Anomaly Detection with Isolation Forest**: Detect outliers that may signal data integrity issues.
- **Missing Value Imputation with K-Nearest Neighbors (KNN)**: Automatically fill missing data points.
- **Business Rule Enforcement with Linear Regression**: Check for unexpected deviations from expected trends.
- **Duplicate Detection with TF-IDF and Cosine Similarity**: Find and handle duplicate entries in text data.

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/data-integrity-anomaly-detection.git
    cd data-integrity-anomaly-detection
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the example notebooks to try out each technique:
    ```bash
    python notebooks/anomaly_detection_isolation_forest.py
    python notebooks/missing_value_imputation_knn.py
    ```

4. Test your models with unit tests:
    ```bash
    pytest tests/
    ```

## Example Output

### Isolation Forest for Anomaly Detection:

```python
Processing batch...
Anomalies detected at indices: [12, 57, 102, 230]
```

## Dependencies

- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `pytest`

## License

MIT License - see the [LICENSE](LICENSE) file for details.
