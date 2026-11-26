# Water Quality and Potability Assessment

## Project Overview
This project uses Machine Learning to predict whether water is potable or non-potable based on its physical and chemical properties.

## Dataset
- Dataset: Water Potability Dataset
- Total records: 3,276
- Features: 9 water quality parameters
- Target: Potability
  - 0 = Not Potable
  - 1 = Potable

## Machine Learning Algorithms

### 1. Decision Tree Classifier
The original implementation uses:
- Median imputation
- Decision Tree Classifier
- GridSearchCV for hyperparameter tuning
- Stratified 5-fold cross-validation

### 2. Decision Tree with SMOTE
The second implementation uses:
- Mean imputation
- StandardScaler
- SMOTE for handling class imbalance
- Decision Tree Classifier
- GridSearchCV for hyperparameter tuning
- Stratified 5-fold cross-validation

## Project Files
- `main.py` – Original Decision Tree implementation
- `DT_with_smote.py` – SMOTE-based Decision Tree implementation
- `water_potability.csv` – Dataset
- `requirements.txt` – Required Python libraries

## How to Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the original Decision Tree:

```bash
python main.py
```

Run the SMOTE-based Decision Tree:

```bash
python DT_with_smote.py
```

## Evaluation Metrics
Both models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Feature Importance

## Objective
To compare the performance of a standard Decision Tree classifier with a Decision Tree classifier using SMOTE for handling class imbalance.

*Note: Model predictions are estimates and should not replace laboratory water-quality testing.*
