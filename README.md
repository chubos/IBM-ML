# Rainfall Prediction Classifier (Melbourne, Australia)

An end-to-end Machine Learning project that predicts daily rainfall in the Melbourne metropolitan area using historical observations from the Australian Bureau of Meteorology.

## Project Overview

* **Objective:** Predict whether it will rain on a given day using historical weather metrics from the preceding day.
* **Key Challenge Addressed:** Mitigated data leakage by forecasting the current day's weather based strictly on metrics observed up to yesterday.
* **Target Granularity:** Filtered to geographic clusters around Melbourne (Melbourne, Melbourne Airport, Watsonia) to preserve local meteorological consistency.

## Pipeline & Tech Stack

* **Language & Core Libraries:** Python, `pandas`, `NumPy`, `scikit-learn`, `matplotlib`, `seaborn`
* **Feature Engineering:** Extracted seasonal trends (`Season`) from observation dates; one-hot encoded categorical variables (`WindDir`, `Location`, etc.).
* **Preprocessing:** Modular `ColumnTransformer` with `StandardScaler` for numerical metrics and `OneHotEncoder(handle_unknown='ignore')` for categorical features.
* **Optimization:** `GridSearchCV` combined with `StratifiedKFold` (5-fold) cross-validation to account for class imbalance.

## Models Evaluated

1. **Random Forest Classifier:**
   * Tuned over tree depth, split constraints, and number of estimators.
   * **Test Accuracy:** ~84%
   * **True Positive Rate (Recall for Rain):** ~51%
2. **Logistic Regression (Liblinear):**
   * Tuned with L1/L2 penalties and balanced class weighting.
   * **Test Accuracy:** ~83%
   * **True Positive Rate:** Evaluated against baseline to significantly penalize false negatives.

## Key Insights

* **Feature Importance:** Late-afternoon humidity (`Humidity3pm`) and barometric pressure trends were identified as the strongest predictors of precipitation events.
* **Metric Selection:** Due to class imbalance (dry days dominate ~76% of the dataset), raw accuracy is deceptive. A high True Positive Rate (Recall) was prioritized to minimize missed rainy days.

## How to Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/chubos/IBM-ML.git](https://github.com/chubos/IBM-ML.git)
   cd IBM-ML