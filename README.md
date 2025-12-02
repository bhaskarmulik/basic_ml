# basic_ml

A small collection of standalone Python scripts demonstrating core machine learning workflows on classic datasets:

- Binary classification on the Adult Income dataset
- Binary classification on the Titanic dataset
- Regression on the Boston Housing dataset
- Unsupervised learning on the Wholesale Customers dataset
- Model comparison on Adult Income using Gradient Boosting and Random Forest

Each script is self-contained and can be run directly from the command line.

---

## Contents

| File                              | Task type         | Dataset               | Description |
| --------------------------------- | ----------------- | --------------------- | ----------- |
| `adult_income.py`                 | Classification    | Adult Income (UCI)    | Baseline model to predict whether income exceeds a threshold (e.g. `>50K`) from demographic features. |
| `adult_income_gb.py`              | Classification    | Adult Income (UCI)    | Gradient Boosting classifier for Adult Income with improved performance vs a simple baseline. |
| `adult_income_rf.py`              | Classification    | Adult Income (UCI)    | Random Forest classifier for Adult Income, useful for comparison with Gradient Boosting. |
| `boston_housing_regression.py`    | Regression        | Boston Housing        | Predicts house prices from neighborhood and structural features using regression models. |
| `titanic_dataset.py`             | Classification    | Titanic (Kaggle)      | Predicts passenger survival using demographic and ticket-related features. |
| `wholesale_cust_unsupervised.py` | Unsupervised ML   | Wholesale Customers   | Applies clustering / unsupervised techniques to segment wholesale customers. |

> **Note:** Exact preprocessing steps, model hyperparameters and evaluation metrics are defined inside each script.

---

## Requirements

You’ll need a recent version of Python and typical ML / data libraries.

Suggested environment:

- Python 3.8+
- `pip` or `conda`

Python packages (likely needed by most scripts):

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn` *(if used for plotting in the scripts)*

Install dependencies (example with `pip`):

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
