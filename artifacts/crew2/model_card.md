# Model Card — Customer Churn

## Model details

- Model task: binary churn prediction for customer retention outreach
- Selected model: random_forest
- Selection rule: highest 5-fold cross-validated ROC-AUC on the training set
- Decision threshold: 0.5
- Contract version: 1.0
- Random seed: 42
- Total features: 26
- Training rows: 4059
- Test rows: 1015
- Training churn rate: 16.6%
- Test churn rate: 16.6%

The model uses customer activity, satisfaction, order, and payment-related features. Gender and MaritalStatus were excluded from the feature set.

## Intended use

This model is intended to rank customers for retention outreach.

Appropriate uses:
- Prioritizing customers for outreach, offers, or intervention
- Ranking customers by churn risk for operational follow-up
- Supporting retention campaign planning

Out-of-scope uses:
- Hiring, lending, insurance, housing, or other decisions unrelated to retention
- Any use as a sole basis for automated adverse action
- Any use that treats the score as a causal explanation of churn
- Any use outside the dataset contract and feature engineering pipeline

## Training data

The model was trained under contract version 1.0.

Contract assumptions:
- One row per customer; CustomerID is unique.
- Churn is the target: 1 = customer churned, 0 = customer stayed.
- Missing numeric values were filled with the median of the respective column.
- 556 duplicate rows were removed during cleaning.
- Inconsistent category spellings were merged: Phone was changed to Mobile Phone, CC was changed to Credit Card, and COD was changed to Cash on Delivery.

Feature set:
- Numeric columns used: Tenure, CityTier, WarehouseToHome, HourSpendOnApp, NumberOfDeviceRegistered, SatisfactionScore, NumberOfAddress, Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount
- One-hot encoded text columns: PreferredLoginDevice, PreferredPaymentMode, PreferedOrderCat
- Engineered features:
  - is_new_customer: 1 if Tenure <= 1 month
  - cashback_per_order: CashbackAmount / OrderCount, with OrderCount floored at 1
  - coupons_per_order: CouponUsed / OrderCount, with OrderCount floored at 1
  - days_since_order_per_tenure: DaySinceLastOrder / (Tenure + 1)

Validation passed with 71 contract checks and 0 violations.

## Evaluation

The selected model is random_forest because it had the highest 5-fold cross-validated ROC-AUC on the training set.

Model comparison on the held-out test set:

- logistic_regression
  - CV ROC-AUC: 89.4% ± 1.5%
  - ROC-AUC: 89.0%
  - PR-AUC: 68.3%
  - Precision: 47.8%
  - Recall: 83.3%
  - F1: 60.7%
- random_forest
  - CV ROC-AUC: 96.0% ± 0.7%
  - ROC-AUC: 98.0%
  - PR-AUC: 92.6%
  - Precision: 75.9%
  - Recall: 91.7%
  - F1: 83.0%
- gradient_boosting
  - CV ROC-AUC: 95.9% ± 1.2%
  - ROC-AUC: 98.5%
  - PR-AUC: 95.2%
  - Precision: 80.9%
  - Recall: 93.5%
  - F1: 86.7%

Top permutation importance features:
- Complain: 4.5%
- Tenure: 2.6%
- NumberOfAddress: 2.5%
- is_new_customer: 1.8%
- SatisfactionScore: 1.5%
- CashbackAmount: 1.3%
- cashback_per_order: 1.1%
- CityTier: 1.0%
- days_since_order_per_tenure: 1.0%
- WarehouseToHome: 1.0%

Important note: importance does not show direction. A feature with high importance may matter a lot to the model, but this does not tell us whether higher values increase or decrease churn risk.

## Limitations

There is mild leakage in the preprocessing pipeline: Crew 1 filled missing numeric values with medians computed on the full dataset before the train/test split. That means the test distribution influenced preprocessing. This is a mild leak, not a catastrophic one, but it can make reported performance slightly optimistic.

Other limitations:
- The model is evaluated on one fixed train/test split; results may vary on new data.
- Very strong test metrics may not fully transfer to production data.
- Permutation importance does not tell us the direction of effects.
- The model can only learn patterns present in the training data and the contract-defined features.
- Excluding sensitive columns does not guarantee fairness.

## Ethical considerations

Gender and MaritalStatus were excluded from the features. That reduces direct use of those columns, but it does not guarantee fairness.

Fairness by group, reported as actual churn rate, flagged rate, and recall:

Gender:
- Female
  - Customers: 404
  - Actual churn rate: 18.6%
  - Flagged rate: 22.8%
  - Recall: 85.3%
- Male
  - Customers: 611
  - Actual churn rate: 15.2%
  - Flagged rate: 18.2%
  - Recall: 96.8%

MaritalStatus:
- Divorced
  - Customers: 157
  - Actual churn rate: 17.2%
  - Flagged rate: 20.4%
  - Recall: 96.3%
- Married
  - Customers: 540
  - Actual churn rate: 11.5%
  - Flagged rate: 16.9%
  - Recall: 95.2%
- Single
  - Customers: 318
  - Actual churn rate: 24.8%
  - Flagged rate: 25.2%
  - Recall: 87.3%

Plainly, the model serves Single customers worse on recall than Divorced or Married customers. It also has lower recall for Female customers than Male customers. These differences should be monitored before deployment and during use.

## How to use

The saved bundle can be loaded as follows:

```python
import joblib

bundle = joblib.load("artifacts/crew2/model.joblib")
model = bundle["model"]
features = bundle["features"]
threshold = bundle["threshold"]
contract_version = bundle["contract_version"]
```

New data must pass the dataset contract gate and go through the same feature engineering first. After that, score the model using the saved feature list and apply the threshold to convert probabilities into flagged customers.

## Appendix — metrics (generated from metrics.json)

| Model | CV ROC-AUC (train) | Test ROC-AUC | Test PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| logistic_regression | 0.8938 ± 0.0149 | 0.8901 | 0.6831 | 0.4778 | 0.8333 | 0.6074 |
| random_forest **(selected)** | 0.9596 ± 0.0070 | 0.9802 | 0.9259 | 0.7586 | 0.9167 | 0.8302 |
| gradient_boosting | 0.9590 ± 0.0124 | 0.9853 | 0.9522 | 0.8093 | 0.9345 | 0.8674 |
