# Model Card — Customer Churn

## Model details

- Model name: customer churn ranker
- Selected model: random forest
- Selection rule: highest 5-fold cross-validated ROC-AUC on the training set
- Decision threshold: 0.5
- Contract version: 1.0
- Seed: 42
- Primary key: CustomerID
- Target: Churn, where 1 = customer churned and 0 = customer stayed
- Number of features: 26
- Train rows: 4059
- Test rows: 1015
- Train churn rate: 16.6%
- Test churn rate: 16.6%

## Intended use

This model is intended to rank customers for retention outreach. It is meant to help prioritize who should receive outreach, not to make fully automated decisions about customers.

Out of scope uses include:
- approving or denying customer service
- setting prices, discounts, or credit terms
- making employment, housing, insurance, or other high-stakes decisions
- using the model outside the dataset contract and feature pipeline described here
- using the model without applying the same feature engineering and contract checks used during training

## Training data

The model was trained and evaluated under contract version 1.0.

Dataset contract assumptions:
- One row per customer; CustomerID is unique.
- Churn is the target: 1 = customer churned, 0 = customer stayed.
- All duplicate rows were removed from the dataset, totaling 556 duplicates dropped.
- Missing values were filled with the median for the following columns: Tenure (9 months), WarehouseToHome (13 km), HourSpendOnApp (3 hours), OrderAmountHikeFromlastYear (15 percent), CouponUsed (1 count), OrderCount (2 count), and DaySinceLastOrder (3 days).
- Inconsistent category spellings were standardized: Phone was merged to Mobile Phone, CC to Credit Card, COD to Cash on Delivery, and Mobile to Mobile Phone.

The model uses these feature groups:
- Numeric columns: Tenure, CityTier, WarehouseToHome, HourSpendOnApp, NumberOfDeviceRegistered, SatisfactionScore, NumberOfAddress, Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount
- One-hot encoded text columns: PreferredLoginDevice, PreferredPaymentMode, PreferedOrderCat
- Engineered features: is_new_customer, cashback_per_order, coupons_per_order, days_since_order_per_tenure

Sensitive columns were excluded from the model:
- Gender
- MaritalStatus

One engineered feature deserves special note:
- is_new_customer is 1 if Tenure <= 1 month, and churn is about 51% in this group versus less than 10% after that point.

## Evaluation

The selected random forest performed as follows on the test set:
- ROC-AUC: 98.0%
- PR-AUC: 92.6%
- Precision: 75.9%
- Recall: 91.7%
- F1: 83.0%

For context, the other models were:
- Logistic regression: ROC-AUC 89.0%, PR-AUC 68.3%, precision 47.8%, recall 83.3%, F1 60.7%
- Gradient boosting: ROC-AUC 98.5%, PR-AUC 95.2%, precision 80.9%, recall 93.5%, F1 86.7%

The random forest was selected because it had the highest 5-fold cross-validated ROC-AUC on the training set:
- Random forest: 95.96%
- Gradient boosting: 95.90%
- Logistic regression: 89.38%

Interpretation for retention use:
- Precision tells you, among customers flagged for outreach, how many actually churned.
- Recall tells you, among customers who actually churned, how many were flagged.

At the operating point used here, the model catches most churners while also limiting wasted outreach.

The top features by permutation importance were:
- Complain: 4.48%
- Tenure: 2.61%
- NumberOfAddress: 2.51%
- is_new_customer: 1.78%
- SatisfactionScore: 1.52%
- CashbackAmount: 1.34%
- cashback_per_order: 1.08%
- CityTier: 1.04%
- days_since_order_per_tenure: 1.02%
- WarehouseToHome: 0.95%

Permutation importance shows which features matter most to prediction, but it does not show direction. It does not tell you whether a feature increases or decreases churn risk.

## Limitations

- There was mild leakage from Crew 1’s median imputation on the full dataset before the train/test split. This means some test-set information influenced preprocessing. The reported performance may therefore be slightly optimistic.
- The model was evaluated at a fixed threshold of 0.5. Different thresholds would change precision and recall.
- Importance values do not show direction, so they should not be interpreted as causal effects or as evidence that a feature increases or decreases churn.
- High test performance should still be interpreted cautiously because of the preprocessing leak.
- The model is only as valid as the contract it was trained under; new data must satisfy the same contract and feature pipeline.

## Ethical considerations

Gender and MaritalStatus were excluded from the features. Excluding a column does not guarantee fairness.

Fairness-by-group results for Gender:
- Female: 404 customers, actual churn rate 18.6%, flagged rate 22.8%, recall 85.3%
- Male: 611 customers, actual churn rate 15.2%, flagged rate 18.2%, recall 96.8%

Fairness-by-group results for MaritalStatus:
- Divorced: 157 customers, actual churn rate 17.2%, flagged rate 20.4%, recall 96.3%
- Married: 540 customers, actual churn rate 11.5%, flagged rate 16.9%, recall 95.2%
- Single: 318 customers, actual churn rate 24.8%, flagged rate 25.2%, recall 87.3%

Plainly, the model serves some groups worse:
- For Gender, recall is worse for Female than for Male, and the flagged rate is also higher for Female.
- For MaritalStatus, recall is worse for Single than for Divorced or Married.
- The highest actual churn rate is among Single customers, but the model still recalls that group worse than the other marital-status groups.

These are findings from the evaluation, not explanations. They show that removing Gender and MaritalStatus from the feature set did not eliminate performance differences across those groups.

## How to use

```python
import joblib

bundle = joblib.load("artifacts/crew2/model.joblib")
model = bundle["model"]
features = bundle["features"]
threshold = bundle["threshold"]
contract_version = bundle["contract_version"]
```

New data must pass the dataset contract gate and go through the same feature engineering first. In particular, it must satisfy the contract version 1.0 assumptions, and it must be transformed to the same 26-feature schema before scoring.

## Appendix — metrics (generated from metrics.json)

| Model | CV ROC-AUC (train) | Test ROC-AUC | Test PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| logistic_regression | 0.8938 ± 0.0149 | 0.8901 | 0.6831 | 0.4778 | 0.8333 | 0.6074 |
| random_forest **(selected)** | 0.9596 ± 0.0070 | 0.9802 | 0.9259 | 0.7586 | 0.9167 | 0.8302 |
| gradient_boosting | 0.9590 ± 0.0124 | 0.9853 | 0.9522 | 0.8093 | 0.9345 | 0.8674 |
