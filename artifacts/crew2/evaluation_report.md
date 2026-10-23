# Model Evaluation Report

## Summary

We compared three churn model variations using the same feature set and the same fixed split. The model was selected by the highest 5-fold cross-validated ROC-AUC on the training set, not by test performance.

By that rule, the selected model is **random_forest** with a training CV ROC-AUC of **0.9596**. The **gradient_boosting** model was very close at **0.9590**, and **logistic_regression** trailed at **0.8938**.

On the held-out test set, the strongest overall discrimination and classification metrics were actually achieved by **gradient_boosting**:
- ROC-AUC: **0.9853**
- PR-AUC: **0.9522**
- Precision: **0.8093**
- Recall: **0.9345**
- F1: **0.8674**

However, the selected model remains **random_forest** because test results are not allowed to influence model choice. That keeps the evaluation honest and avoids tuning to the test set, even if another model looks better there.

## Setup

The dataset was split into:
- Train: **4059** rows
- Test: **1015** rows

The churn rate was very similar across splits:
- Train churn rate: **16.6%**
- Test churn rate: **16.6%**

The feature set used **26** features. Sensitive columns **Gender** and **MaritalStatus** were excluded from training, and fairness was checked afterward by those groups.

Selection rule:
- Pick the model with the highest **5-fold cross-validated ROC-AUC on the training set**
- Do **not** use the test set to choose the model

Why this matters:
- The test set is meant to be a final, untouched check on generalization.
- If we choose based on test results, we leak evaluation information into model selection and make the test score overly optimistic.
- Even if a model scores higher on test, that does not justify selecting it after the fact.

One caveat in the pipeline is worth stating plainly: **Crew 1 filled missing values with medians computed on the full dataset before the train/test split**. That is a mild form of leakage because information from the test distribution influences preprocessing. It is not a catastrophic leak, but it can make results a little better than a strictly leak-free pipeline.

## Model comparison

### Logistic regression
- CV ROC-AUC: **0.8938** ± **0.0149**
- Test ROC-AUC: **0.8901**
- Test PR-AUC: **0.6831**
- Precision: **0.4778**
- Recall: **0.8333**
- F1: **0.6074**

This model is clearly the weakest of the three. It still catches many churners, but it produces a lot of false positives relative to the other models.

### Random forest
- CV ROC-AUC: **0.9596** ± **0.0070**
- Test ROC-AUC: **0.9802**
- Test PR-AUC: **0.9259**
- Precision: **0.7586**
- Recall: **0.9167**
- F1: **0.8302**

This was the selected model because it had the best cross-validated ROC-AUC on the training set. It also generalized well on the test set.

### Gradient boosting
- CV ROC-AUC: **0.9590** ± **0.0124**
- Test ROC-AUC: **0.9853**
- Test PR-AUC: **0.9522**
- Precision: **0.8093**
- Recall: **0.9345**
- F1: **0.8674**

This model achieved the strongest test metrics across the board. But because the test set is not used for selection, it remains a comparison point rather than the chosen model.

### How to read precision and recall for a retention campaign

- **Precision** answers: of the customers we flagged for retention, how many actually churned?
  - Higher precision means fewer wasted offers, coupons, or outreach contacts.
  - Lower precision means more retention spend goes to customers who would not have churned.

- **Recall** answers: of the customers who actually churned, how many did we catch?
  - Higher recall means fewer missed churners.
  - Lower recall means more at-risk customers slip through without intervention.

For example:
- **logistic_regression** has precision **0.4778** and recall **0.8333**. That means it finds many churners, but nearly half of the flagged customers are false positives.
- **random_forest** improves precision to **0.7586** while keeping recall at **0.9167**, a much better tradeoff.
- **gradient_boosting** is best on both, with precision **0.8093** and recall **0.9345**.

The confusion matrices tell the same story:
- logistic_regression: **153** false positives, **28** false negatives
- random_forest: **49** false positives, **14** false negatives
- gradient_boosting: **37** false positives, **11** false negatives

## Selected model

The selected model is **random_forest**.

Why it was selected:
- It had the highest 5-fold cross-validated ROC-AUC on the training set: **0.9596**
- That is the agreed selection criterion
- Its CV result was also stable, with standard deviation **0.0070**, which is reassuring

Why not gradient_boosting, despite its stronger test metrics:
- Because the test set must not be used to pick the winner
- The difference in test performance may be real, but it may also reflect normal variance, the particular split, or mild leakage effects elsewhere in the pipeline
- The correct choice is still the model that won the training cross-validation contest

## What drives the predictions

The top features by permutation importance were:

1. **Complain**: **0.0448**
2. **Tenure**: **0.0261**
3. **NumberOfAddress**: **0.0251**
4. **is_new_customer**: **0.0178**
5. **SatisfactionScore**: **0.0152**
6. **CashbackAmount**: **0.0134**
7. **cashback_per_order**: **0.0108**
8. **CityTier**: **0.0104**
9. **days_since_order_per_tenure**: **0.0102**
10. **WarehouseToHome**: **0.0095**

Important interpretation note:
- Permutation importance tells us **how much a feature matters** to the model’s predictions.
- It does **not** tell us the **direction** of the effect.
- We should not claim that a feature “increases churn” or “reduces churn” based on importance alone.

What this suggests at a high level:
- Customer complaint behavior is the strongest signal in the model.
- Tenure and new-customer status matter substantially, which is consistent with early-life churn risk.
- Order and delivery-related context, satisfaction, and cashback-related variables also contribute meaningfully.

## Caveats

A few points should keep us grounded:

1. **The test scores are unusually strong**
   - ROC-AUC of **0.9802** for random forest and **0.9853** for gradient boosting are very high.
   - That can happen when the signal is genuinely strong, but it can also be inflated by dataset structure, overlap in customer behavior patterns, or leakage.
   - One known source of leakage here is the median imputation performed on the full dataset before splitting.

2. **The selected model is not the best on every metric**
   - Random forest won on cross-validated ROC-AUC, but gradient boosting was better on the test set for ROC-AUC, PR-AUC, precision, recall, and F1.
   - That does not change the selection rule, but it does mean the “best” model depends on whether you mean “best by the pre-declared selection criterion” or “best on the held-out test split.”

3. **Importance is not causality**
   - A high permutation importance score does not mean a feature causes churn.
   - It only means the model relies on that feature when making predictions.

4. **Fairness metrics show different behaviors by group**
   - For **Gender**:
     - Female: customers **404**, actual churn rate **18.6%**, flagged rate **22.8%**, recall **85.3%**
     - Male: customers **611**, actual churn rate **15.2%**, flagged rate **18.2%**, recall **96.8%**
   - For **MaritalStatus**:
     - Divorced: customers **157**, actual churn rate **17.2%**, flagged rate **20.4%**, recall **96.3%**
     - Married: customers **540**, actual churn rate **11.5%**, flagged rate **16.9%**, recall **95.2%**
     - Single: customers **318**, actual churn rate **24.8%**, flagged rate **25.2%**, recall **87.3%**

   These numbers do not prove bias by themselves, but they do show that model behavior differs across groups and should be monitored before deployment.

Overall, the project produced a strong churn classifier, and the selected **random_forest** model is defensible under the stated selection rule. At the same time, the unusually high test results and the preprocessing leakage mean we should treat the metrics as encouraging, not as proof that the model will perform identically in production.

## Appendix — metrics (generated from metrics.json)

| Model | CV ROC-AUC (train) | Test ROC-AUC | Test PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| logistic_regression | 0.8938 ± 0.0149 | 0.8901 | 0.6831 | 0.4778 | 0.8333 | 0.6074 |
| random_forest **(selected)** | 0.9596 ± 0.0070 | 0.9802 | 0.9259 | 0.7586 | 0.9167 | 0.8302 |
| gradient_boosting | 0.9590 ± 0.0124 | 0.9853 | 0.9522 | 0.8093 | 0.9345 | 0.8674 |
