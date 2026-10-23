# Model Evaluation Report

## Summary

We trained and compared three churn models on Harbor & Vale’s data using the same feature set, the same train/test split, and the same decision threshold of 0.5. The model was selected by the highest 5-fold cross-validated ROC-AUC on the training set, not by test performance.

The selected model is the random forest, with a cross-validated ROC-AUC of 0.9596 and a test ROC-AUC of 0.9802. On the test set, it achieved PR-AUC of 0.9259, precision of 0.7586, recall of 0.9167, and F1 of 0.8302.

The gradient boosting model scored slightly higher on the test set than the random forest on every reported test metric except precision, but it was not selected because model choice must be based on cross-validation on the training data, not on the test set. That is the right rule because the test set is supposed to remain an unbiased final check. If we choose a model because it looks best on test, we leak test information into model selection and overstate how well the process will generalize.

## Setup

The evaluation used 26 features.

Numeric features included:
- Tenure
- CityTier
- WarehouseToHome
- HourSpendOnApp
- NumberOfDeviceRegistered
- SatisfactionScore
- NumberOfAddress
- Complain
- OrderAmountHikeFromlastYear
- CouponUsed
- OrderCount
- DaySinceLastOrder
- CashbackAmount

Text features were one-hot encoded:
- PreferredLoginDevice
- PreferredPaymentMode
- PreferedOrderCat

Engineered features:
- is_new_customer
- cashback_per_order
- coupons_per_order
- days_since_order_per_tenure

Sensitive columns were excluded from modeling:
- Gender
- MaritalStatus

Train/test sizes were 4059 and 1015 rows, respectively. The churn rate was 0.1658 in training and 0.1655 in test, so the split looks well balanced on outcome prevalence.

One important caveat in the pipeline: Crew 1 filled missing values with medians computed on the full dataset before the train/test split. That introduces a mild leak of test information into training. It is not as severe as using the target directly, but it can make results look a bit better than a fully clean pipeline would.

## Model comparison

### Logistic regression
- Cross-validated ROC-AUC: 0.8938
- Cross-validated ROC-AUC std: 0.0149
- Test ROC-AUC: 0.8901
- Test PR-AUC: 0.6831
- Test precision: 0.4778
- Test recall: 0.8333
- Test F1: 0.6074

This model is clearly the weakest of the three. It still captures many churners, but precision is low, which means many of the customers it flags would not churn.

### Random forest
- Cross-validated ROC-AUC: 0.9596
- Cross-validated ROC-AUC std: 0.007
- Test ROC-AUC: 0.9802
- Test PR-AUC: 0.9259
- Test precision: 0.7586
- Test recall: 0.9167
- Test F1: 0.8302

This is the selected model because it had the highest 5-fold cross-validated ROC-AUC on the training set. Its cross-validation score is slightly above gradient boosting’s. On test, it performs very strongly, but we should not use that test result to justify selection.

### Gradient boosting
- Cross-validated ROC-AUC: 0.959
- Cross-validated ROC-AUC std: 0.0124
- Test ROC-AUC: 0.9853
- Test PR-AUC: 0.9522
- Test precision: 0.8093
- Test recall: 0.9345
- Test F1: 0.8674

This model is the strongest on the held-out test set by the reported metrics. However, it was not selected because selection must happen using cross-validation on the training set only. The fact that it edges out the random forest on test could be real, or it could be noise. Since the difference in cross-validated ROC-AUC is tiny, we should treat these two as very close competitors rather than claiming a clear winner from the test set.

### What the classification metrics mean here

Precision answers: among customers we flag for retention outreach, how many actually churned?

- High precision means fewer wasted offers, fewer unnecessary discounts, and lower campaign cost per saved churner.
- Low precision means more offers go to customers who would have stayed anyway.

Recall answers: among customers who actually churned, how many did we catch?

- High recall means fewer missed churners.
- Low recall means more churners slip through untreated.

For a retention campaign, the tradeoff is practical:
- Precision is about avoiding waste.
- Recall is about avoiding missed churners.

The selected random forest gives a good balance: precision of 0.7586 and recall of 0.9167. That means it catches most churners while keeping false positives reasonably controlled. If retention offers are expensive, precision matters more. If losing a churner is very costly, recall matters more. The best operating point may differ from the default threshold of 0.5, but threshold tuning was not part of model selection here.

## Selected model

The selected model is the random forest.

Why it was selected:
- It had the highest 5-fold cross-validated ROC-AUC on the training set: 0.9596.
- Its cross-validation standard deviation was 0.007, which is fairly tight and suggests stable performance across folds.
- We did not use the test set to choose between models, even though gradient boosting scored slightly higher on test.

Test performance of the selected model:
- ROC-AUC: 0.9802
- PR-AUC: 0.9259
- Precision: 0.7586
- Recall: 0.9167
- F1: 0.8302

That is a strong result, but it should still be interpreted cautiously because of the median imputation leak noted above.

## What drives the predictions

Permutation importance tells us which features matter most to the model’s performance, not whether they increase or decrease churn risk. Importance is about magnitude of influence, not direction. So we should not say, for example, that high Complain values “increase churn” based on this list alone.

Top features by permutation importance:
- Complain: 0.0448
- Tenure: 0.0261
- NumberOfAddress: 0.0251
- is_new_customer: 0.0178
- SatisfactionScore: 0.0152
- CashbackAmount: 0.0134
- cashback_per_order: 0.0108
- CityTier: 0.0104
- days_since_order_per_tenure: 0.0102
- WarehouseToHome: 0.0095

A few notes:
- Complain is the most important feature by a clear margin, so complaint behavior is one of the strongest signals the model uses.
- Tenure and is_new_customer both rank highly, which indicates customer age and especially very early tenure are important for prediction.
- NumberOfAddress and WarehouseToHome suggest customer profile and logistics-related signals matter.
- SatisfactionScore also appears important, which is consistent with churn being tied to customer experience.
- CashbackAmount and cashback_per_order show that incentive-related behavior matters to the model.
- days_since_order_per_tenure indicates recency relative to customer age contributes useful information.

Again, these importances do not tell us the direction of effect, only the amount of predictive contribution.

## Caveats

Several reasons to be cautious:

1. The test scores are very high.
   - Random forest and gradient boosting both produce extremely strong ROC-AUC and PR-AUC values.
   - That may reflect genuinely strong signal in the features, but it can also be amplified by the mild preprocessing leak from median imputation on the full dataset.

2. The gap between the top two models is small in cross-validation.
   - Random forest: 0.9596
   - Gradient boosting: 0.959
   - The difference is tiny, so the choice is defensible but not dramatic.

3. Importance is not causality.
   - The top features show what the model uses, not what causes churn.
   - We should not infer business direction from permutation importance alone.

4. Fairness results should be read carefully.
   - The model does not use Gender or MaritalStatus as features, but performance and flagging rates still differ across those groups.
   - For Gender, actual churn rate is 0.1856 for Female and 0.1522 for Male; flagged rate is 0.2277 for Female and 0.1817 for Male; recall is 0.8533 for Female and 0.9677 for Male.
   - For MaritalStatus, actual churn rate is 0.172 for Divorced, 0.1148 for Married, and 0.2484 for Single; flagged rate is 0.2038, 0.1685, and 0.2516, respectively; recall is 0.963 for Divorced, 0.9516 for Married, and 0.8734 for Single.

5. Threshold choice matters.
   - All reported precision, recall, and F1 values use a fixed decision threshold of 0.5.
   - A different threshold could improve recall or precision depending on the business goal, but that would need a separate tuning step.

Overall, the random forest is the best model by the agreed selection rule, and it performs strongly on the holdout set. Still, the unusually strong scores and the preprocessing leak mean we should treat the reported performance as encouraging but not final.

## Appendix — metrics (generated from metrics.json)

| Model | CV ROC-AUC (train) | Test ROC-AUC | Test PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| logistic_regression | 0.8938 ± 0.0149 | 0.8901 | 0.6831 | 0.4778 | 0.8333 | 0.6074 |
| random_forest **(selected)** | 0.9596 ± 0.0070 | 0.9802 | 0.9259 | 0.7586 | 0.9167 | 0.8302 |
| gradient_boosting | 0.9590 ± 0.0124 | 0.9853 | 0.9522 | 0.8093 | 0.9345 | 0.8674 |
