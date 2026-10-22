# Customer Churn — Business Insights

## Headline
Harbor & Vale’s churn rate is **16.57%** (**841 of 5,074** customers). The clearest risk signals are **complaints**, **very short tenure**, and a few high-risk customer segments; these are associations, not proven causes.

## Who churns
- Customers who **complained** churned at **31.31%** vs **10.77%** for those who did not; Complain has the strongest positive churn correlation at **0.249**.
- Customers with **0–1 months tenure** churned at **51.30%**, versus **7.37%** at **2–6 months**, **9.54%** at **7–12 months**, **6.81%** at **13–24 months**, and **0.0%** at **25+ months**. Tenure has the strongest overall correlation with churn at **-0.332**.
- **Single** customers churned at **26.72%**, compared with **14.62%** for **Divorced** and **11.30%** for **Married** customers.
- Customers whose preferred order category was **Mobile Phone** churned at **26.29%**, higher than **Fashion (16.25%)**, **Laptop & Accessory (10.50%)**, **Others (7.85%)**, and **Grocery (4.37%)**.
- Customers using **Cash on Delivery** churned at **25.38%** and **E wallet** at **22.78%**, above **UPI (15.56%)**, **Debit Card (15.01%)**, and **Credit Card (14.16%)**.

## What drives churn
- **More devices registered** is associated with higher churn: **8.63%** at **2 devices**, **14.52%** at **3**, **16.21%** at **4**, **22.65%** at **5**, and **35.62%** at **6** devices. The correlation is **0.117**.
- **SatisfactionScore** is associated with higher churn in this data: churn rises to **23.05%** at score **5** versus **11.51%** at score **1**. Correlation is **0.101**.
- Churners averaged **3.91** months of **Tenure** vs **11.4** for stayers, **16.65** for **WarehouseToHome** vs **15.19**, **4.5** for **NumberOfAddress** vs **4.13**, **15.52** for **OrderAmountHikeFromlastYear** vs **15.71**, **1.79** for **CouponUsed** vs **1.77**, **2.93** for **OrderCount** vs **3.06**, **3.31** for **DaySinceLastOrder** vs **4.73**, and **161.78** for **CashbackAmount** vs **180.79**. Because **DaySinceLastOrder** is lower among churners, churners ordered **more recently**, not less recently.

## Recommended actions
- Prioritize retention outreach for customers with **0–1 months tenure**, where churn is **51.30%**.
- Create a fast-resolution path for customers who **complain**, given their **31.31%** churn rate and **0.249** churn correlation.
- Target offers and onboarding support to **Single** customers (**26.72%** churn), **Mobile Phone** buyers (**26.29%**), and **Cash on Delivery** users (**25.38%**).
- Review the experience of customers with **5–6 registered devices** because churn reaches **22.65%–35.62%** there.
- Do not assume higher satisfaction means lower risk; the data shows the opposite association, with churn rising to **23.05%** at satisfaction score **5** versus **11.51%** at score **1**.

## Data notes
The raw dataset had **5,630** rows; the cleaned dataset has **5,074** rows. We removed **556 duplicate rows** and standardized categories such as **Phone → Mobile Phone** (**1,102** values), **CC → Credit Card / COD → Cash on Delivery** (**570** values), and **Mobile → Mobile Phone** (**726** values). Missing values were filled with medians for **7** columns, including **Tenure (231 filled, median 9)** and **DaySinceLastOrder (288 filled, median 3)**.
