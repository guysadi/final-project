# Customer Churn — Business Insights

## Headline
Churn is **16.6%** overall, with **841 of 5,074 customers** leaving. The clearest risk signal is short tenure: churn is **51.3%** in the **0–1 month** bucket, then drops to **7.4%** at **2–6 months**, **9.5%** at **7–12 months**, **6.8%** at **13–24 months**, and **0.0%** at **25+ months**.

## Who churns
Customers who churn are more likely to be early-life accounts, with average tenure of **3.9 months** versus **11.4 months** for customers who stayed. Churn is also higher for customers who complained: **31.3%** churn versus **10.8%** with no complaints. Higher-risk customer segments include **Single** customers at **26.7%**, **Divorced** at **14.6%**, and **Married** at **11.3%**; and **City Tier 3** at **21.9%**, **Tier 2** at **17.4%**, and **Tier 1** at **14.0%**.

## What drives churn
Several variables are associated with churn in the same direction the EDA found:
- **Tenure** is the strongest separator, with correlation **-0.332** and much lower tenure among churners (**3.9 months** vs **11.4 months**).
- **Complaints** are positively associated with churn, with correlation **0.249** and churn rates of **31.3%** vs **10.8%**.
- **Registered devices** rise with churn, from **8.8%** at **1 device** and **8.6%** at **2** to **35.6%** at **6 devices**.
- **Preferred order category** is uneven: **Mobile Phone** customers churn at **26.3%**, versus **16.3%** for **Fashion**, **10.5%** for **Laptop & Accessory**, **7.9%** for **Others**, and **4.4%** for **Grocery**.
- **Payment mode** also separates churn: **Cash on Delivery** is highest at **25.4%**, followed by **E wallet** at **22.8%**, **UPI** at **15.6%**, **Debit Card** at **15.0%**, and **Credit Card** at **14.2%**.
- **Satisfaction score** is associated with higher churn, with correlation **0.101** and churn rising to **23.1%** at score **5**.
- **Cashback** is lower among churners: **161.78 USD** versus **180.79 USD**, with correlation **-0.143**.

## Recommended actions
1. Prioritize **new customers in months 0–1** and build an early-retention playbook, since churn is **51.3%** in that bucket.
2. Create a fast-response workflow for **complaining customers**, because their churn rate is **31.3%** versus **10.8%**.
3. Target **Cash on Delivery** and **E wallet** users with retention offers, since their churn rates are **25.4%** and **22.8%**.
4. Flag customers with **3+ registered devices**, where churn rises from **14.5%** at **3 devices** to **35.6%** at **6**.
5. Review the **Mobile Phone** and **Fashion** segments first, because their churn rates are **26.3%** and **16.3%**.

## Data notes
The raw file had **5,630 rows** and **20 columns**. Cleaning reduced it to **5,074 rows** by removing **556 duplicate rows** and fixing category labels, including **1,102** device values, **570** payment values, and **726** order-category values. Seven numeric fields had missing values; all were filled with medians, including **Tenure = 9 months**, **WarehouseToHome = 13**, **HourSpendOnApp = 3 hours**, **OrderAmountHikeFromlastYear = 15**, **CouponUsed = 1**, **OrderCount = 2**, and **DaySinceLastOrder = 3 days**.
