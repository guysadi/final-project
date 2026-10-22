# Customer Churn — Business Insights

## Headline
Harbor & Vale’s clean dataset has **5,074 customers** and a churn rate of **16.6%** (**841 churners**). The clearest churn signals are **short tenure**, **complaints**, and **higher device complexity**. The biggest risk pocket is customers with **0–1 months tenure**, where churn reaches **51.3%**.

## Who churns
- Customers with **0–1 months tenure** have the highest churn at **51.3%**; churn is much lower at **2–6 months (7.37%)**, **7–12 months (9.54%)**, and **13–24 months (6.81%)**. Average tenure is **3.91 months** for churners versus **11.4 months** for customers who stay, with correlation **-0.332**.
- Customers who **complained** churn at **31.31%**, versus **10.77%** for customers with no complaint. Complaint status has correlation **0.249** with churn.
- Customers with **more registered devices** churn more often: **8.76%** at **1 device**, **8.63%** at **2**, **14.52%** at **3**, **16.21%** at **4**, **22.65%** at **5**, and **35.62%** at **6 devices**. Correlation is **0.117**.
- **Mobile Phone** buyers churn at **26.29%**, higher than **Laptop & Accessory (10.50%)**, **Fashion (16.25%)**, **Grocery (4.37%)**, and **Others (7.85%)**.
- **Single** customers churn at **26.72%**, compared with **14.62%** for **Divorced** and **11.30%** for **Married** customers.
- **Tier 3** city customers churn at **21.93%**, versus **17.44%** in **Tier 2** and **13.97%** in **Tier 1**.

## What drives churn
- **Satisfaction is associated with higher churn in this data**, which is counterintuitive: churn rises from **11.51%** at score **1** to **23.05%** at score **5**. Correlation is **0.101**.
- Behavioral measures are lower among churners, but the effects are modest: churners average **2.93 orders** versus **3.06** for stayers, **161.78 USD cashback** versus **180.79 USD**, and **3.31 days since last order** versus **4.73 days** for stayers. Since **lower days since last order means a more recent order**, churners ordered more recently here. Correlations are **-0.017** for order count, **-0.143** for cashback, and **-0.148** for days since last order.

## Recommended actions
- Prioritize retention outreach in the **first month** of the customer journey, especially customers at **0–1 months tenure** and those with **complaints**.
- Flag customers with **3+ registered devices** for proactive support, since churn rises from **8.63%** at **2 devices** to **35.62%** at **6 devices**.
- Build targeted campaigns for **Mobile Phone** buyers, **Single** customers, and **Tier 3** city customers, where churn is **26.29%**, **26.72%**, and **21.93%**, respectively.
- Review the **satisfaction signal** carefully before using it as a success metric, because higher scores are associated with **higher** churn here.

## Data notes
- The raw file had **5,630 rows**; cleaning reduced it to **5,074 rows** by removing **556 duplicate rows** and standardizing category labels.
- Missing values were filled with medians for **7 columns**: **tenure (9)**, **WarehouseToHome (13)**, **HourSpendOnApp (3)**, **OrderAmountHikeFromlastYear (15)**, **CouponUsed (1)**, **OrderCount (2)**, and **DaySinceLastOrder (3)**.
- Count fields were cast to integers, so each row now represents **one customer** with consistent values.
