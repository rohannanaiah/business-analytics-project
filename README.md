# Bank Term Deposit Subscription Predictor

## Business Problem
A bank runs phone-call marketing campaigns to sell term deposits, but calling every customer is costly, and most calls don't convert (~89% say no).

## Objective
Develop a classification model that predicts whether a customer is likely to subscribe to a term deposit.

## Dataset
- Source: UCI Machine Learning Repository (Bank Marketing dataset)
- Rows: 41,188
- Columns: 21 original (20 after cleaning)
- Target variable: y (Yes/No — term deposit subscription)

## Methodology
Data loaded → cleaned (dropped `duration` due to data leakage, dropped `default` due to high missingness/imbalance, engineered `was_previously_contacted` from `pdays`) → EDA (age, poutcome, correlation analysis) → feature encoding (one-hot) → train/test split (80/20, stratified) → model training (Logistic Regression, Random Forest) → evaluation → model saved (joblib) → Streamlit app → deployment.

## Model Used
Random Forest Classifier (`n_estimators=100`, `max_depth=15`, `class_weight='balanced'`)

## Model Performance
The model achieves an overall accuracy of ~88%, with a precision of ~49% and recall of ~57%. Accuracy alone is misleading here since ~89% of customers already say no — even a model that predicted "no" for everyone would score ~89% accuracy. Recall is the priority metric for this business problem, since missing a genuine subscriber costs the bank more than an extra wasted call. A recall of ~57% means the model catches about 57 out of every 100 customers who would actually subscribe — a substantial improvement over an earlier, larger version of this model (~29% recall), achieved by limiting tree depth to reduce overfitting.

**Key factors:** age, euribor3m (economic interest-rate indicator), and campaign (number of contacts made this campaign). Several top factors reflect broader economic conditions rather than just individual customer traits.

## Business Insights
- **Age**: Young and retirement-age customers subscribe proportionally more than middle-aged customers, likely due to fewer financial obligations (e.g. no mortgage, no dependents).
- **Previous campaign outcome**: Customers who previously subscribed successfully convert again at ~65%, compared to ~9–14% for others — the bank should prioritize recontacting past successful customers.
- **Multicollinearity**: Several economic indicators (`emp.var.rate`, `euribor3m`, `nr.employed`) are highly correlated with each other, indicating redundancy.

## Business Decision
If the model predicts a customer is likely to subscribe, the bank should prioritize them for a call, focusing limited marketing/calling resources on customers most likely to convert. Customers predicted as unlikely to subscribe can be deprioritized, reducing wasted effort on low-probability contacts.

## How to Run the Application
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## Limitations
The Streamlit app only collects 6 input fields for simplicity (age, job, marital status, education, campaign contacts, previous outcome), while the underlying model was trained on ~50 features. All other fields — including `euribor3m`, one of the model's most important predictors — default to 0 in the app, which doesn't reflect real-world values. This means the app's live predictions are likely less reliable than the model's actual test performance. Additionally, the model was trained on historical data from a specific bank and time period, so it may not generalize well to different economic conditions or customer bases. The dataset's class imbalance (~89% no) still limits the model's precision — even with improved recall (~57%), roughly half of customers flagged as "likely yes" will not actually subscribe.

## Cost of Errors
If the model produces a false positive (predicting "yes" when the customer would actually say no), the bank simply wastes a phone call — a relatively low-cost mistake. If the model produces a false negative (predicting "no" when the customer would have actually subscribed), the bank misses a genuine business opportunity and loses potential revenue — this is the more costly type of error, which is why the project prioritized recall over precision when evaluating model performance.

## Deployed Application Link
[Add once deployed on Streamlit Community Cloud]

## GitHub Repository
[Add your repo URL here]
