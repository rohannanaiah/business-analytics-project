import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")
df_clean = pd.read_csv("data/dataset.csv")

numeric_cols = ['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']

page = st.sidebar.selectbox("Navigate", ["Business Problem", "Data Insights", "Prediction"])

if page == "Business Problem":
    st.title("Bank Term Deposit Subscription Predictor")
    st.subheader("Business Problem")
    st.write("A bank runs phone-call marketing campaigns to sell term deposits, but calling every customer is costly, and most calls don't convert (~89% say no).")
    st.subheader("Analytics Objective")
    st.write("Develop a classification model that predicts whether a customer is likely to subscribe to a term deposit.")
    st.subheader("Dataset")
    st.write("UCI Bank Marketing dataset - 41,188 rows, 21 original columns, from a Portuguese bank's direct marketing campaigns.")
    st.subheader("Target Variable")
    st.write("y - Yes/No, whether the customer subscribed to a term deposit.")

elif page == "Data Insights":
    st.title("Data Insights")

    st.subheader("Age Distribution by Subscription Outcome")
    fig1, ax1 = plt.subplots()
    df_clean[df_clean['y']=='yes']['age'].hist(bins=30, alpha=0.5, label='yes', density=True, ax=ax1)
    df_clean[df_clean['y']=='no']['age'].hist(bins=30, alpha=0.5, label='no', density=True, ax=ax1)
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Density')
    ax1.legend()
    st.pyplot(fig1)
    st.write("Young and retirement-age customers subscribe proportionally more than middle-aged customers, likely due to fewer financial obligations.")

    st.subheader("Subscription Rate by Previous Campaign Outcome")
    fig2, ax2 = plt.subplots()
    pd.crosstab(df_clean['poutcome'], df_clean['y'], normalize='index').plot(kind='bar', ax=ax2)
    st.pyplot(fig2)
    st.write("Customers who previously subscribed successfully convert again at ~65%, compared to ~9-14% for others.")

    st.subheader("Correlation Heatmap of Economic Indicators")
    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax3)
    st.pyplot(fig3)
    st.write("Several economic indicators are highly correlated, indicating redundancy.")

elif page == "Prediction":
    st.title("Make a Prediction")

    age = st.slider("Age", 17, 98, 40)
    job = st.selectbox("Job", df_clean['job'].unique())
    marital = st.selectbox("Marital Status", df_clean['marital'].unique())
    education = st.selectbox("Education", df_clean['education'].unique())
    campaign = st.slider("Number of contacts this campaign", 1, 56, 2)
    poutcome = st.selectbox("Previous Campaign Outcome", df_clean['poutcome'].unique())

    if st.button("Predict"):
        input_dict = {col: 0 for col in model_columns}
        input_dict['age'] = age
        input_dict['campaign'] = campaign
        for col, val in [('job', job), ('marital', marital), ('education', education), ('poutcome', poutcome)]:
            dummy_col = f"{col}_{val}"
            if dummy_col in input_dict:
                input_dict[dummy_col] = 1

        input_df = pd.DataFrame([input_dict])[model_columns]
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"Prediction: Likely to Subscribe (probability: {probability:.2%})")
            st.write("Recommendation: Prioritize this customer for a call.")
        else:
            st.warning(f"Prediction: Unlikely to Subscribe (probability: {probability:.2%})")
            st.write("Recommendation: Lower priority for calling.")
