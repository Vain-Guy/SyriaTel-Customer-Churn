import streamlit as st
import joblib
import pandas as pd

# --------------------------
# LOAD MODEL
# --------------------------
model = joblib.load("churn_pipeline.pkl")   # Change if needed

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Prediction App")
st.write("Fill in customer details below to get a churn prediction.")

# --------------------------
# SIDEBAR INFO
# --------------------------
st.sidebar.title("ℹ️ How it works")
st.sidebar.write("""
This app uses your trained machine learning model to predict whether a customer is likely to churn.
Simply enter the customer’s information in the form and click **Predict**.
""")

# --------------------------
# INPUT FORM
# --------------------------
with st.form("churn_form"):

    st.subheader("Customer Info")

    state = st.selectbox(
        "State", 
        options=[
            "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY",
            "LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND",
            "OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
        ]
    )

    account_length = st.number_input("Account Length (days)", min_value=1, max_value=250, value=120)

    intl_plan = st.selectbox("Has International Plan?", [0, 1])
    voicemail_plan = st.selectbox("Has Voicemail Plan?", [0, 1])

    total_day_mins = st.number_input("Total Day Minutes", 0.0, 400.0, 200.0)
    total_day_calls = st.number_input("Total Day Calls", 0, 200, 100)

    total_eve_mins = st.number_input("Total Evening Minutes", 0.0, 400.0, 200.0)
    total_eve_calls = st.number_input("Total Evening Calls", 0, 200, 100)

    total_night_mins = st.number_input("Total Night Minutes", 0.0, 400.0, 200.0)
    total_night_calls = st.number_input("Total Night Calls", 0, 200, 100)

    total_intl_mins = st.number_input("Total International Minutes", 0.0, 50.0, 10.0)
    total_intl_calls = st.number_input("Total International Calls", 0, 20, 2)

    customer_service_calls = st.number_input("Customer Service Calls", 0, 20, 1)

    submitted = st.form_submit_button("Predict")

# --------------------------
# PREDICTION
# --------------------------
if submitted:
    data = {
        "state": state,
        "account_length": account_length,
        "international_plan": intl_plan,
        "voice_mail_plan": voicemail_plan,
        "total_day_minutes": total_day_mins,
        "total_day_calls": total_day_calls,
        "total_eve_minutes": total_eve_mins,
        "total_eve_calls": total_eve_calls,
        "total_night_minutes": total_night_mins,
        "total_night_calls": total_night_calls,
        "total_intl_minutes": total_intl_mins,
        "total_intl_calls": total_intl_calls,
        "customer_service_calls": customer_service_calls
    }

    df = pd.DataFrame([data])
    proba = model.predict_proba(df)[0][1]
    label = int(proba >= 0.5)

    st.subheader("🔮 Prediction Results")
    st.write(f"**Churn Probability:** `{proba:.3f}`")
    st.write(f"**Churn Prediction:** {'Yes (1)' if label == 1 else 'No (0)' }")

    if label == 1:
        st.error("⚠️ This customer is likely to churn!")
    else:
        st.success("✅ This customer is NOT likely to churn.")