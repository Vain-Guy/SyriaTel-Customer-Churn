import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

# Import all libraries used in your pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost.core import CatBoostClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE  

# Page config
st.set_page_config(
    page_title="SyriaTel Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Styling 
st.markdown("""
<style>
/* Main gradient background: Deep Blue -> Cyan */
.stApp {
    background: linear-gradient(135deg, #0f2027 0%, #2c5364 50%, #203a43 100%);
}
/* Neumorphic-style cards */
.metric-card {
    background: rgba(255, 255, 255, 0.9);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 8px 8px 16px rgba(0,0,0,0.15), -8px -8px 16px rgba(255,255,255,0.2);
    margin: 10px 0;
}
/* Header styling */
h1, h2, h3 {
    color: white !important;
    font-weight: 700 !important;
}
/* Expander styling */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.9) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
/* Button styling */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #36d1dc 0%, #5b86e5 100%);
    color: white;
    font-weight: 600;
    font-size: 18px;
    padding: 15px;
    border-radius: 10px;
    border: none;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: transform 0.2s;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}
/* Metric styling */
[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: 700;
}
/* Info boxes */
.stAlert {
    border-radius: 10px;
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Load preprocessor and model separately
# --------------------------
@st.cache_resource
def load_objects():
    preprocessor = joblib.load("../App/preprocessor.pkl")
    model = joblib.load("../App/catboost_model.pkl")
    return preprocessor, model

preprocessor, model = load_objects()

# Header
st.markdown("""
<h1 style='text-align:center; color:white; font-size:48px; font-weight:700; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
SyriaTel Churn Intelligence Platform
</h1>
<p style='text-align:center; color:white; font-size:18px;'>Customer Retention Analytics</p>
""", unsafe_allow_html=True)

# --------------------------
# Customer Input Section
# --------------------------
with st.expander("👤 Customer Profile & Account Details", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        state = st.selectbox("📍 State", options=[
            "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID",
            "IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO",
            "MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA",
            "RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
        ])
        account_length = st.number_input("📅 Account Length in Days", 1, 300, 120)
    with col2:
        intl_plan = st.toggle("🌍 International Calls Plan", value=False)
        voicemail_plan = st.toggle("📧 Voicemail Plan", value=True)
        customer_service_calls = st.number_input("📞 Total Customer Service Calls Made", 0, 20, 1)
    with col3:
        total_day_mins = st.number_input("☀️ Total Minutes Used During Daytime", 0.0, 500.0, 180.0, step=10.0)
        total_eve_mins = st.number_input("🌆 Total Minutes Used During Evening", 0.0, 500.0, 200.0, step=10.0)
        total_night_mins = st.number_input("🌙 Total Minutes Used During Nighttime", 0.0, 500.0, 200.0, step=10.0)
    with col4:
        total_day_calls = st.number_input("☀️ Total Calls Made During Daytime", 0, 200, 100)
        total_eve_calls = st.number_input("🌆 Total Calls Made During Evening", 0, 200, 80)
        total_night_calls = st.number_input("🌙 Total Calls Made During Nighttime", 0, 200, 90)
with st.expander("🌐 International Usage", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        total_intl_mins = st.number_input("🌍 Total International Minutes Used", 0.0, 50.0, 10.0, step=1.0)
    with col2:
        total_intl_calls = st.number_input("🌍 Total International Calls Made", 0, 20, 3)

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------
# Predict Button
# --------------------------
if st.button("🔮 Analyze Churn Risk", type="primary"):
    custserv_per_month = (customer_service_calls / account_length) * 30 if account_length>0 else 0

    df = pd.DataFrame([{
        "state": state,
        "account_length": account_length,
        "intl_plan": int(intl_plan),
        "voicemail_plan": int(voicemail_plan),
        "total_day_mins": total_day_mins,
        "total_day_calls": total_day_calls,
        "total_eve_mins": total_eve_mins,
        "total_eve_calls": total_eve_calls,
        "total_night_mins": total_night_mins,
        "total_night_calls": total_night_calls,
        "total_intl_mins": total_intl_mins,
        "total_intl_calls": total_intl_calls,
        "customer_service_calls": customer_service_calls,
        "custserv_per_month": custserv_per_month
    }])

    # Transform input with preprocessor
    X_transformed = preprocessor.transform(df)

    # Predict
    proba = model.predict_proba(X_transformed)[0][1]
    pred = int(proba >= 0.5)

    # --------------------------
    # Churn Gauge
    # --------------------------
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=proba*100,
        domain={'x':[0,1],'y':[0,1]},
        title={'text':'Churn Probability', 'font':{'size':24, 'color':'white'}},
        delta={'reference':50, 'increasing':{'color':'red'}, 'decreasing':{'color':'green'}},
        gauge={
            'axis': {'range':[0,100], 'tickcolor':'white'},
            'bar': {'color':'#5b86e5'},
            'bgcolor':'rgba(0,0,0,0)',
            'steps':[
                {'range':[0,33], 'color':'#10b981'},
                {'range':[33,66], 'color':'#fbbf24'},
                {'range':[66,100], 'color':'#ef4444'}
            ],
        }
    ))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=350)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # --------------------------
    # KPIs
    # --------------------------
    st.markdown("### 📊 Key Performance Indicators")
    total_usage = total_day_mins + total_eve_mins + total_night_mins
    kpi_data = [
        {"label":"📞 Total Customer Service Calls", "value":customer_service_calls, "delta":f"{custserv_per_month:.1f}/month", "color":"inverse" if customer_service_calls>3 else "normal"},
        {"label":"⏱️ Total Usage", "value":f"{total_usage:.0f} min", "delta":"All periods", "color":"off"},
        {"label":"📅 Tenure", "value":f"{account_length} days", "delta":f"{account_length/30:.1f} months", "color":"normal" if account_length>60 else "inverse"},
        {"label":"📋 Active Plans", "value":len([p for p in [intl_plan, voicemail_plan] if p]), "delta":"", "color":"off"},
        {"label":"🌍 International Minutes", "value":total_intl_mins, "delta":"", "color":"normal"},
        {"label":"📧 Voicemail Plan", "value":"Active" if voicemail_plan else "Inactive", "delta":"", "color":"off"},
        {"label":"🌙 Night Minutes", "value":total_night_mins,"delta":"","color":"off"},
        {"label":"🌆 Evening Minutes","value":total_eve_mins,"delta":"","color":"off"}
    ]
    for i in range(0,len(kpi_data),4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            if i+j<len(kpi_data):
                k = kpi_data[i+j]
                col.metric(label=k["label"], value=k["value"], delta=k["delta"], delta_color=k["color"])

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------
    # Usage Charts
    # --------------------------
    st.markdown("### 📈 Usage Analytics")
    col1, col2 = st.columns(2)
    usage_df = pd.DataFrame({
        "Period":["Day","Evening","Night","International"],
        "Minutes":[total_day_mins,total_eve_mins,total_night_mins,total_intl_mins],
        "Calls":[total_day_calls,total_eve_calls,total_night_calls,total_intl_calls],
        "Color":["#36d1dc","#5b86e5","#36d1dc","#5b86e5"]
    })
    with col1:
        fig1 = go.Figure(go.Bar(
            x=usage_df["Period"], y=usage_df["Minutes"],
            marker_color=usage_df["Color"], text=usage_df["Minutes"],
            textposition='outside'
        ))
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.1)',
                           title="Minutes by Period", title_font=dict(color='white'), height=500,
                           margin=dict(l=40, r=40, t=80, b=40))
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = go.Figure(go.Bar(
            x=usage_df["Period"], y=usage_df["Calls"],
            marker_color=usage_df["Color"], text=usage_df["Calls"],
            textposition='outside'
        ))
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.1)',
                           title="Calls by Period", title_font=dict(color='white'), height=500,
                           margin=dict(l=40, r=40, t=80, b=40))
        st.plotly_chart(fig2, use_container_width=True)

    # --------------------------
    # Retention Recommendations
    # --------------------------
    st.markdown("### 🎯 Retention Strategy")
    if pred==1:
        st.error("⚠️ **URGENT ACTION REQUIRED** - High churn risk detected")
        actions=[]
        if customer_service_calls>5:
            actions.append("🔴 **Priority Support**: Assign dedicated account manager immediately")
        if custserv_per_month>2:
            actions.append("🔴 **Service Quality Review**: Investigate recurring issues causing frequent calls")
        if intl_plan and total_intl_mins>15:
            actions.append("🟡 **Plan Optimization**: Customer may benefit from premium international package")
        elif not intl_plan and total_intl_mins>0:
            actions.append("🟢 **Upsell Opportunity**: Promote international plan with 20% discount")
        if total_usage<300:
            actions.append("🟡 **Engagement Campaign**: Send personalized offers to increase usage")
        if account_length<90:
            actions.append("🟡 **Early Lifecycle Support**: Strengthen onboarding and engagement")
        if not actions:
            actions.append("🔵 **Proactive Outreach**: Schedule courtesy call to ensure satisfaction")
        for i, act in enumerate(actions,1):
            st.markdown(f"**{i}.** {act}")
    else:
        st.success("✅ **Customer Status: HEALTHY** - Continue standard engagement")
        st.markdown("""
        **Recommended Actions:**
        - 🎁 **Loyalty Reward**: Recognize tenure with special offer
        - 📊 **Usage Monitoring**: Track for any sudden changes
        - 💬 **Feedback Collection**: Request satisfaction survey
        - 🌟 **Referral Program**: Encourage customer advocacy
        """)

    # --------------------------
    # Customer Profile Summary
    # --------------------------
    with st.expander("📋 Customer Profile", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            **Account Information**
            - State: `{state}`
            - Tenure: `{account_length}` days
            - Int'l Plan: `{"Active" if intl_plan else "Inactive"}`
            - Voicemail: `{"Active" if voicemail_plan else "Inactive"}`
            """)
        with col2:
            st.markdown(f"""
            **Usage Patterns**
            - Day: `{total_day_mins:.0f}` min / `{total_day_calls}` calls
            - Evening: `{total_eve_mins:.0f}` min / `{total_eve_calls}` calls
            - Night: `{total_night_mins:.0f}` min / `{total_night_calls}` calls
            """)
        with col3:
            st.markdown(f"""
            **Support History**
            - Total Calls: `{customer_service_calls}`
            - Monthly Rate: `{custserv_per_month:.2f}`
            - Int'l Usage: `{total_intl_mins:.0f}` min / `{total_intl_calls}` calls
            """)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color: rgba(255,255,255,0.7);'>Powered by Ahjin Analytics · SyriaTel Analytics Platform</p>", unsafe_allow_html=True)