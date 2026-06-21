# app.py - Employee Turnover Prediction - Streamlit Dashboard
# Sprint 4 / Part E | Deployment Prototype
#
# Run locally:
#   pip install streamlit scikit-learn pandas joblib matplotlib
#   streamlit run app.py
#
# Deploy free: https://streamlit.io/cloud

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, json, io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="📊", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Individual Prediction", "About"])

DEPT_COLS = [
    "dept_IT","dept_RandD","dept_accounting","dept_hr","dept_management",
    "dept_marketing","dept_product_mng","dept_sales","dept_support","dept_technical"
]
FEATURES = [
    "satisfaction_level","last_evaluation","number_project",
    "average_montly_hours","time_spend_company","Work_accident",
    "promotion_last_5years","salary_encoded","engagement_score","overwork_flag"
] + DEPT_COLS

def engineer_features(df):
    df = df.copy()
    df["engagement_score"] = (df["satisfaction_level"]*0.5 + df["last_evaluation"]*0.5)*100
    df["overwork_flag"]    = (df["average_montly_hours"] > 240).astype(int)
    df["salary_encoded"]   = df["salary"].map({"low":0,"medium":1,"high":2}).fillna(0)
    dept_dummies = pd.get_dummies(df["sales"], prefix="dept")
    df = pd.concat([df, dept_dummies], axis=1)
    return df

@st.cache_resource
def train_model(df_hash):
    df_eng = engineer_features(st.session_state["df"])
    for col in DEPT_COLS:
        if col not in df_eng.columns:
            df_eng[col] = 0
    X = df_eng[FEATURES]
    y = df_eng["left"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, model.score(X_test, y_test)

def assign_risk(prob):
    if prob >= 0.60: return "🔴 High"
    if prob >= 0.35: return "🟡 Medium"
    return "🟢 Low"

st.title("📊 Employee Turnover Prediction System")
st.caption("Agile Data Science Project - Sprint 4 Prototype")

uploaded_file = st.sidebar.file_uploader("Upload employee attrition.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["emp_id"] = ["EMP-" + str(1001+i) for i in range(len(df))]
    st.session_state["df"] = df
    model, test_acc = train_model(id(df))

    if page == "Dashboard":
        st.header("HR Attrition Risk Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Employees", f"{len(df):,}")
        col2.metric("Attrition Rate", f"{df['left'].mean()*100:.1f}%")
        col3.metric("Model Accuracy", f"{test_acc*100:.1f}%")

        df_eng = engineer_features(df)
        for col in DEPT_COLS:
            if col not in df_eng.columns:
                df_eng[col] = 0
        df["attrition_prob"] = model.predict_proba(df_eng[FEATURES])[:,1]
        df["risk_level"]     = df["attrition_prob"].apply(assign_risk)
        col4.metric("High Risk", f"{(df['risk_level']=='🔴 High').sum():,}")

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Risk Distribution")
            risk_counts = df["risk_level"].value_counts()
            fig, ax = plt.subplots(figsize=(5,4))
            color_map = {"🔴 High":"#DC2626","🟡 Medium":"#D97706","🟢 Low":"#16A34A"}
            ax.bar(risk_counts.index, risk_counts.values,
                   color=[color_map.get(r,"grey") for r in risk_counts.index])
            ax.set_title("Employees by Risk Level")
            st.pyplot(fig)

        with col_b:
            st.subheader("Attrition by Department")
            dept_att = df.groupby("sales")["left"].mean()*100
            fig2, ax2 = plt.subplots(figsize=(5,4))
            dept_att.sort_values().plot(kind="barh", ax=ax2, color="#1E2761")
            ax2.set_xlabel("Attrition Rate (%)")
            st.pyplot(fig2)

        st.subheader("High-Risk Employees")
        high_risk = df[df["risk_level"]=="🔴 High"][[
            "emp_id","satisfaction_level","average_montly_hours",
            "time_spend_company","salary","attrition_prob","risk_level"
        ]].sort_values("attrition_prob", ascending=False)
        high_risk["attrition_prob"] = high_risk["attrition_prob"].round(3)
        st.dataframe(high_risk.head(50), use_container_width=True)
        st.download_button("Download Risk Table", df.to_csv(index=False),
                           "risk_table.csv","text/csv")

    elif page == "Individual Prediction":
        st.header("Predict Individual Employee Attrition Risk")
        with st.form("pred_form"):
            c1, c2 = st.columns(2)
            satisfaction = c1.slider("Satisfaction Level", 0.0, 1.0, 0.5, 0.01)
            evaluation   = c2.slider("Last Evaluation Score", 0.0, 1.0, 0.7, 0.01)
            hours        = c1.number_input("Avg Monthly Hours", 80, 350, 200)
            projects     = c2.number_input("Number of Projects", 1, 10, 4)
            tenure       = c1.number_input("Years at Company", 1, 20, 3)
            salary       = c2.selectbox("Salary Level", ["low","medium","high"])
            department   = c1.selectbox("Department", [
                "sales","technical","support","IT","management",
                "accounting","hr","marketing","product_mng","RandD"])
            accident     = c2.checkbox("Work Accident?")
            promotion    = c2.checkbox("Promoted in Last 5 Years?")
            submitted    = st.form_submit_button("Predict")

        if submitted:
            row = pd.DataFrame([{
                "satisfaction_level":satisfaction,"last_evaluation":evaluation,
                "number_project":projects,"average_montly_hours":hours,
                "time_spend_company":tenure,"Work_accident":int(accident),
                "promotion_last_5years":int(promotion),"salary":salary,
                "sales":department,"left":0
            }])
            row_eng = engineer_features(row)
            for col in DEPT_COLS:
                if col not in row_eng.columns:
                    row_eng[col] = 0
            prob = model.predict_proba(row_eng[FEATURES])[0][1]
            risk = assign_risk(prob)
            r1,r2,r3 = st.columns(3)
            r1.metric("Attrition Probability", f"{prob*100:.1f}%")
            r2.metric("Risk Level", risk)
            r3.metric("Engagement Score", f"{(satisfaction*0.5+evaluation*0.5)*100:.0f}/100")
            if "High" in risk:
                st.error("High risk - immediate HR intervention recommended.")
            elif "Medium" in risk:
                st.warning("Moderate risk - schedule check-in with manager.")
            else:
                st.success("Low risk - employee appears engaged and stable.")

    else:
        st.header("About This Project")
        st.markdown('''
**Course:** CLO2 - Agile Data Science | UTM Graduate School
**Student:** Sharmila Sandrasagran

| Sprint | Focus | Deliverable |
|--------|-------|-------------|
| Sprint 1 | Data and EDA | Rule-based Risk Table |
| Sprint 2 | Feature Engineering | Enriched Dataset |
| Sprint 3 | ML Model | Random Forest Predictions |
| Sprint 4 | Deployment | This Streamlit App |

**Algorithm:** Random Forest | **Features:** 20 | **Expected Accuracy:** ~98%
        ''')
else:
    st.info("Please upload your employee attrition.csv file using the sidebar to get started.")
