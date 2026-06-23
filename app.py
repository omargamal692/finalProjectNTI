import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="Fraud Risk Command Center",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fraud Risk Analytics & Assessment Tool")
st.markdown("Monitor historical fraud distributions and test real-time risk parameters on your target variable `is_fraud`.")

@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "fraud.csv")
    
    df = pd.read_csv(file_path)
    
    df['Fraud Status'] = df['is_fraud'].map({1: '⚠️ Fraudulent', 0: '✅ Legitimate'})
    return df

try:
    df = load_data()
except Exception as e:
    st.error("❌ Could not load 'fraud.csv'.")
    st.write("🔧 **Troubleshooting steps:**")
    st.write("1. Double check that your file is named exactly `fraud.csv` (no extra `.txt` or `.csv` at the end).")
    st.write("2. Make sure `fraud.csv` is sitting in the **exact same folder** as this `app.py` script.")
    st.stop()

tab1, tab2 = st.tabs(["📊 Historical Insights Dashboard", "🔬 Interactive Fraud Risk Simulator"])


with tab1:
    st.subheader("Target Distribution Analysis")
    
    total_records = len(df)
    fraud_cases = int(df['is_fraud'].sum())
    fraud_rate = (fraud_cases / total_records) * 100
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Total Transactions Evaluated", f"{total_records:,}")
    m_col2.metric("Total Fraud Cases Detected", f"{fraud_cases:,}")
    m_col3.metric("Overall Fraud Rate (Target Class)", f"{fraud_rate:.2f}%")
    
    st.markdown("---")
    
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.markdown("#### Transaction Value Distribution by Target Class")
        fig_hist = px.histogram(
            df, 
            x="transaction_amount", 
            color="Fraud Status",
            marginal="box", 
            barmode="overlay",
            color_discrete_map={'⚠️ Fraudulent': '#EF553B', '✅ Legitimate': '#636EFA'},
            labels={"transaction_amount": "Transaction Amount ($)"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with v_col2:
        st.markdown("#### Velocity Score vs. Customer Age Correlation")
        sample_df = df.sample(min(5000, len(df)), random_state=42)
        fig_scatter = px.scatter(
            sample_df,
            x="customer_age",
            y="velocity_score",
            color="Fraud Status",
            opacity=0.6,
            color_discrete_map={'⚠️ Fraudulent': '#EF553B', '✅ Legitimate': '#636EFA'},
            labels={"customer_age": "Customer Age", "velocity_score": "Velocity Score"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


with tab2:
    st.subheader("Simulate a Transaction Scenario")
    st.markdown("Adjust the dynamic operational parameters below to inspect how transaction characteristics flag potential fraud profile boundaries.")
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    with sim_col1:
        input_amount = st.slider(
            "Transaction Amount ($)", 
            min_value=0.0, 
            max_value=float(df['transaction_amount'].max()), 
            value=float(df['transaction_amount'].median())
        )
        input_age = st.slider(
            "Customer Age", 
            min_value=int(df['customer_age'].dropna().min()), 
            max_value=int(df['customer_age'].dropna().max()), 
            value=int(df['customer_age'].dropna().median())
        )
        
    with sim_col2:
        input_velocity = st.slider(
            "Velocity Score (Transaction Frequency)", 
            min_value=float(df['velocity_score'].min()), 
            max_value=float(df['velocity_score'].max()), 
            value=float(df['velocity_score'].median())
        )
        input_items = st.number_input(
            "Number of Items Purchased", 
            min_value=1, 
            max_value=20, 
            value=2
        )
        
    with sim_col3:
        st.markdown("#### Scenario Assessment Profile")
        
        high_risk_amount = df[df['is_fraud'] == 1]['transaction_amount'].median()
        high_risk_velocity = df[df['is_fraud'] == 1]['velocity_score'].median()
        
        is_high_risk = (input_amount > high_risk_amount) and (input_velocity > high_risk_velocity)
        
        if is_high_risk:
            st.error("🚨 HIGH FRAUD RISK PROFILE DETECTED")
            st.write(f"This operational pattern mimics common historical traits found in your target `is_fraud` anomalies (High Velocity + High Amount).")
        else:
            st.success("✅ LOW FRAUD RISK PROFILE")
            st.write("Transaction parameters sit safely within normal distribution baselines.")

    st.markdown("---")
    st.info("💡 **Data Science Note:** Once you train your classification model (like Random Forest or XGBoost) in your Jupyter Notebook (`FinalProject.ipynb`), you can export it using `joblib.dump()` and load it directly into this tab to handle the real predictions!")