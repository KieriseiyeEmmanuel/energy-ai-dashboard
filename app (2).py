# VORA 2.0 – Chevron-Grade AI Analyst Platform

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import numpy_financial as npf
from prophet import Prophet
import cohere
from fpdf import FPDF
import io
import base64
from datetime import datetime
import json
import os

# Setup
st.set_page_config(page_title="VORA – Chevron AI Analyst", layout="wide", page_icon="🛸")

# --- CSS Sci-Fi Styling ---
st.markdown("""
    <style>
    body { background-color: #0f111a; color: #c9d1d9; font-family: 'Orbitron', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0d1117, #111b26); }
    .stButton>button { background: #58a6ff; border-radius: 10px; font-weight: bold; }
    .stTextInput>div>div>input, .stTextArea>div>textarea { background: #161b22; color: white; border-radius: 10px; }
    .stSidebar { background: #0d1117; }
    </style>
""", unsafe_allow_html=True)

st.title("🛸 VORA – Chevron AI Intelligence Platform")

# Globals
cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("Enter Cohere API Key", type="password")
session_log = st.session_state.get("log", [])
uploaded_file = st.file_uploader("📥 Upload Chevron-Style Excel File", type=["xlsx"])

# --- Helper: PDF Report Generator ---
def generate_pdf_report(title, content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=content)
    file_path = "vora_report.pdf"
    pdf.output(file_path)
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        return f'<a href="data:application/pdf;base64,{b64}" download="VORA_Report.pdf">📄 Download Report</a>'

# --- Helper: AI Call to Cohere ---
def cohere_insight(message):
    if not cohere_key:
        return "🔐 Please provide Cohere API key."
    try:
        co = cohere.Client(cohere_key)
        response = co.chat(message=message, model="command-r-plus", temperature=0.4)
        return response.text
    except Exception as e:
        return f"❌ AI Error: {e}"

# --- Main Logic ---
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ File uploaded successfully.")
    preview = df.head(5).to_string(index=False)
    summary = df.describe(include="all").to_string()
    structure = f"Preview:\n{preview}\n\nStats:\n{summary}"

    # Smart Role Detection
    st.subheader("🔍 Smart Role Detection")
    role_prompt = f"You are a Chevron data analyst. Based on this dataset structure, what type of business role is this? Choose from: Project Finance, Operations, Market Intelligence, Forecasting, Policy, Logistics\n\n{structure}"
    detected_role = cohere_insight(role_prompt)
    st.markdown(f"**🔧 Detected Role:** `{detected_role.strip()}`")

    # AI KPIs and Visual Suggestions
    st.subheader("📊 AI Insights & KPIs")
    kpi_prompt = f"You are an energy data scientist. Suggest 3 KPIs and charts to show for this dataset.\n\n{structure}"
    kpi_response = cohere_insight(kpi_prompt)
    st.info(kpi_response)

    # Embedded VORA Chat
    st.subheader("🤖 Ask Vora Anything")
    query = st.text_area("Ask a question about your data")
    if st.button("Ask"):
        sample = df.head(10).to_csv(index=False)
        response = cohere_insight(f"Dataset:\n{sample}\n\nQuestion: {query}")
        st.success(response)

    # Dynamic Visual Cards
    st.subheader("📌 Data Snapshot")
    for col in df.select_dtypes(include=['float64', 'int64']).columns[:3]:
        st.metric(label=f"{col} (avg)", value=f"{df[col].mean():,.2f}")

    # Chart
    st.subheader("📈 Smart Chart")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if numeric_cols:
        x_col = st.selectbox("X-axis", df.columns)
        y_col = st.selectbox("Y-axis", numeric_cols)
        st.plotly_chart(px.line(df, x=x_col, y=y_col), use_container_width=True)

    # Forecasting
    st.markdown("### 🧪 Forecasting")
    if "Date" in df.columns:
        forecast_df = df[["Date", y_col]].rename(columns={"Date": "ds", y_col: "y"}).dropna()
        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
        m = Prophet()
        m.fit(forecast_df)
        future = m.make_future_dataframe(periods=6, freq="M")
        forecast = m.predict(future)
        st.plotly_chart(px.line(forecast, x="ds", y="yhat"), use_container_width=True)
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

    # Report Download
    st.subheader("📄 AI Report")
    report = generate_pdf_report("Vora Chevron Summary", kpi_response)
    st.markdown(report, unsafe_allow_html=True)

else:
    st.warning("👈 Upload a Chevron-style Excel file to begin")

# Log user activity
session_log.append({"timestamp": str(datetime.now()), "file_uploaded": bool(uploaded_file), "query": query if 'query' in locals() else ""})
st.session_state["log"] = session_log
