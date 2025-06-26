import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import cohere
import io
import base64
from prophet import Prophet
from prophet.plot import plot_plotly
from fpdf import FPDF
from datetime import datetime

st.set_page_config(
    page_title="Vora – Chevron AI Analyst",
    layout="wide",
    page_icon="🧠"
)

st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .block-container {padding-top: 2rem;}
    .stButton>button {
        background-color: #004080;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
    }
    .stTextInput>div>div>input {
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Vora – Chevron-Grade AI Energy Intelligence")
st.markdown("Empowering analysts with AI-driven decision tools and dynamic data insights.")

roles = [
    "📊 Project Finance & Economics",
    "⚙️ Production & Operations Analyst",
    "🌍 Market Intelligence",
    "📘 Energy Policy Scenarios",
    "🚚 Supply Chain & Logistics",
    "📈 Forecasting (Prophet/ARIMA)",
    "🤖 Ask Vora (AI Assistant)"
]
selected_role = st.sidebar.selectbox("Select Analyst Role", roles)

uploaded_file = st.file_uploader("📥 Upload Chevron-Style Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")

    def generate_pdf_report(title, summary):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(33, 37, 41)
        pdf.set_fill_color(240, 248, 255)
        pdf.cell(200, 10, txt=title, ln=True, align='C')
        pdf.multi_cell(0, 10, txt=summary)
        buffer = io.BytesIO()
        pdf.output(buffer)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">📄 Download Finance Report</a>'
        return href

    def ai_insight(prompt_text):
        cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("Enter Cohere API Key", type="password")
        if cohere_key:
            try:
                co = cohere.Client(cohere_key)
                df_sample = df.head().to_string(index=False)
                summary = df.describe(include='all').fillna('').to_string()
                context = f"Data Preview:\n{df_sample}\n\nSummary:\n{summary}"
                response = co.chat(
                    message=f"{prompt_text}\n\n{context}",
                    model="command-r-plus",
                    temperature=0.4
                )
                st.success(response.text)
            except Exception as e:
                st.error(f"Cohere Error: {e}")

    if selected_role == "📈 Forecasting (Prophet/ARIMA)":
        st.header("📈 Forecasting with Prophet")
        if "Date" in df.columns:
            time_col = st.selectbox("Select Time Column", options=[col for col in df.columns if "date" in col.lower() or "Date" in col])
            value_col = st.selectbox("Select Value Column", options=[col for col in df.columns if df[col].dtype in ['float64', 'int64']])

            df_forecast = df[[time_col, value_col]].dropna()
            df_forecast.columns = ['ds', 'y']
            df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])

            m = Prophet()
            m.fit(df_forecast)
            future = m.make_future_dataframe(periods=12, freq='M')
            forecast = m.predict(future)

            st.plotly_chart(plot_plotly(m, forecast))
            st.subheader("Forecasted Data")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))
            st.write("\n🔍 **AI Insights on Forecasting**")
            ai_insight("You are forecasting future values using Prophet. Explain the trends and what the forecast shows.")
        else:
            st.warning("❗ Your file must include a 'Date' column for forecasting.")

    # Other role logic remains unchanged...

else:
    st.info("👈 Select a role and upload your Chevron-style Excel file to begin.")
