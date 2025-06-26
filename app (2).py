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
from statsmodels.tsa.arima.model import ARIMA

# 🌌 Set Sci-Fi theme & layout
st.set_page_config(
    page_title="Vora – AI Chevron Analyst",
    layout="wide",
    page_icon="🛸"
)

# 🌌 Custom CSS for a SciFi Look
st.markdown("""
    <style>
    body, .main {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Orbitron', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #111b26 100%);
        color: #c9d1d9;
    }
    .stTextInput > div > div > input, .stTextArea > div > textarea {
        background-color: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #58a6ff;
        color: black;
        border-radius: 1rem;
        font-weight: bold;
    }
    .stSidebar {
        background-color: #0d1117;
    }
    .css-1d391kg, .css-1kyxreq, .css-1cpxqw2, .css-1offfwp, .css-1v0mbdj {
        color: #58a6ff !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛸 Vora – Chevron-Grade AI Energy Intelligence")
st.markdown("Empowering analysts with AI-driven decision tools in a futuristic interface.")

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
        href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">📄 Download PDF Report</a>'
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
        st.header("📈 Forecasting with Prophet & ARIMA")
        if "Date" in df.columns:
            time_col = st.selectbox("Select Time Column", options=[col for col in df.columns if "date" in col.lower() or "Date" in col])
            value_col = st.selectbox("Select Value Column", options=[col for col in df.columns if df[col].dtype in ['float64', 'int64']])

            df_forecast = df[[time_col, value_col]].dropna()
            df_forecast.columns = ['ds', 'y']
            df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])

            st.subheader("🔮 Prophet Forecast")
            m = Prophet()
            m.fit(df_forecast)
            future = m.make_future_dataframe(periods=12, freq='M')
            forecast = m.predict(future)

            st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))

            st.subheader("🧠 AI Insight on Prophet")
            ai_insight("You are forecasting future values using Prophet. Explain the trends and what the forecast shows.")

            st.subheader("📉 ARIMA Forecast")
            df_arima = df_forecast.set_index('ds')
            try:
                arima_model = ARIMA(df_arima['y'], order=(1, 1, 1))
                arima_result = arima_model.fit()
                forecast_arima = arima_result.forecast(steps=12)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_arima.index, y=df_arima['y'], name='Observed'))
                future_dates = pd.date_range(start=df_arima.index[-1], periods=12, freq='M')
                fig.add_trace(go.Scatter(x=future_dates, y=forecast_arima, name='ARIMA Forecast'))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"ARIMA Error: {e}")

            st.subheader("🧠 AI Insight on ARIMA")
            ai_insight("You are forecasting using ARIMA. Explain the trend and performance.")
        else:
            st.warning("❗ Your file must include a 'Date' column for forecasting.")

    # Other role logic here (not shown for brevity)...

else:
    st.info("👈 Select a role and upload your Chevron-style Excel file to begin.")
