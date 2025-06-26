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

    # 📊 Project Finance & Economics
    if selected_role == "📊 Project Finance & Economics":
        st.header("📊 Project Finance & Economics")
        if 'Project' in df.columns and 'Cash Flow (USD)' in df.columns:
            project = st.selectbox("Select Project", df["Project"].unique())
            rate = st.slider("Discount Rate (%)", 0.01, 0.3, 0.1)
            project_df = df[df["Project"] == project]
            cash_flows = project_df["Cash Flow (USD)"].tolist()
            npv = sum(cf / (1 + rate)**i for i, cf in enumerate(cash_flows))
            irr = npf.irr(cash_flows)
            cumulative = np.cumsum(cash_flows)
            payback = next((i for i, v in enumerate(cumulative) if v >= 0), None)

            col1, col2, col3 = st.columns(3)
            col1.metric("NPV", f"${npv:,.2f}")
            col2.metric("IRR", f"{irr:.2%}" if irr else "N/A")
            col3.metric("Payback", f"{payback} years" if payback else "Beyond range")

            st.subheader("📊 Cash Flow Timeline")
            project_df["Year"] = list(range(1, len(project_df) + 1))
            st.plotly_chart(px.bar(project_df, x="Year", y="Cash Flow (USD)"), use_container_width=True)
            st.markdown("\n🔍 **AI Insight for Financial Analyst Role**")
            ai_insight("You are a financial analyst evaluating project cash flows. Provide interpretation of NPV, IRR, and payback period trends.")
        else:
            st.warning("🚫 Required columns not found: 'Project', 'Cash Flow (USD)'")

    # 🤖 Ask Vora (AI Assistant)
    elif selected_role == "🤖 Ask Vora (AI Assistant)":
        st.header("🤖 Ask Vora – AI Assistant")
        ai_insight("You're an energy analyst assistant. Help answer any questions about the uploaded data.")

    # 📈 Forecasting (Prophet/ARIMA)
    elif selected_role == "📈 Forecasting (Prophet/ARIMA)":
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

            st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)
            st.subheader("Forecasted Data")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))
            st.markdown("\n🔍 **AI Insights on Forecasting**")
            ai_insight("You are forecasting future values using Prophet. Explain the trends and what the forecast shows.")
        else:
            st.warning("❗ Your file must include a 'Date' column for forecasting.")

else:
    st.info("👈 Select a role and upload your Chevron-style Excel file to begin.")
