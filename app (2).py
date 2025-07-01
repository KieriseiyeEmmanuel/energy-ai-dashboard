import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
from fpdf import FPDF
import cohere
import io
import base64

# 🌌 Sci-Fi Theme
st.set_page_config(page_title="Vora – AI Chevron Analyst", layout="wide", page_icon="🛸")

# 🌌 Custom Sci-Fi CSS
st.markdown("""
<style>
body, .main { background-color: #0d1117; color: #c9d1d9; font-family: 'Orbitron', sans-serif; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #111b26 100%); color: #c9d1d9; }
.stTextInput > div > div > input, .stTextArea > div > textarea {
    background-color: #161b22; color: #c9d1d9; border: 1px solid #30363d; border-radius: 10px;
}
.stButton > button { background-color: #58a6ff; color: black; border-radius: 1rem; font-weight: bold; }
.stSidebar { background-color: #0d1117; }
</style>
""", unsafe_allow_html=True)

st.title("🛸 Vora – Chevron-Grade AI Energy Intelligence")
st.markdown("Empowering analysts with AI-driven decisions and Sci-Fi visual interface.")

uploaded_file = st.file_uploader("📥 Upload Excel File (Chevron-Style)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")

    # === API Key Setup ===
    cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("🔐 Enter Cohere API Key", type="password")

    # === AI Summary ===
    if cohere_key:
        try:
            co = cohere.Client(cohere_key)
            preview = df.head(5).to_string(index=False)
            stats = df.describe(include='all').fillna('').to_string()
            structure = f"Data Preview:\n{preview}\n\nSummary Stats:\n{stats}"

            st.markdown("### 🧠 AI Insight & Role Prediction")
            response = co.chat(
                message=f"""You are a Chevron enterprise analyst assistant named Vora.
Given the following data preview and summary, detect the role (finance, market intelligence, operations, etc), suggest the best KPIs, charts, and insights to display:
\n\n{structure}""",
                model="command-r-plus",
                temperature=0.5
            )
            st.success(response.text)
        except Exception as e:
            st.error(f"Cohere AI Error: {e}")

    # === Ask Vora Anything ===
    st.markdown("### 🤖 Ask Vora Anything")
    question = st.text_area("What would you like Vora to do with your data? (e.g., 'Plot trend', 'Forecast sales')")
    if st.button("🔍 Analyze") and cohere_key and question:
        try:
            csv_sample = df.head(10).to_csv(index=False)
            response = co.chat(
                message=f"You are Vora, a data analyst. Here's sample data:\n{csv_sample}\n\nTask: {question}",
                model="command-r-plus"
            )
            st.success(response.text)
        except Exception as e:
            st.error(f"Cohere AI Error: {e}")

    # === Basic KPI Cards if numeric columns available ===
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) >= 3:
        st.subheader("📊 KPI Dashboard")
        cols = st.columns(3)
        for i, col in enumerate(numeric_cols[:3]):
            cols[i].metric(f"{col}", f"{df[col].mean():,.2f}")

    # === Chart Section ===
    st.subheader("📈 Quick Visualizer")
    with st.expander("📊 Generate Custom Chart"):
        x_axis = st.selectbox("X-Axis", options=df.columns)
        y_axis = st.selectbox("Y-Axis", options=numeric_cols)
        chart_type = st.radio("Chart Type", ["Line", "Bar", "Scatter"])
        if chart_type == "Line":
            st.plotly_chart(px.line(df, x=x_axis, y=y_axis), use_container_width=True)
        elif chart_type == "Bar":
            st.plotly_chart(px.bar(df, x=x_axis, y=y_axis), use_container_width=True)
        else:
            st.plotly_chart(px.scatter(df, x=x_axis, y=y_axis), use_container_width=True)

    # === Forecasting with Prophet ===
    if "Date" in df.columns or any("date" in c.lower() for c in df.columns):
        st.subheader("📈 Forecasting with Prophet")
        time_col = st.selectbox("Select Time Column", [c for c in df.columns if "date" in c.lower()])
        value_col = st.selectbox("Select Value to Forecast", [c for c in numeric_cols])
        df_forecast = df[[time_col, value_col]].dropna()
        df_forecast.columns = ['ds', 'y']
        df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])

        m = Prophet()
        m.fit(df_forecast)
        future = m.make_future_dataframe(periods=12, freq='M')
        forecast = m.predict(future)
        st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))

    # === PDF Export Feature ===
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
        href = f'<a href="data:application/pdf;base64,{b64}" download="vora_report.pdf">📄 Download PDF Report</a>'
        return href

    st.markdown("### 📄 Export AI Summary as PDF")
    if st.button("Generate Report PDF") and cohere_key:
        try:
            summary_prompt = f"Summarize key insights for a Chevron exec from this dataset:\n\n{df.head(10).to_string(index=False)}"
            pdf_summary = co.chat(message=summary_prompt, model="command-r-plus").text
            st.markdown(generate_pdf_report("Vora Executive Summary", pdf_summary), unsafe_allow_html=True)
        except Exception as e:
            st.error("Failed to generate PDF summary.")

else:
    st.info("👈 Upload your Chevron-style Excel file to begin.")
