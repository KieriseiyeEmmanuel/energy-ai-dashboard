import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import io, base64, os
from fpdf import FPDF
from datetime import datetime
import cohere
import json
import tempfile
import speech_recognition as sr
import soundfile as sf

# --------- Basic Setup ---------
st.set_page_config(page_title="VORA X – AI Chevron Analyst", layout="wide", page_icon="🛸")
st.title("🛸 VORA X – Chevron Intelligence Redefined")

# --------- Access Control ---------
access_granted = False
with st.sidebar:
    st.markdown("## 🔐 Analyst Login")
    password = st.text_input("Enter Access Code", type="password")
    if password == "vora2025":
        access_granted = True
        st.success("Access Granted")
    else:
        st.warning("Waiting for valid code...")

if not access_granted:
    st.stop()

# --------- Upload Section ---------
uploaded_file = st.file_uploader("📥 Upload Chevron Excel File", type=["xlsx"])
cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("🔑 Enter Cohere API Key", type="password")

# --------- Session Memory ---------
if "query_log" not in st.session_state:
    st.session_state.query_log = []

# --------- Helper Functions ---------
def ai_insight(prompt_text, df):
    try:
        co = cohere.Client(cohere_key)
        preview = df.head(5).to_string(index=False)
        summary = df.describe(include='all').fillna('').to_string()
        context = f"Data Preview:\n{preview}\n\nSummary:\n{summary}"
        response = co.chat(
            message=f"{prompt_text}\n\n{context}",
            model="command-r-plus",
            temperature=0.4
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

def generate_pdf_report(title, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=summary)
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmpfile.name)
    with open(tmpfile.name, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="vora_report.pdf">📄 Download AI Report</a>'
    return href

# --------- Audio Voice Input ---------
st.sidebar.markdown("### 🗣️ Ask Vora via Voice")
audio_file = st.sidebar.file_uploader("Upload .wav voice file", type=["wav"])
if audio_file:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            voice_query = recognizer.recognize_google(audio_data)
            st.sidebar.success(f"You said: {voice_query}")
            st.session_state.query_log.append(voice_query)
        except:
            st.sidebar.error("Could not recognize speech")

# --------- If File Uploaded ---------
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ File Uploaded")

    # --------- Smart Role Detection ---------
    if cohere_key:
        role_response = ai_insight("Identify the appropriate Chevron analyst role this dataset fits", df)
        st.info(f"🧠 **Detected Role:** {role_response}")

    # --------- Dynamic KPI Cards ---------
    st.subheader("📊 Key Metrics")
    for col in df.select_dtypes(include=np.number).columns[:3]:
        st.metric(label=col, value=f"{df[col].mean():,.2f}")

    # --------- Auto Charts ---------
    st.subheader("📈 Smart Chart Suggestions")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) >= 2:
        chart = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], trendline="ols")
        st.plotly_chart(chart)

    # --------- Forecasting (Prophet + ARIMA) ---------
    st.markdown("### 🔮 Forecasting Engine")
    time_col = st.selectbox("Choose Time Column", options=[col for col in df.columns if "date" in col.lower()])
    value_col = st.selectbox("Choose Value Column", options=numeric_cols)
    model_choice = st.radio("Choose Model", ["Prophet", "ARIMA"])

    df_forecast = df[[time_col, value_col]].dropna()
    df_forecast.columns = ['ds', 'y']
    df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])

    if model_choice == "Prophet":
        model = Prophet()
        model.fit(df_forecast)
        future = model.make_future_dataframe(periods=12, freq="M")
        forecast = model.predict(future)
        fig = px.line(forecast, x='ds', y='yhat', title="Prophet Forecast")
        st.plotly_chart(fig)
    else:
        df_arima = df_forecast.set_index("ds")
        model = ARIMA(df_arima, order=(1, 1, 1))
        result = model.fit()
        forecast = result.forecast(steps=12)
        st.line_chart(forecast)

    # --------- AI Assistant ---------
    st.markdown("### 🤖 Ask Vora (Chat)")
    query = st.text_area("Type or voice your question:")
    if st.button("Ask") and (query or audio_file) and cohere_key:
        full_query = voice_query if audio_file else query
        st.session_state.query_log.append(full_query)
        ai_response = ai_insight(full_query, df)
        st.success(ai_response)

    # --------- AI Report ---------
    st.markdown("### 📄 AI Report Generator")
    if st.button("Generate PDF Report"):
        summary_text = ai_insight("Summarize this Chevron dataset and offer strategic recommendations.", df)
        st.markdown(generate_pdf_report("VORA Intelligence Report", summary_text), unsafe_allow_html=True)

    # --------- Session Memory ---------
    st.markdown("### 🧠 Query Memory")
    st.json(st.session_state.query_log)
else:
    st.info("👈 Upload your Excel file and enter AI key to begin.")
