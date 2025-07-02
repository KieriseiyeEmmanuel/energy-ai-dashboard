import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px
from prophet import Prophet
from fpdf import FPDF
import cohere
import io
import base64

# 🎨 Page setup
st.set_page_config(page_title="🛸 Vora – Chevron AI Analyst", layout="wide")

# 🌌 Sci-fi theme styling
st.markdown("""
    <style>
    .main, .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Orbitron', sans-serif;
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
    </style>
""", unsafe_allow_html=True)

st.title("🛸 Vora – Chevron-Grade AI Energy Dashboard")
st.markdown("### 👁️ Chevron’s Vision of AI-Augmented Business Intelligence")

# 📤 Upload
uploaded_file = st.file_uploader("📥 Upload Chevron Excel File", type=["xlsx"])

# 🧠 API Key
cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("🔑 Enter Cohere API Key", type="password")

# 🔍 PDF Report Export
def generate_pdf_report(title, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(33, 37, 41)
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=summary)
    buffer = io.BytesIO()
    pdf.output(buffer, 'F')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="vora_report.pdf">📄 Download PDF Report</a>'

# 🧠 Smart Assistant + Auto Detection
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")

    st.subheader("🧬 AI Role Detection & Insights")
    preview = df.head(5).to_string(index=False)
    describe = df.describe(include='all').fillna('').to_string()

    try:
        co = cohere.Client(cohere_key)
        detect_response = co.chat(
            message=f"Given this Chevron dataset preview:\n\n{preview}\n\n{describe}\n\nWhat type of business role best fits this data (e.g. finance, operations, supply chain, etc)? What KPIs and visualizations would be useful?",
            model="command-r-plus",
            temperature=0.5
        )
        st.success(detect_response.text)

        st.markdown("### 📌 Detected Role Suggestions and KPIs")
        kpi_response = co.chat(
            message=f"Summarize 3 key KPIs from this data:\n\n{preview}\n\n{describe}",
            model="command-r-plus"
        )
        st.info(kpi_response.text)
        st.markdown(generate_pdf_report("Vora Chevron Summary", kpi_response.text), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"💥 AI Error: {e}")

    # 📊 Visual Generator
    st.markdown("### 📈 Smart Chart Generator")
    try:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) >= 2:
            x_col = st.selectbox("📌 Select X-Axis", num_cols)
            y_col = st.selectbox("📌 Select Y-Axis", [c for c in num_cols if c != x_col])
            st.plotly_chart(px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}"), use_container_width=True)
        else:
            st.warning("⚠️ Not enough numeric columns for visualization.")
    except Exception as e:
        st.error(f"Plotting error: {e}")

    # 🧪 Forecasting
    st.markdown("### 🧪 Forecasting")
    try:
        time_col = st.selectbox("📆 Select Date Column", [c for c in df.columns if 'date' in c.lower()])
        metric_col = st.selectbox("📊 Metric to Forecast", [c for c in df.columns if df[c].dtype in ['float64', 'int64']])
        df_fc = df[[time_col, metric_col]].dropna()
        df_fc.columns = ['ds', 'y']
        df_fc['ds'] = pd.to_datetime(df_fc['ds'])
        model = Prophet()
        model.fit(df_fc)
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)
        fig = px.line(forecast, x='ds', y='yhat', title=f"{metric_col} Forecast")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Forecasting failed: {e}")

    # 🧠 Ask Vora (Custom Prompt)
    st.markdown("### 🤖 Ask Vora")
    user_q = st.text_area("Type your question (e.g. What is the average output per region?)")
    if st.button("🧠 Ask Vora"):
        try:
            context = df.head(10).to_csv(index=False)
            vora_response = co.chat(
                message=f"Chevron dataset preview:\n{context}\n\nUser asks: {user_q}",
                model="command-r-plus"
            )
            st.success(vora_response.text)
        except Exception as e:
            st.error(f"AI Assistant Error: {e}")

else:
    st.info("👈 Upload your Chevron-style Excel file to get started.")
