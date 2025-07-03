import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from fpdf import FPDF
import io, base64, datetime
import cohere

# 🔧 Sci-Fi Theme
st.set_page_config(page_title="VORA X – Chevron AI Analyst", layout="wide")
st.markdown("""
<style>
body {
    background: #0f0f23;
    color: #e0e0e0;
    font-family: 'Orbitron', sans-serif;
}
.stApp {
    background: linear-gradient(145deg, #0f0f23, #1a1a40);
    color: #ffffff;
}
h1, h2, h3 {
    color: #58a6ff;
}
.stButton>button {
    background-color: #58a6ff !important;
    color: black;
    border-radius: 0.5rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 VORA X – AI Chevron Analyst Dashboard")

# Upload file
uploaded_file = st.file_uploader("📥 Upload Chevron-style Excel file", type=["xlsx"])
cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("Enter Cohere API Key", type="password")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ File uploaded!")

    # --------------------- SMART ROLE DETECTION ---------------------
    def detect_role(df):
        cols = df.columns.str.lower().tolist()
        if 'project' in cols and 'cash flow (usd)' in cols:
            return "Finance"
        elif 'well' in cols and 'daily output' in cols:
            return "Production"
        elif 'date' in cols and 'brent price' in cols:
            return "Market"
        elif 'scenario' in cols and 'co2 emissions' in cols:
            return "Policy"
        elif 'route' in cols and 'delivery time (days)' in cols:
            return "SupplyChain"
        elif 'date' in cols and any(df.dtypes == 'datetime64[ns]'):
            return "Forecast"
        return "Generic"

    role = detect_role(df)
    st.markdown(f"🧠 **Detected Role:** `{role}`")

    # --------------------- DYNAMIC KPIs ---------------------
    st.subheader("📊 KPIs")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    kpi_cols = st.columns(min(3, len(numeric_cols)))
    for i, col in enumerate(numeric_cols[:3]):
        kpi_cols[i].metric(col, f"{df[col].mean():,.2f}")

    # --------------------- SMART CHARTING ---------------------
    st.subheader("📈 Auto-Generated Charts")
    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color=numeric_cols[1])
        st.plotly_chart(fig, use_container_width=True)
    if 'date' in df.columns.str.lower().tolist():
        date_col = [c for c in df.columns if "date" in c.lower()][0]
        val_col = numeric_cols[0]
        fig = px.line(df, x=date_col, y=val_col, title=f"Trend of {val_col}")
        st.plotly_chart(fig, use_container_width=True)

    # --------------------- FORECASTING ---------------------
    if 'date' in df.columns.str.lower().tolist():
        st.subheader("🔮 Forecasting")
        time_col = st.selectbox("Time Column", df.columns[df.dtypes == 'datetime64[ns]'])
        value_col = st.selectbox("Value to Forecast", numeric_cols)
        forecast_df = df[[time_col, value_col]].dropna()
        forecast_df.columns = ['ds', 'y']
        model = Prophet()
        model.fit(forecast_df)
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)
        st.plotly_chart(px.line(forecast, x='ds', y='yhat', title="Forecasted Values"), use_container_width=True)

    # --------------------- ASK VORA ---------------------
    st.subheader("🧠 Ask VORA (AI Agent)")
    ask = st.text_area("Ask Vora anything about this data...")
    if st.button("🧠 Run AI Query") and cohere_key and ask:
        co = cohere.Client(cohere_key)
        df_sample = df.head(10).to_csv(index=False)
        prompt = f"""You are VORA, Chevron’s elite AI analyst. Analyze this dataset:
        Sample Data:\n{df_sample}\n\nQuestion: {ask}"""
        response = co.chat(message=prompt, model="command-r-plus")
        st.success(response.text)

    # --------------------- PDF REPORT GENERATOR ---------------------
    def generate_pdf_report(title, summary):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(200, 10, txt=title, ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt=summary)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        b64 = base64.b64encode(pdf_bytes).decode()
        return f'<a href="data:application/pdf;base64,{b64}" download="vora_report.pdf">📄 Download Report</a>'

    if st.button("📄 Generate AI Report"):
        summary_text = f"Data Columns: {', '.join(df.columns)}\nDetected Role: {role}\nKPIs:\n"
        for col in numeric_cols[:3]:
            summary_text += f"{col}: Mean = {df[col].mean():,.2f}\n"
        st.markdown(generate_pdf_report("VORA Summary Report", summary_text), unsafe_allow_html=True)

else:
    st.info("👆 Upload a Chevron-style Excel file to begin.")
