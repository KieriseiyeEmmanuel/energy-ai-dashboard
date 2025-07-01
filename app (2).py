import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cohere
import io
import base64
from prophet import Prophet
from fpdf import FPDF

st.set_page_config(page_title="🛸 Vora – Chevron AI Analyst", layout="wide")

st.title("🛸 Vora – Chevron AI Analyst")
st.markdown("AI-Powered Chevron-Grade Analysis Dashboard")

uploaded_file = st.file_uploader("📥 Upload Chevron-style Excel file", type=["xlsx"])

# Utility Functions
def generate_pdf_report(title, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(33, 37, 41)
    pdf.set_fill_color(240, 248, 255)
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=summary)

    # ✅ Return as download link using base64
    pdf_output = pdf.output(dest='S').encode('latin1')
    b64 = base64.b64encode(pdf_output).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="vora_report.pdf">📄 Download PDF Report</a>'
    return href

def ai_insight(prompt_text, df, api_key):
    try:
        co = cohere.Client(api_key)
        preview = df.head(10).to_string(index=False)
        context = f"{prompt_text}\n\nData Preview:\n{preview}"
        response = co.chat(message=context, model="command-r-plus", temperature=0.5)
        return response.text
    except Exception as e:
        return f"AI Error: {e}"

def plot_auto_charts(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        st.subheader("📊 Auto-Generated Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.bar(df, x=numeric_cols[0], y=numeric_cols[1], title="Bar Chart"))
        with col2:
            st.plotly_chart(px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title="Scatter Plot"))
        st.plotly_chart(px.line(df[numeric_cols], title="Line Trends Over Time"))

def detect_anomalies(df):
    st.subheader("🚨 Anomaly Detection")
    for col in df.select_dtypes(include=np.number).columns:
        outliers = df[(df[col] - df[col].mean()).abs() > 2 * df[col].std()]
        if not outliers.empty:
            st.warning(f"Anomalies detected in '{col}':")
            st.dataframe(outliers)

def forecast_time_series(df):
    time_cols = [col for col in df.columns if 'date' in col.lower()]
    val_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if time_cols and val_cols:
        st.subheader("📈 Forecasting with Prophet")
        time_col = time_cols[0]
        val_col = st.selectbox("Select Value to Forecast", val_cols)
        data = df[[time_col, val_col]].dropna()
        data.columns = ['ds', 'y']
        data['ds'] = pd.to_datetime(data['ds'])
        m = Prophet()
        m.fit(data)
        future = m.make_future_dataframe(periods=6, freq='M')
        forecast = m.predict(future)
        fig = px.line(forecast, x='ds', y='yhat', title="Forecast")
        st.plotly_chart(fig)
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6))

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name=None)
    sheet_names = list(df.keys())

    st.sidebar.success("✅ File uploaded.")
    selected_sheet = st.sidebar.selectbox("Select Sheet", sheet_names)
    df_selected = df[selected_sheet]
    st.subheader(f"📄 Sheet: {selected_sheet}")
    st.dataframe(df_selected.head())

    cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("Enter Cohere API Key", type="password")

    if cohere_key:
        role_response = ai_insight("What Chevron department or role is most relevant for this dataset?", df_selected, cohere_key)
        st.markdown("### 🧠 AI Role Detection")
        st.success(role_response)

        kpi_response = ai_insight("Extract KPIs and business insights from this data. Format nicely.", df_selected, cohere_key)
        st.markdown("### 📈 AI KPI Insights")
        st.info(kpi_response)

        viz_response = ai_insight("Suggest 2-3 visualizations for this data.", df_selected, cohere_key)
        st.markdown("### 📊 AI Visualization Recommendations")
        st.info(viz_response)

        st.markdown("### 📉 Auto Charts")
        plot_auto_charts(df_selected)

        st.markdown("### 🔍 Anomaly Insights")
        detect_anomalies(df_selected)

        st.markdown("### 🧪 Forecasting")
        forecast_time_series(df_selected)

        st.markdown("### 📄 Smart Report Export")
        report = generate_pdf_report("Vora Chevron Summary", kpi_response)
        st.markdown(report, unsafe_allow_html=True)

        st.markdown("### 🤖 Ask Vora")
        query = st.text_area("Ask anything about your dataset")
        if st.button("Submit"):
            custom_response = ai_insight(query, df_selected, cohere_key)
            st.success(custom_response)
else:
    st.info("👈 Upload a Chevron-style Excel file to begin.")
