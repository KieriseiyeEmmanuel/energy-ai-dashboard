import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cohere
from prophet import Prophet
from prophet.plot import plot_plotly
from fpdf import FPDF
import base64
from io import BytesIO

st.set_page_config(page_title="🛸 Vora AI Analyst", layout="wide")

st.title("🛸 Vora – Chevron-Grade AI Analyst")
st.markdown("Upload Chevron-style data and let Vora automatically generate visual insights, KPIs, forecasting, and reports.")

uploaded_file = st.file_uploader("📥 Upload Excel file", type=["xlsx"])

def generate_pdf_report(title, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(33, 37, 41)
    pdf.set_fill_color(240, 248, 255)
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=summary)

    # Convert to base64 download link
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="vora_report.pdf">📄 Download Vora PDF Report</a>'
    return href

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")

    cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("🔑 Enter Cohere API Key", type="password")

    # AI-Based Data Summary
    if cohere_key:
        try:
            co = cohere.Client(cohere_key)
            st.subheader("🧠 AI Auto Summary & Insights")
            preview = df.head(5).to_string(index=False)
            description = df.describe(include='all').fillna('').to_string()
            context = f"Data Preview:\n{preview}\n\nSummary:\n{description}"
            kpi_response = co.chat(
                message=f"You are a Chevron analyst. Based on the data below, identify roles, KPIs, insights, charts, and give an executive summary:\n\n{context}",
                model="command-r-plus",
                temperature=0.5
            ).text
            st.success(kpi_response)

            # PDF Report
            report_html = generate_pdf_report("Vora Chevron Summary", kpi_response)
            st.markdown(report_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"AI Error: {e}")

    # Dataframe preview
    st.markdown("### 📋 Raw Data")
    st.dataframe(df)

    # 📊 KPIs for numeric columns
    st.markdown("### 📈 Auto KPIs")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        kpi_cols = st.columns(min(3, len(num_cols)))
        for i, col in enumerate(num_cols[:3]):
            kpi_cols[i].metric(col, f"{df[col].mean():,.2f}")
    
    # 📉 Chart Selector
    st.markdown("### 📊 Chart Visualizer")
    chart_type = st.selectbox("Select Chart Type", ["Line Chart", "Bar Chart", "Scatter Plot"])
    x_axis = st.selectbox("Select X-axis", options=df.columns)
    y_axis = st.selectbox("Select Y-axis", options=num_cols)

    if chart_type == "Line Chart":
        st.plotly_chart(px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}"))
    elif chart_type == "Bar Chart":
        st.plotly_chart(px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}"))
    elif chart_type == "Scatter Plot":
        st.plotly_chart(px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}"))

    # 🧪 Forecasting
    st.markdown("### 🧪 Forecasting")
    if "Date" in df.columns or any("date" in c.lower() for c in df.columns):
        date_col = st.selectbox("Select Time Column", options=[col for col in df.columns if "date" in col.lower()])
        target_col = st.selectbox("Select Target Column to Forecast", options=num_cols)

        df_forecast = df[[date_col, target_col]].dropna()
        df_forecast.columns = ['ds', 'y']
        df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])

        m = Prophet()
        m.fit(df_forecast)
        future = m.make_future_dataframe(periods=12, freq='M')
        forecast = m.predict(future)

        st.plotly_chart(plot_plotly(m, forecast))
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))
    else:
        st.warning("No suitable date column found for forecasting.")

    # 🤖 Ask Anything
    st.markdown("### 🤖 Ask Vora Anything")
    question = st.text_area("Ask a question or command (e.g. 'Summarize this', 'Show production trend')")
    if st.button("Ask Vora") and cohere_key and question:
        try:
            sample_csv = df.head(10).to_csv(index=False)
            ask_prompt = f"You are a Chevron analyst. Given this sample data:\n{sample_csv}\n\nAnswer: {question}"
            ask_response = co.chat(message=ask_prompt, model="command-r-plus").text
            st.success(ask_response)
        except Exception as e:
            st.error(f"Vora AI Error: {e}")
else:
    st.info("👈 Upload a Chevron-style Excel file to begin.")

