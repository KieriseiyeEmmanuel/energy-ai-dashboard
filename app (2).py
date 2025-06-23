import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import seaborn as sns
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

st.set_page_config(page_title="AI Energy Analyst Dashboard", layout="wide")
st.title("🔬 AI Energy & Business Intelligence Dashboard")

roles = [
    "Financial Analyst (NPV/IRR/Payback)",
    "Data Analyst (Trends & Correlations)",
    "Market Intelligence (Oil Prices)",
    "Energy Economist (Policy Scenarios)",
    "Supply Chain Analyst (Logistics KPIs)",
    "AI Business Advisor (Cohere)"
]
selected_role = st.sidebar.radio("Select Role Module", roles)

uploaded_file = st.file_uploader("📥 Upload Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("File uploaded!")

    def generate_pdf_report(title, summary):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=title, ln=True, align='C')
        pdf.multi_cell(0, 10, txt=summary)
        buffer = io.BytesIO()
        pdf.output(buffer)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">📄 Download PDF Report</a>'
        return href

    if selected_role == "Financial Analyst (NPV/IRR/Payback)":
        st.header("📈 Financial Evaluation")
        expected_cols = {"Project", "Cash Flow (USD)"}
        if not expected_cols.issubset(df.columns):
            st.error("❌ File must contain: 'Project' and 'Cash Flow (USD)'")
        else:
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

            st.subheader("Cash Flow Chart")
            project_df["Year"] = list(range(1, len(project_df) + 1))
            st.plotly_chart(px.bar(project_df, x="Year", y="Cash Flow (USD)"), use_container_width=True)

            report_text = f"Project: {project}\n\nNPV: ${npv:,.2f}\nIRR: {irr:.2% if irr else 'N/A'}\nPayback: {payback if payback else 'Beyond range'} years"
            st.markdown(generate_pdf_report("Financial Analyst Report", report_text), unsafe_allow_html=True)

    elif selected_role == "Data Analyst (Trends & Correlations)":
        st.header("📊 Data Trends & Correlation Analysis")
        st.dataframe(df.head())

        st.write("### Correlation Heatmap")
        corr = df.select_dtypes(include=np.number).corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='Viridis'))
        st.plotly_chart(fig, use_container_width=True)

        st.write("### Trends Over Time")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        time_col = st.selectbox("Time Column", df.columns)
        value_col = st.selectbox("Value Column", numeric_cols)
        st.line_chart(df.set_index(time_col)[value_col])

        st.write("### Forecast with Prophet")
        if time_col and value_col:
            prophet_df = df[[time_col, value_col]].rename(columns={time_col: 'ds', value_col: 'y'})
            try:
                m = Prophet()
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=12, freq='M')
                forecast = m.predict(future)
                fig = plot_plotly(m, forecast)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Forecasting Error: {e}")

    elif selected_role == "Market Intelligence (Oil Prices)":
        st.header("🌍 Oil Market Intelligence")
        st.dataframe(df.head())

        st.write("### Oil Price Trends")
        fig = px.line(df, x="Date", y=["Brent Price (USD)", "WTI Price (USD)", "OPEC Basket (USD)"], title="Oil Price Indices")
        st.plotly_chart(fig, use_container_width=True)

        st.write("### Global Demand vs Supply")
        fig2 = px.line(df, x="Date", y=["Global Demand (mb/d)", "Global Supply (mb/d)"], title="Global Oil Demand & Supply")
        st.plotly_chart(fig2, use_container_width=True)

    elif selected_role == "Energy Economist (Policy Scenarios)":
        st.header("📉 Policy Scenario Analysis")
        st.dataframe(df)

        st.write("### Emissions vs GDP Growth")
        fig = px.scatter(df, x="CO₂ Emissions (Mt)", y="Expected GDP Growth (%)", color="Scenario", size="Carbon Tax ($/ton)", hover_data=["Policy"])
        st.plotly_chart(fig, use_container_width=True)

        st.write("### Scenario Comparison")
        fig2 = px.bar(df, x="Scenario", y="CO₂ Emissions (Mt)", color="Policy", title="Emissions under Different Policies")
        st.plotly_chart(fig2, use_container_width=True)

    elif selected_role == "Supply Chain Analyst (Logistics KPIs)":
        st.header("🏭 Supply Chain & Logistics Analysis")
        st.dataframe(df.head())

        kpi_cols = ["Inventory Level (bbl)", "Transport Cost (USD)", "Lead Time (days)", "Uptime (%)", "Delivery Accuracy (%)"]
        for col in kpi_cols:
            st.line_chart(df.pivot(index="Month", columns="Site", values=col), use_container_width=True)

        st.write("### KPI Distribution")
        selected_kpi = st.selectbox("Select KPI to analyze", kpi_cols)
        fig = px.box(df, x="Site", y=selected_kpi, title=f"{selected_kpi} Distribution by Site")
        st.plotly_chart(fig, use_container_width=True)

    elif selected_role == "Vora":
        st.header("🤖 AI Advisor")
        cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("Enter Cohere API Key", type="password")
        question = st.text_area("Ask something about your uploaded data:")

        if st.button("Ask AI") and cohere_key and question:
            try:
                co = cohere.Client(cohere_key)
                df_summary = df.describe(include='all').fillna('').to_string()
                head = df.head(5).to_string(index=False)
                context = f"Here is a preview of the uploaded dataset:\n\n{head}\n\nSummary Statistics:\n{df_summary}"
                full_prompt = f"""
You are a helpful business analyst AI. Use the uploaded dataset below to answer the user's question intelligently.

DATA:
{context}

QUESTION:
{question}
"""
                response = co.chat(
                    message=full_prompt,
                    model="command-r-plus",
                    temperature=0.5
                )
                st.success(response.text)
            except Exception as e:
                st.error(f"Cohere Error: {e}")
else:
    st.info("Select a role and upload a file to begin.")
