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

st.set_page_config(page_title="Vora – Chevron-Ready AI Energy Dashboard", layout="wide")
st.title("🗂️ Vora – Chevron-Grade AI Energy Intelligence Dashboard")

roles = [
    "Project Finance & Economics",
    "Production & Operations Analyst",
    "Market Intelligence",
    "Energy Policy Scenarios",
    "Supply Chain & Logistics",
    "Ask Vora (AI Assistant)"
]
selected_role = st.sidebar.radio("Select Analyst Role", roles)

uploaded_file = st.file_uploader("📅 Upload Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")

    def generate_pdf_report(title, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.multi_cell(0, 10, txt=summary)

    # ✅ Return PDF as bytes using `dest='S'`
    pdf_bytes = pdf.output(dest='S').encode('latin1')  # required encoding
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">📄 Download Finance Report</a>'
    return href


    def ai_insight(prompt_text):
        cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("Enter Cohere API Key", type="password")
        if cohere_key:
            try:
                co = cohere.Client(cohere_key)
                df_sample = df.head().to_string(index=False)
                summary = df.describe(include='all').fillna('').to_string()
                context = f"Data Sample:\n{df_sample}\n\nSummary:\n{summary}"
                response = co.chat(
                    message=f"{prompt_text}\n\n{context}",
                    model="command-r-plus",
                    temperature=0.4
                )
                st.success(response.text)
            except Exception as e:
                st.error(f"Cohere Error: {e}")

    if selected_role == "Project Finance & Economics":
        st.header("💰 Chevron-Style Project Financial Evaluation")
        expected_cols = {"Project", "Year", "Cash Flow (USD)"}
        if not expected_cols.issubset(df.columns):
            st.error("❌ File must contain: Project, Year, Cash Flow (USD)")
        else:
            project = st.selectbox("Select Project", df["Project"].unique())
            rate = st.slider("Discount Rate (%)", 0.01, 0.3, 0.1)
            project_df = df[df["Project"] == project]
            cash_flows = project_df["Cash Flow (USD)"].dropna().tolist()

            if not cash_flows:
                st.warning("⚠️ No valid cash flows available for this project.")
            else:
                npv = sum(cf / (1 + rate)**i for i, cf in enumerate(cash_flows))
                try:
                    irr = npf.irr(cash_flows)
                    irr_str = f"{irr:.2%}" if np.isfinite(irr) else "N/A"
                except:
                    irr_str = "N/A"
                cumulative = np.cumsum(cash_flows)
                payback = next((i for i, v in enumerate(cumulative) if v >= 0), None)
                payback_str = f"{payback} years" if payback is not None else "Beyond range"

                col1, col2, col3 = st.columns(3)
                col1.metric("NPV", f"${npv:,.2f}")
                col2.metric("IRR", irr_str)
                col3.metric("Payback", payback_str)

                st.subheader("📊 Cash Flow Timeline")
                st.plotly_chart(px.bar(project_df, x="Year", y="Cash Flow (USD)", color="Project"), use_container_width=True)

                st.markdown(generate_pdf_report("Finance Report", f"Project: {project}\nNPV: ${npv:,.2f}\nIRR: {irr_str}\nPayback: {payback_str}"), unsafe_allow_html=True)
                st.write("\n🔎 **AI Insights**")
                ai_insight("You are a Chevron project finance analyst. Provide insights into the uploaded project finance data.")

    elif selected_role == "Production & Operations Analyst":
        st.header("⛽ Production & Operational Dashboard")
        st.dataframe(df.head())

        if "Month" in df.columns and "Oil Production (bbl)" in df.columns:
            st.line_chart(df.set_index("Month")["Oil Production (bbl)"])

        if "Downtime (hrs)" in df.columns:
            st.write("### Downtime Analysis")
            st.bar_chart(df.set_index("Month")["Downtime (hrs)"])

        if "Uptime (%)" in df.columns:
            st.write("### Uptime Trends")
            st.line_chart(df.set_index("Month")["Uptime (%)"])

        st.write("\n🔎 **AI Insights**")
        ai_insight("As an operations analyst, evaluate production and uptime/downtime behavior in the dataset.")

    elif selected_role == "Market Intelligence":
        st.header("📈 Oil Market Monitoring")
        st.dataframe(df.head())

        if "Date" in df.columns:
            st.plotly_chart(px.line(df, x="Date", y=[col for col in df.columns if "Price" in col], title="Price Indexes"))

        if "Global Demand (mb/d)" in df.columns and "Global Supply (mb/d)" in df.columns:
            st.plotly_chart(px.line(df, x="Date", y=["Global Demand (mb/d)", "Global Supply (mb/d)"], title="Demand vs Supply"))

        st.write("\n🔎 **AI Insights**")
        ai_insight("As a market intelligence analyst, interpret the trends and economic implications from oil price and supply-demand data.")

    elif selected_role == "Energy Policy Scenarios":
        st.header("📘 Policy & Regulatory Forecasting")
        st.dataframe(df)

        if "Scenario" in df.columns:
            st.plotly_chart(px.bar(df, x="Scenario", y="CO₂ Emissions (Mt)", color="Policy", title="Emissions by Policy"))
            st.plotly_chart(px.scatter(df, x="CO₂ Emissions (Mt)", y="GDP Growth (%)", color="Scenario", size="Carbon Tax ($/ton)", hover_data=["Policy"]))

        st.write("\n🔎 **AI Insights**")
        ai_insight("You're an energy policy expert. Evaluate the impact of different carbon tax policies on emissions and GDP.")

    elif selected_role == "Supply Chain & Logistics":
        st.header("🚢 Logistics & KPI Analysis")
        st.dataframe(df.head())

        kpi_cols = [col for col in df.columns if col not in ["Month", "Site"] and df[col].dtype != 'O']
        for col in kpi_cols:
            st.line_chart(df.pivot(index="Month", columns="Site", values=col), use_container_width=True)

        selected_kpi = st.selectbox("Select KPI for Box Plot", kpi_cols)
        st.plotly_chart(px.box(df, x="Site", y=selected_kpi, title=f"{selected_kpi} Distribution"))

        st.write("\n🔎 **AI Insights**")
        ai_insight("As a Chevron logistics analyst, provide a logistics and supply chain KPI analysis.")

    elif selected_role == "Ask Vora (AI Assistant)":
        st.header("🧐 Ask Vora: File-Aware AI Assistant")
        cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("Enter Cohere API Key", type="password")
        question = st.text_area("Ask Vora something about your uploaded file:")

        if st.button("Ask Vora") and cohere_key and question:
            try:
                co = cohere.Client(cohere_key)
                df_sample = df.head().to_string(index=False)
                summary = df.describe(include='all').fillna('').to_string()
                context = f"Here is a preview of the uploaded dataset:\n\n{df_sample}\n\nSummary:\n{summary}"
                prompt = f"You are a Chevron-level data advisor. Based on this data, answer the user's question intelligently.\n\n{context}\n\nQuestion: {question}"
                response = co.chat(message=prompt, model="command-r-plus", temperature=0.5)
                st.success(response.text)
            except Exception as e:
                st.error(f"Cohere Error: {e}")
else:
    st.info("👈 Select a role and upload your Chevron-style Excel file to begin.")
