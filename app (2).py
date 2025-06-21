import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px
import cohere  # ✅ NEW: Using Cohere instead of OpenAI

st.set_page_config(page_title="AI Energy Analyst Dashboard", layout="wide")
st.title("🔬 AI Energy & Business Intelligence Dashboard")

uploaded_file = st.file_uploader("📥 Upload Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("File uploaded!")

    roles = [
        "Financial Analyst (NPV/IRR/Payback)",
        "Data Analyst (Trends & Correlations)",
        "Market Intelligence (Oil Prices)",
        "Energy Economist (Policy Scenarios)",
        "Supply Chain Analyst (Logistics KPIs)",
        "AI Business Advisor (Cohere)"
    ]
    selected_role = st.sidebar.radio("Select Role Module", roles)

    if selected_role == "Financial Analyst (NPV/IRR/Payback)":
        st.header("📈 Financial Evaluation")
        st.info("Ensure your Excel includes: 'Project', 'Cash Flow (USD)'")

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

    elif selected_role == "AI Business Advisor (Cohere)":
        st.header("🤖 AI Project Advisor (Free via Cohere)")
        cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("Enter Cohere API Key", type="password")
        question = st.text_area("Ask a project-related question:")

        if st.button("Ask AI") and cohere_key and question:
            try:
                co = cohere.Client(cohere_key)
                response = co.chat(
                    message=question,
                    model="command-r-plus",
                    temperature=0.7
                )
                st.success(response.text)
            except Exception as e:
                st.error(f"Cohere Error: {e}")
else:
    st.info("Upload a file to begin.")
