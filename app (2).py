import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px
import openai

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
        "AI Business Advisor (ChatGPT)"
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

    elif selected_role == "AI Business Advisor (ChatGPT)":
        st.header("🤖 AI Project Advisor")
        openai_key = st.secrets["AUTH_KEY"] if "AUTH_KEY" in st.secrets else st.text_input("Enter OpenAI API Key", type="password")
        question = st.text_area("Ask a project-related question:")
        if st.button("Ask AI") and openai_key and question:
            openai.api_key = openai_key
            prompt = f"You are an energy business analyst. Answer the following: {question}"
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.success(response.choices[0].message.content)
            except Exception as e:
                st.error(f"OpenAI Error: {e}")
else:
    st.info("Upload a file to begin.")
