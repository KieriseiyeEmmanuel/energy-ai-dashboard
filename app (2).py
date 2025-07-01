import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px
import cohere  # ✅ Using Cohere instead of OpenAI

st.set_page_config(page_title="AI Energy Analyst Dashboard", layout="wide")
st.title("🔬 AI Energy & Business Intelligence Dashboard")

roles = [
    "Financial Analyst (NPV/IRR/Payback)",
    "Data Analyst (Trends & Correlations)",
    "Market Intelligence Analyst (Oil Prices)",
    "Energy Economist (Policy Scenarios)",
    "Supply Chain Analyst (Logistics KPIs)",
    "Production & Operations Analyst",
    "AI Business Advisor (Ask Vora)"
]
selected_role = st.sidebar.radio("Select Role Module", roles)

uploaded_file = st.file_uploader("📥 Upload Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("File uploaded!")

    def ai_insight(prompt_text):
        cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("Enter Cohere API Key", type="password")
        if cohere_key:
            try:
                co = cohere.Client(cohere_key)
                context = df.head(10).to_csv(index=False)
                response = co.chat(
                    message=f"{prompt_text}\n\nUse this data if helpful:\n{context}",
                    model="command-r-plus",
                    temperature=0.5
                )
                st.markdown("**🤖 AI Insight:**")
                st.success(response.text)
            except Exception as e:
                st.error(f"Cohere Error: {e}")

    if selected_role == "Financial Analyst (NPV/IRR/Payback)":
        st.header("📈 Financial Evaluation")
        required_cols = {"Project", "Cash Flow (USD)"}
        if required_cols.issubset(df.columns):
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
            ai_insight("You are a Chevron financial analyst. Provide insight on this project's cash flows and profitability.")
        else:
            st.warning(f"⚠️ Missing required columns: {', '.join(required_cols)}")

    elif selected_role == "Production & Operations Analyst":
        st.header("🏭 Production & Operations Dashboard")
        required_cols = {"Operation ID", "Facility", "Downtime (hrs)", "Utilization Rate (%)", "Maintenance Cost (USD)"}
        if required_cols.issubset(df.columns):
            st.dataframe(df)
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Downtime", f"{df['Downtime (hrs)'].mean():.1f} hrs")
            col2.metric("Utilization", f"{df['Utilization Rate (%)'].mean():.1f}%")
            col3.metric("Total Maintenance", f"${df['Maintenance Cost (USD)'].sum():,.0f}")

            st.subheader("Downtime by Facility")
            st.plotly_chart(px.bar(df, x="Facility", y="Downtime (hrs)", color="Facility"))
            ai_insight("You are a Chevron operations analyst. Identify any facility-level inefficiencies and trends.")
        else:
            st.warning(f"⚠️ Missing required columns: {', '.join(required_cols)}")

    elif selected_role == "Market Intelligence Analyst (Oil Prices)":
        st.header("🌍 Oil Market Intelligence")
        required_cols = {"Date", "Brent Price", "WTI Price", "Demand (MBPD)", "Supply (MBPD)"}
        if required_cols.issubset(df.columns):
            df['Date'] = pd.to_datetime(df['Date'])
            st.line_chart(df.set_index("Date")[['Brent Price', 'WTI Price']])
            st.area_chart(df.set_index("Date")[['Demand (MBPD)', 'Supply (MBPD)']])
            ai_insight("You're a Chevron economist. Explain what's happening in the oil market using this data.")
        else:
            st.warning(f"⚠️ Missing required columns: {', '.join(required_cols)}")

    elif selected_role == "Energy Economist (Policy Scenarios)":
        st.header("📘 Energy Policy Scenarios")
        required_cols = {"Scenario", "CO2 Emissions", "Tax Rate", "Renewable Share (%)"}
        if required_cols.issubset(df.columns):
            st.bar_chart(df.set_index("Scenario")[['CO2 Emissions', 'Tax Rate']])
            st.plotly_chart(px.pie(df, values="Renewable Share (%)", names="Scenario"))
            ai_insight("You're a Chevron policy advisor. Discuss tradeoffs and benefits of each energy scenario.")
        else:
            st.warning(f"⚠️ Missing required columns: {', '.join(required_cols)}")

    elif selected_role == "Supply Chain Analyst (Logistics KPIs)":
        st.header("🚚 Logistics & Supply Chain Dashboard")
        required_cols = {"Route", "Delivery Time (days)", "Cost per Barrel", "Delays (#)"}
        if required_cols.issubset(df.columns):
            st.dataframe(df)
            st.plotly_chart(px.box(df, x="Route", y="Delivery Time (days)"))
            col1, col2 = st.columns(2)
            col1.metric("Avg Cost/Barrel", f"${df['Cost per Barrel'].mean():.2f}")
            col2.metric("Avg Delays", f"{df['Delays (#)'].mean():.2f}")
            ai_insight("You are a Chevron logistics analyst. Analyze delivery performance and cost efficiency.")
        else:
            st.warning(f"⚠️ Missing required columns: {', '.join(required_cols)}")

    elif selected_role == "Data Analyst (Trends & Correlations)":
        st.header("📊 Trends & Correlation Analysis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select numeric columns to analyze:", options=numeric_cols, default=numeric_cols[:2])
            if len(selected_cols) >= 2:
                fig = px.scatter_matrix(df[selected_cols])
                st.plotly_chart(fig, use_container_width=True)
                ai_insight("You're a data analyst. Identify interesting patterns and correlations in the selected data.")
        else:
            st.warning("⚠️ No numeric columns found in dataset.")

    elif selected_role == "AI Business Advisor (Ask Vora)":
        st.header("🤖 Ask Vora - Your AI Project Advisor")
        question = st.text_area("Ask a project-related question:")
        if question:
            ai_insight(question)
else:
    st.info("Upload a file to begin.")
