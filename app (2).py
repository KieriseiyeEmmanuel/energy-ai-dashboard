import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cohere
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from prophet import Prophet
from prophet.plot import plot_plotly

# ----- CONFIG -----
st.set_page_config(page_title="🛸 Vora – Chevron AI Analyst", layout="wide")

# ----- STYLING -----
st.markdown("""
    <style>
    body, .main {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Orbitron', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #111b26 100%);
    }
    .stTextInput > div > div > input, .stTextArea > div > textarea {
        background-color: #161b22;
        color: #c9d1d9;
    }
    .stButton > button {
        background-color: #58a6ff;
        color: black;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛸 Vora – Chevron-Grade AI Analyst")

# ----- FILE UPLOAD -----
uploaded_file = st.file_uploader("📥 Upload your Chevron-style Excel file", type=["xlsx"])
cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("Enter Cohere API Key", type="password")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ File uploaded successfully")

    st.subheader("🔍 AI Role Identification & Insights")
    if cohere_key:
        try:
            co = cohere.Client(cohere_key)
            sample = df.head(5).to_string(index=False)
            stats = df.describe(include='all').fillna('').to_string()
            structure = f"Sample Data:\n{sample}\n\nStats:\n{stats}"

            detect_prompt = f"""
            You are an AI that analyzes Excel data. Detect which of the following roles this dataset fits:
            - Finance
            - Operations
            - Market
            - Supply Chain
            - Policy
            Then suggest key KPIs and chart types.\n\n{structure}
            """

            response = co.chat(message=detect_prompt, model="command-r-plus", temperature=0.4)
            st.info(response.text)
        except Exception as e:
            st.error(f"Cohere Error: {e}")

    # ----- DYNAMIC KPI & CHARTS -----
    st.subheader("📊 Dynamic KPIs & Visuals")
    if {'Cash Flow (USD)', 'Project'}.issubset(df.columns):
        project = st.selectbox("Select Project", df["Project"].unique())
        proj_df = df[df["Project"] == project]
        cash_flows = proj_df["Cash Flow (USD)"].tolist()
        npv = sum(cf / (1 + 0.1)**i for i, cf in enumerate(cash_flows))
        irr = np.irr(cash_flows)
        cumulative = np.cumsum(cash_flows)
        payback = next((i for i, v in enumerate(cumulative) if v >= 0), None)

        col1, col2, col3 = st.columns(3)
        col1.metric("NPV", f"${npv:,.2f}")
        col2.metric("IRR", f"{irr:.2%}" if irr else "N/A")
        col3.metric("Payback", f"{payback} years" if payback else "Not recovered")

        st.plotly_chart(px.bar(proj_df, x=proj_df.index, y="Cash Flow (USD)", title="Project Cash Flow"))
    
    elif {'Well', 'Daily Output'}.issubset(df.columns):
        st.metric("Avg Daily Output", f"{df['Daily Output'].mean():,.2f} bbl")
        st.metric("Max Output", f"{df['Daily Output'].max():,.2f} bbl")
        st.plotly_chart(px.line(df, x='Well', y='Daily Output', title="Production per Well"))

    elif {'Brent Price', 'WTI Price'}.issubset(df.columns):
        df['Date'] = pd.to_datetime(df['Date'])
        st.line_chart(df.set_index('Date')[['Brent Price', 'WTI Price']])

    elif {'Route', 'Delivery Time (days)'}.issubset(df.columns):
        st.plotly_chart(px.box(df, x="Route", y="Delivery Time (days)", title="Delivery Efficiency"))

    # ----- ASK VORA -----
    st.subheader("🤖 Ask Vora (AI Assistant)")
    user_q = st.text_area("Ask anything about your data (e.g. average cost, plot pressure trend):")
    if st.button("Ask Vora") and user_q and cohere_key:
        try:
            sample = df.head(10).to_csv(index=False)
            prompt = f"You are a Chevron-grade data analyst. Analyze this sample:\n{sample}\n\nNow answer: {user_q}"
            response = co.chat(message=prompt, model="command-r-plus", temperature=0.5)
            st.success(response.text)
        except Exception as e:
            st.error(f"Vora AI Error: {e}")
else:
    st.info("⬅️ Upload a file to get started.")
