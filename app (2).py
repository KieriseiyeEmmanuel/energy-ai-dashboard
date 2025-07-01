import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px
import cohere  # ✅ Using Cohere instead of OpenAI

st.set_page_config(page_title="AI Energy Analyst Dashboard", layout="wide")
st.title("🔬 AI Energy & Business Intelligence Dashboard")

uploaded_file = st.file_uploader("📥 Upload Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")

    cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("Enter Cohere API Key", type="password")

    # Automatic AI-based Data Summary and Suggestions
    if cohere_key:
        try:
            co = cohere.Client(cohere_key)
            preview = df.head(5).to_string(index=False)
            description = df.describe(include='all').fillna('').to_string()
            structure = f"Preview of Uploaded Data:\n{preview}\n\nSummary Stats:\n{description}"

            st.markdown("### 🧠 AI Auto Analysis")
            response = co.chat(
                message=f"You are a Chevron-style data analyst. Summarize this data, identify what kind of role it belongs to (finance, ops, market, etc), suggest insights, KPIs, and visualizations:\n\n{structure}",
                model="command-r-plus",
                temperature=0.5
            )
            st.success(response.text)
        except Exception as e:
            st.error(f"AI Error: {e}")

    # Ask AI anything feature
    st.markdown("### 🤖 Ask Vora Anything")
    question = st.text_area("Type a command or question about your data (e.g., 'What is the average cost?', 'Plot trend over time')")

    if st.button("Run Command") and cohere_key and question:
        try:
            sample = df.head(10).to_csv(index=False)
            prompt = f"You are an expert data analyst. Given this dataset (first 10 rows):\n\n{sample}\n\nNow: {question}"
            response = co.chat(
                message=prompt,
                model="command-r-plus",
                temperature=0.5
            )
            st.success(response.text)
        except Exception as e:
            st.error(f"AI Error: {e}")
else:
    st.info("Upload a file to begin.")
