import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cohere  # ✅ Using Cohere instead of OpenAI

st.set_page_config(page_title="AI Energy Analyst Dashboard", layout="wide")
st.title("🔬 AI Energy & Business Intelligence Dashboard")

uploaded_file = st.file_uploader("📥 Upload Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")

    cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("Enter Cohere API Key", type="password")

    if cohere_key:
        co = cohere.Client(cohere_key)

        st.markdown("### 🧠 AI Smart Summary")
        preview = df.head(5).to_string(index=False)
        description = df.describe(include='all').fillna('').to_string()
        structure = f"Preview:\n{preview}\n\nDescription:\n{description}"

        summary_response = co.chat(
            message=f"As a Chevron energy analyst, analyze this dataset and give summary insights with KPI suggestions:\n{structure}",
            model="command-r-plus",
            temperature=0.5
        )
        st.success(summary_response.text)

        # 🧠 AI-Generated Visual Suggestions
        chart_response = co.chat(
            message=f"You are an AI data visualization assistant. Based on this data (head sample):\n\n{df.head(10).to_string(index=False)}\n\nSuggest 2 charts with column names, like: 'Bar chart of Year vs Revenue', 'Line chart of Date vs Production'.",
            model="command-r-plus",
            temperature=0.4
        )
        st.markdown("### 🧬 AI Chart Suggestions")
        st.info(chart_response.text)

        # 🧠 Try to auto-plot based on top numeric columns
        st.markdown("### 📈 Auto Visual Explorer")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            col_x = st.selectbox("Choose X-axis", numeric_cols, key="xaxis")
            col_y = st.selectbox("Choose Y-axis", numeric_cols, key="yaxis")
            chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Area"])
            if chart_type == "Line":
                st.plotly_chart(px.line(df, x=col_x, y=col_y), use_container_width=True)
            elif chart_type == "Bar":
                st.plotly_chart(px.bar(df, x=col_x, y=col_y), use_container_width=True)
            elif chart_type == "Area":
                st.plotly_chart(px.area(df, x=col_x, y=col_y), use_container_width=True)

        # 💳 KPI Cards (first 3 numerical columns)
        st.markdown("### 💳 KPI Cards")
        kpi_cols = numeric_cols[:3]
        k1, k2, k3 = st.columns(3)
        if len(kpi_cols) >= 1: k1.metric(kpi_cols[0], f"{df[kpi_cols[0]].mean():,.2f}")
        if len(kpi_cols) >= 2: k2.metric(kpi_cols[1], f"{df[kpi_cols[1]].mean():,.2f}")
        if len(kpi_cols) >= 3: k3.metric(kpi_cols[2], f"{df[kpi_cols[2]].mean():,.2f}")

        # 🔥 Correlation Heatmap
        st.markdown("### ♨️ Correlation Matrix")
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Numeric Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)

        # 🗣️ Ask anything
        st.markdown("### 🤖 Ask Vora About Your Data")
        user_q = st.text_area("Ask about trends, stats, or insights...")
        if st.button("Ask AI"):
            context_csv = df.head(15).to_csv(index=False)
            prompt = f"Here is the data:\n\n{context_csv}\n\nNow answer this: {user_q}"
            ai_response = co.chat(message=prompt, model="command-r-plus", temperature=0.5)
            st.success(ai_response.text)
else:
    st.info("Upload a file to begin.")
