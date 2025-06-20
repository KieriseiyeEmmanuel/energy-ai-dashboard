import streamlit as st
import numpy_financial as npf

# Main page
st.title("Energy AI Dashboard (Demo)")
st.write("This demo includes the GPT-4 advisor and NPV/IRR module.")

upload = st.file_uploader("Upload Excel with 'Project' and 'Cash Flow (USD)' columns", type=["xlsx"])
if upload:
    import pandas as pd, numpy as np
    df = pd.read_excel(upload)
    st.write(df.head())
    project = st.selectbox("Project", df["Project"].unique())
    rate = st.slider("Discount Rate (%)", 0.01, 0.25, 0.1)
    cash = df[df.Project == project]["Cash Flow (USD)"].tolist()
    npv = sum(cf/(1+rate)**i for i, cf in enumerate(cash))
    irr = npf.irr(cash)
    st.metric("NPV", f"${npv:.2f}")
    st.metric("IRR", f"{irr:.2%}")

st.write("Chat with AI below (demo stub)...")
