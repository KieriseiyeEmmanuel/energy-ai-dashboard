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
from datetime import datetime

# 🌌 Set Sci-Fi theme & layout
st.set_page_config(
    page_title="Vora – AI Chevron Analyst",
    layout="wide",
    page_icon="🛸"
)

# 🌌 Custom CSS for a SciFi Look
st.markdown("""
    <style>
    body, .main {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Orbitron', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #111b26 100%);
        color: #c9d1d9;
    }
    .stTextInput > div > div > input, .stTextArea > div > textarea {
        background-color: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #58a6ff;
        color: black;
        border-radius: 1rem;
        font-weight: bold;
    }
    .stSidebar {
        background-color: #0d1117;
    }
    .css-1d391kg, .css-1kyxreq, .css-1cpxqw2, .css-1offfwp, .css-1v0mbdj {
        color: #58a6ff !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛸 Vora – Chevron-Grade AI Energy Intelligence")
st.markdown("Empowering analysts with AI-driven decision tools in a futuristic interface.")

roles = [
    "📊 Financial Analyst (NPV/IRR)",
    "📈 Data Analyst (Trends)",
    "🌍 Market Intelligence (Oil Prices)",
    "📘 Energy Economist (Policy)",
    "⚙️ Supply Chain Analyst (Logistics)",
    "🛢️ Operations Analyst",
    "🔮 Forecasting (Prophet)",
    "🤖 Ask Vora (AI Assistant)"
]
selected_role = st.sidebar.selectbox("Select Analyst Role", roles)

uploaded_file = st.file_uploader("📥 Upload Chevron-Style CSV/Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Read file based on extension
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.sidebar.success("✅ File uploaded successfully!")
    st.sidebar.dataframe(df.head(3))

    def generate_pdf_report(title, summary):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(33, 37, 41)
        pdf.set_fill_color(240, 248, 255)
        pdf.cell(200, 10, txt=title, ln=True, align='C')
        pdf.multi_cell(0, 10, txt=summary)
        buffer = io.BytesIO()
        pdf.output(buffer)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">📄 Download PDF Report</a>'
        return href

    def ai_insight(prompt_text):
        cohere_key = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else st.text_input("Enter Cohere API Key", type="password")
        if cohere_key:
            try:
                co = cohere.Client(cohere_key)
                df_sample = df.head().to_string(index=False)
                summary = df.describe(include='all').fillna('').to_string()
                context = f"Data Preview:\n{df_sample}\n\nSummary:\n{summary}"
                response = co.chat(
                    message=f"{prompt_text}\n\n{context}",
                    model="command-r-plus",
                    temperature=0.4
                )
                st.success(response.text)
            except Exception as e:
                st.error(f"Cohere Error: {e}")

    # Role-specific analysis
    if selected_role == "📊 Financial Analyst (NPV/IRR)":
        st.header("📊 Project Finance Dashboard")
        if {'Project', 'Year', 'Cash Flow (USD)'}.issubset(df.columns):
            project = st.selectbox("Select Project", df["Project"].unique())
            rate = st.slider("Discount Rate (%)", 0.01, 0.3, 0.1, 0.01)
            
            project_df = df[df["Project"] == project].sort_values("Year")
            cash_flows = project_df["Cash Flow (USD)"].tolist()
            
            # Calculate metrics
            npv = sum(cf / (1 + rate)**(i+1) for i, cf in enumerate(cash_flows))
            try:
                irr = npf.irr(cash_flows)
            except:
                irr = None
            cumulative = np.cumsum(cash_flows)
            payback = next((i+1 for i, v in enumerate(cumulative) if v >= 0), None)

            col1, col2, col3 = st.columns(3)
            col1.metric("NPV", f"${npv/1e6:,.2f}M" if abs(npv) > 1e6 else f"${npv:,.2f}")
            col2.metric("IRR", f"{irr:.2%}" if irr else "N/A")
            col3.metric("Payback", f"{payback} years" if payback else "Beyond range")

            # Visualizations
            fig = px.line(project_df, x="Year", y="Cash Flow (USD)", 
                         title=f"Cash Flow Timeline - {project}",
                         markers=True)
            fig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)
            
            st.plotly_chart(px.bar(project_df, x="Year", y="Cash Flow (USD)", 
                                 title="Annual Cash Flows"), use_container_width=True)
            
            st.markdown("\n🔍 **AI Insight**")
            ai_insight(f"Analyze the financial viability of {project} project with NPV=${npv/1e6:.2f}M and IRR={irr:.2%}. Provide recommendations.")

    elif selected_role == "📈 Data Analyst (Trends)":
        st.header("📈 Production & Emissions Analysis")
        if {'Date', 'Production Volume (bbl)', 'CO2 Emissions (tons)'}.issubset(df.columns):
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Calculate efficiency metric
            df['Emissions Intensity'] = df['CO2 Emissions (tons)'] / df['Production Volume (bbl)']
            
            col1, col2 = st.columns(2)
            col1.metric("Avg Daily Production", f"{df['Production Volume (bbl)'].mean():,.0f} bbl")
            col2.metric("Avg Emissions Intensity", f"{df['Emissions Intensity'].mean():.4f} tons/bbl")
            
            # Time series plot
            fig = px.line(df, x='Date', y=['Production Volume (bbl)', 'CO2 Emissions (tons)'],
                         title="Production vs Emissions Over Time",
                         labels={"value": "Metric", "variable": "Measure"})
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            st.subheader("Correlation Analysis")
            fig = px.scatter(df, x='Production Volume (bbl)', y='CO2 Emissions (tons)',
                           trendline="ols",
                           title="Production vs Emissions Correlation")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("\n🔍 **AI Insight**")
            ai_insight("Analyze production-emissions relationship and suggest optimization strategies.")

    elif selected_role == "🌍 Market Intelligence (Oil Prices)":
        st.header("🌍 Crude Oil Price Dashboard")
        if {'Date', 'WTI Price ($)', 'Brent Price ($)'}.issubset(df.columns):
            df['Date'] = pd.to_datetime(df['Date'])
            df['Price Spread'] = df['Brent Price ($)'] - df['WTI Price ($)']
            
            # Calculate 7-day rolling averages
            df['WTI_7d_MA'] = df['WTI Price ($)'].rolling(7).mean()
            df['Brent_7d_MA'] = df['Brent Price ($)'].rolling(7).mean()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current WTI", f"${df['WTI Price ($)'].iloc[-1]:.2f}")
            col2.metric("Current Brent", f"${df['Brent Price ($)'].iloc[-1]:.2f}")
            col3.metric("Spread", f"${df['Price Spread'].iloc[-1]:.2f}")
            
            # Price chart
            fig = px.line(df, x='Date', y=['WTI Price ($)', 'Brent Price ($)', 'WTI_7d_MA', 'Brent_7d_MA'],
                         title="Crude Oil Prices with 7-Day Moving Averages",
                         labels={"value": "Price ($)", "variable": "Price Type"})
            st.plotly_chart(fig, use_container_width=True)
            
            # Spread analysis
            st.plotly_chart(px.area(df, x='Date', y='Price Spread',
                                  title="Brent-WTI Price Spread"), use_container_width=True)
            
            st.markdown("\n🔍 **AI Insight**")
            ai_insight("Analyze current oil price trends and spread dynamics. What market factors might be influencing this?")

    elif selected_role == "📘 Energy Economist (Policy)":
        st.header("📘 Policy Scenario Analysis")
        if {'Policy', 'Projected Demand (MMbbl/day)', 'Emissions Reduction (%)', 'Cost Impact ($/bbl)'}.issubset(df.columns):
            # Policy comparison
            fig = px.bar(df, x='Policy', y=['Projected Demand (MMbbl/day)', 'Emissions Reduction (%)'],
                        barmode='group',
                        title="Policy Impact Comparison")
            st.plotly_chart(fig, use_container_width=True)
            
            # Cost-benefit analysis
            fig = px.scatter(df, x='Emissions Reduction (%)', y='Cost Impact ($/bbl)',
                           color='Policy', size='Projected Demand (MMbbl/day)',
                           title="Cost vs Emissions Reduction Tradeoff")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("\n🔍 **AI Insight**")
            ai_insight("Compare these policy scenarios from an energy economics perspective. Which offers the best balance of emissions reduction and economic impact?")

    elif selected_role == "⚙️ Supply Chain Analyst (Logistics)":
        st.header("⚙️ Well Performance Dashboard")
        if {'Well', 'Daily Output (bbl)', 'Uptime (%)', 'Pressure (PSI)'}.issubset(df.columns):
            # Top/bottom performers
            st.subheader("Top Performers")
            top_wells = df.nlargest(5, 'Daily Output (bbl)')
            st.dataframe(top_wells)
            
            st.subheader("Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Daily Output", f"{df['Daily Output (bbl)'].mean():,.0f} bbl")
            col2.metric("Avg Uptime", f"{df['Uptime (%)'].mean():.1f}%")
            col3.metric("Avg Pressure", f"{df['Pressure (PSI)'].mean():,.0f} PSI")
            
            # Relationship analysis
            fig = px.scatter(df, x='Pressure (PSI)', y='Daily Output (bbl)',
                           color='Uptime (%)',
                           title="Output vs Pressure with Uptime Coloring")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("\n🔍 **AI Insight**")
            ai_insight("Analyze well performance data. Identify any relationships between pressure, output, and uptime.")

    elif selected_role == "🛢️ Operations Analyst":
        st.header("🛢️ Facility Operations Dashboard")
        if {'Operation ID', 'Facility', 'Downtime (hrs)', 'Utilization Rate (%)', 'Maintenance Cost (USD)'}.issubset(df.columns):
            # Facility comparison
            facility = st.selectbox("Select Facility", df['Facility'].unique())
            facility_df = df[df['Facility'] == facility]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Downtime", f"{facility_df['Downtime (hrs)'].mean():.1f} hrs")
            col2.metric("Avg Utilization", f"{facility_df['Utilization Rate (%)'].mean():.1f}%")
            col3.metric("Avg Maint Cost", f"${facility_df['Maintenance Cost (USD)'].mean():,.0f}")
            
            # Trend analysis
            fig = px.line(df, x='Operation ID', y=['Downtime (hrs)', 'Utilization Rate (%)'],
                         color='Facility',
                         title="Performance Across Facilities")
            st.plotly_chart(fig, use_container_width=True)
            
            # Cost analysis
            st.plotly_chart(px.box(df, x='Facility', y='Maintenance Cost (USD)',
                                 title="Maintenance Cost Distribution by Facility"), use_container_width=True)
            
            st.markdown("\n🔍 **AI Insight**")
            ai_insight(f"Analyze operational performance for {facility}. Identify improvement opportunities in downtime and utilization.")

    elif selected_role == "🔮 Forecasting (Prophet)":
        st.header("🔮 Time Series Forecasting")
        if "Date" in df.columns:
            value_col = st.selectbox("Select Metric to Forecast", 
                                  options=[col for col in df.columns if df[col].dtype in ['float64', 'int64']])
            
            df_forecast = df[['Date', value_col]].dropna()
            df_forecast.columns = ['ds', 'y']
            df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])
            
            m = Prophet(seasonality_mode='multiplicative')
            m.fit(df_forecast)
            future = m.make_future_dataframe(periods=90)  # 3 months forecast
            forecast = m.predict(future)
            
            st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)
            
            st.subheader("Forecast Components")
            fig2 = m.plot_components(forecast)
            st.pyplot(fig2)
            
            st.markdown("\n🔍 **AI Insight**")
            ai_insight(f"Interpret the forecast for {value_col}. What trends and seasonality patterns do you observe?")

    elif selected_role == "🤖 Ask Vora (AI Assistant)":
        st.header("🤖 AI Energy Analyst Assistant")
        question = st.text_input("Ask anything about your data:")
        if question:
            ai_insight(question)

else:
    st.info("👈 Select a role and upload your Chevron-style data file to begin.")
    st.markdown("### Sample Data Formats:")
    
    with st.expander("📊 Financial Analyst Data"):
        st.write("Columns: Project, Year, Cash Flow (USD)")
        st.code("""Permian Expansion,2023,-150000000
Permian Expansion,2024,25000000
Permian Expansion,2025,80000000""")
    
    with st.expander("📈 Data Analyst Data"):
        st.write("Columns: Date, Production Volume (bbl), CO2 Emissions (tons)")
        st.code("""01/01/2023,50000,1200
01/02/2023,52500,1250
01/03/2023,48000,1180""")

    with st.expander("⚙️ Operations Analyst Data"):
        st.write("Columns: Operation ID, Facility, Downtime (hrs), Utilization Rate (%), Maintenance Cost (USD)")
        st.code("""OP-001,Refinery A,12,88,250000
OP-002,Offshore Rig B,24,75,500000
OP-003,Pipeline C,8,92,180000""")
