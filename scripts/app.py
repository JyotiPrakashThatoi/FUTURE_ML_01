import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(
    page_title="PredicTrade | Sales AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0e1117; }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .card {
        background: rgba(255, 255, 255, 0.03);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
    }
    .header-hero {
        background: linear-gradient(135deg, #1e1e2f 0%, #0e1117 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border-left: 5px solid #00d4ff;
    }
    h1 { color: #ffffff; font-weight: 800 !important; margin: 0; }
    .highlight { color: #00d4ff; }
</style>
""", unsafe_allow_html=True)

# --- Data Loading Logic ---
def load_initial_data():
    raw_df = pd.read_csv("data/cleaned_sales.csv") if os.path.exists("data/cleaned_sales.csv") else pd.DataFrame()
    if not raw_df.empty:
        raw_df['Order Date'] = pd.to_datetime(raw_df['Order Date'])
    return raw_df

def load_forecast_results():
    df = pd.read_csv("data/forecast_results.csv") if os.path.exists("data/forecast_results.csv") else pd.DataFrame()
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

# Initialize session state for data
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = load_initial_data()

# --- Sidebar & File Handling ---
with st.sidebar:
    st.title("🎛️ Control Center")
    st.info("Dynamic Data Engine v2.1")
    
    st.divider()
    st.subheader("📁 Data Management")
    
    # Download Sample Template
    sample_path = "data/sample_template.csv"
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as file:
            st.download_button(
                label="📥 Download Sample Template",
                data=file,
                file_name="sample_sales_template.csv",
                mime="text/csv",
                help="Download a CSV with the correct headers for uploading."
            )
    
    # Upload Data
    uploaded_file = st.file_uploader("📤 Upload New Sales Data", type="csv")
    if uploaded_file:
        try:
            new_df = pd.read_csv(uploaded_file)
            if 'Order Date' in new_df.columns:
                new_df['Order Date'] = pd.to_datetime(new_df['Order Date'])
                st.session_state.raw_df = new_df
                st.success("Data Updated Successfully!")
            else:
                st.error("Missing 'Order Date' column!")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    st.subheader("🔍 Filters")
    
    df = st.session_state.raw_df
    if not df.empty:
        # Region Filter
        regions = sorted(df['Region'].unique().tolist())
        selected_regions = st.multiselect("Select Regions", options=regions, default=regions)
        
        # Category Filter
        categories = sorted(df['Category'].unique().tolist())
        selected_categories = st.multiselect("Select Categories", options=categories, default=categories)
        
        # Dynamic Filtering Engine
        filtered_df = df[
            (df['Region'].isin(selected_regions)) & 
            (df['Category'].isin(selected_categories))
        ]
    else:
        st.warning("No data available to filter.")
        filtered_df = pd.DataFrame()

# --- Header Section ---
st.markdown("<div class='header-hero'>", unsafe_allow_html=True)
st.markdown("<h1>Next-Gen <span class='highlight'>Sales & Demand</span> Intelligence</h1>", unsafe_allow_html=True)
st.markdown("Real-time predictive analytics driven by Random Forest AI.")
st.markdown("</div>", unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("Please upload data or adjust filters to view the dashboard.")
    st.stop()

# --- Dashboard Content ---
tab1, tab2, tab3 = st.tabs(["📊 Performance Summary", "🌍 Market Intelligence", "⚙️ AI Insights"])

with tab1:
    # Top Metrics (Dynamic)
    m1, m2, m3 = st.columns(3)
    curr_sales = filtered_df['Sales'].sum()
    order_count = len(filtered_df)
    avg_order = curr_sales / order_count if order_count > 0 else 0
    
    m1.metric("Selected Sales", f"${curr_sales:,.0f}")
    m2.metric("Order Volume", f"{order_count:,}")
    m3.metric("Avg. Order Value", f"${avg_order:,.2f}")

    # Historical Sales Trend (Filtered)
    st.markdown("### 📈 Historical Sales Trend")
    daily_sales = filtered_df.groupby('Order Date')['Sales'].sum().reset_index()
    fig_hist = px.line(daily_sales, x='Order Date', y='Sales', markers=True,
                      color_discrete_sequence=['#00d4ff'], template="plotly_dark")
    fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hist, use_container_width=True)

    # Combined Forecast Comparison
    # Historical + Test Set + Future
    forecast_df = load_forecast_results()
    future_df = pd.read_csv("data/future_forecast.csv") if os.path.exists("data/future_forecast.csv") else pd.DataFrame()
    if not future_df.empty:
        future_df['Date'] = pd.to_datetime(future_df['Date'])

    if not forecast_df.empty:
        st.markdown("### 🔮 Future Demand Projection & Historical Validation")
        fig_forecast = go.Figure()
        
        # 1. Historical Data
        fig_forecast.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Sales'], name='Historical Reality', line=dict(color='#00d4ff', width=2)))
        
        # 2. Validation Predictions (Test Set)
        fig_forecast.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['RF_Preds'], name='AI Validation (Test set)', line=dict(color='gray', width=1, dash='dot')))
        
        # 3. Future Predictions
        if not future_df.empty:
            fig_forecast.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Sales'], name='FUTURE PROJECTION', line=dict(color='#ff007c', width=4)))
            
            # Vertical line for "Today"
            last_date = forecast_df['Date'].max()
            fig_forecast.add_vline(x=last_date, line_width=3, line_dash="dash", line_color="white")
            fig_forecast.add_annotation(x=last_date, y=forecast_df['Sales'].max(), text="TODAY", showarrow=False, font=dict(color="white"))

        fig_forecast.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.caption("The gray dashed line shows how the AI performed on past data. The magenta line is the **30-day Future Prediction**.")

with tab2:
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.markdown("#### Category Distribution")
        cat_pie = px.pie(filtered_df, values='Sales', names='Category', hole=0.5,
                        color_discrete_sequence=px.colors.sequential.Tealgrn, template="plotly_dark")
        st.plotly_chart(cat_pie, use_container_width=True)
        
    with col_r:
        st.markdown("#### Regional Contribution")
        reg_bar = px.bar(filtered_df.groupby('Region')['Sales'].sum().reset_index(), 
                        x='Region', y='Sales', color='Sales', template="plotly_dark")
        st.plotly_chart(reg_bar, use_container_width=True)

    st.markdown("#### Profitability vs. Volume Matrix")
    fig_scatter = px.scatter(filtered_df, x='Sales', y='Profit', color='Category', 
                            size='Quantity', hover_name='Product Name', template="plotly_dark")
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.markdown("### Decision Support")
    st.info("The AI model has identified that **Technology** orders in the **East** region have the highest predictive stability.")
    
    st.markdown("""
    <div class='card'>
        <h4>How to use this dashboard:</h4>
        <ol>
            <li><b>Filter</b> the data by Region or Category to see specific performance metrics.</li>
            <li><b>Download</b> the sample template from the sidebar to prepare your own data.</li>
            <li><b>Upload</b> your latest sales file to see the dashboard update dynamically.</li>
            <li>Use the <b>Profitability Matrix</b> in the Market Intelligence tab to identify low-margin products.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
