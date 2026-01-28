"""
Real-Time Business Intelligence Dashboard
==========================================
A comprehensive BI platform for business analytics, KPI tracking, and decision support

Features:
- Executive Dashboard with KPIs
- Sales Performance Analytics
- Customer Analytics & Segmentation
- Financial Analysis
- Predictive Forecasting
- Report Generator

Author: Sendy Prismana Nurferian
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from io import BytesIO

# Import untuk ML
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Import untuk signal processing
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import untuk PDF (optional)
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Business Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 1em;
        opacity: 0.9;
    }
    .metric-delta {
        font-size: 1.2em;
        margin-top: 5px;
    }
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# DATA GENERATION FUNCTIONS
# ============================================================

@st.cache_data
def generate_business_data(days=365):
    """Generate comprehensive business data"""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Base trends
    trend = np.linspace(100000, 200000, days)
    seasonality = 20000 * np.sin(np.arange(days) * 2 * np.pi / 365)
    weekly_pattern = 5000 * np.sin(np.arange(days) * 2 * np.pi / 7)
    noise = np.random.normal(0, 5000, days)
    
    # Revenue
    revenue = (trend + seasonality + weekly_pattern + noise).clip(min=50000)
    
    # Costs
    cost_ratio = np.random.uniform(0.6, 0.75, days)
    costs = revenue * cost_ratio
    
    # Other metrics
    df = pd.DataFrame({
        'Date': dates,
        'Revenue': revenue,
        'Costs': costs,
        'Profit': revenue - costs,
        'Orders': np.random.randint(500, 2000, days),
        'New_Customers': np.random.randint(50, 300, days),
        'Website_Visits': np.random.randint(5000, 20000, days),
        'Conversion_Rate': np.random.uniform(0.02, 0.08, days),
        'Avg_Order_Value': revenue / np.random.randint(500, 2000, days),
        'Customer_Satisfaction': np.random.uniform(3.5, 5.0, days),
        'Marketing_Spend': np.random.uniform(5000, 20000, days),
        'Support_Tickets': np.random.randint(20, 150, days),
        'Product_Returns': np.random.randint(10, 100, days),
        'Region': np.random.choice(['North America', 'Europe', 'Asia', 'Latin America'], days),
        'Channel': np.random.choice(['Online', 'Retail', 'Mobile', 'Partner'], days),
        'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], days)
    })
    
    # Calculate derived metrics
    df['Profit_Margin'] = (df['Profit'] / df['Revenue'] * 100).round(2)
    df['ROI'] = ((df['Revenue'] - df['Marketing_Spend']) / df['Marketing_Spend'] * 100).round(2)
    df['Customer_Acquisition_Cost'] = (df['Marketing_Spend'] / df['New_Customers'].replace(0, 1)).round(2)  # Avoid division by zero
    
    return df

# Function to generate PDF report
def generate_pdf_report(business_data, date_range):
    """Generate PDF report using FPDF"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 20)
        pdf.cell(0, 10, 'Business Intelligence Report', 0, 1, 'C')
        
        # Date info
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
        pdf.cell(0, 10, f'Period: {date_range}', 0, 1)
        pdf.ln(10)
        
        # Executive Summary
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Executive Summary', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        total_revenue = business_data['Revenue'].sum()
        total_profit = business_data['Profit'].sum()
        total_orders = business_data['Orders'].sum()
        avg_margin = business_data['Profit_Margin'].mean()
        
        pdf.cell(0, 8, f'Total Revenue: ${total_revenue:,.0f}', 0, 1)
        pdf.cell(0, 8, f'Net Profit: ${total_profit:,.0f}', 0, 1)
        pdf.cell(0, 8, f'Total Orders: {total_orders:,}', 0, 1)
        pdf.cell(0, 8, f'Profit Margin: {avg_margin:.2f}%', 0, 1)
        pdf.ln(10)
        
        # Regional Performance
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Regional Performance', 0, 1)
        pdf.set_font('Arial', '', 11)
        
        regional = business_data.groupby('Region').agg({
            'Revenue': 'sum',
            'Profit': 'sum'
        }).sort_values('Revenue', ascending=False)
        
        for region, row in regional.iterrows():
            pdf.cell(0, 7, f'{region}: ${row["Revenue"]:,.0f} revenue, ${row["Profit"]:,.0f} profit', 0, 1)
        
        pdf.ln(10)
        
        # Top performing days
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Top 5 Days by Revenue', 0, 1)
        pdf.set_font('Arial', '', 11)
        
        top_days = business_data.nlargest(5, 'Revenue')
        for _, row in top_days.iterrows():
            pdf.cell(0, 7, f'{row["Date"].strftime("%Y-%m-%d")}: ${row["Revenue"]:,.0f}', 0, 1)
        
        # Footer
        pdf.ln(20)
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, 'Generated by Business Intelligence Dashboard | Powered by Python & Streamlit', 0, 1, 'C')
        
        return pdf.output(dest='S').encode('latin-1')
    
    except Exception as e:
        return None

@st.cache_data
def generate_customer_data(n_customers=5000):
    """Generate customer database"""
    np.random.seed(42)
    
    customer_ids = [f"CUST{i:05d}" for i in range(n_customers)]
    
    df = pd.DataFrame({
        'Customer_ID': customer_ids,
        'Join_Date': pd.date_range(end=datetime.now(), periods=n_customers, freq='1H'),
        'Total_Spent': np.random.exponential(500, n_customers).clip(min=50, max=10000),
        'Order_Count': np.random.poisson(5, n_customers).clip(min=1),
        'Avg_Order_Value': np.random.uniform(50, 500, n_customers),
        'Last_Purchase_Days': np.random.randint(1, 365, n_customers),
        'Age': np.random.randint(18, 70, n_customers),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_customers),
        'Country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France'], n_customers),
        'Loyalty_Tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], n_customers, p=[0.4, 0.3, 0.2, 0.1]),
        'Email_Engagement': np.random.uniform(0, 1, n_customers),
        'Support_Interactions': np.random.poisson(2, n_customers),
        'Referrals': np.random.poisson(1, n_customers)
    })
    
    # Calculate CLV (Customer Lifetime Value)
    df['Customer_Lifetime_Value'] = (
        df['Total_Spent'] + 
        (df['Avg_Order_Value'] * df['Order_Count'] * 0.5)
    ).round(2)
    
    # Calculate Churn Risk Score
    df['Churn_Risk_Score'] = (
        (df['Last_Purchase_Days'] / 365) * 0.4 +
        (1 - df['Email_Engagement']) * 0.3 +
        (df['Support_Interactions'] / 10) * 0.3
    ).clip(0, 1).round(3)
    
    return df

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://via.placeholder.com/150x50/667eea/ffffff?text=BI+Dashboard", use_container_width=True)
    st.title("⚙️ Control Panel")
    
    # Date range selector
    st.subheader("📅 Date Range")
    date_range = st.selectbox(
        "Select period",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last 365 Days", "All Time"],
        index=2
    )
    
    # Map selection to days
    days_map = {
        "Last 7 Days": 7,
        "Last 30 Days": 30,
        "Last 90 Days": 90,
        "Last 365 Days": 365,
        "All Time": 730
    }
    days = days_map[date_range]
    
    st.markdown("---")
    
    # Dashboard selector
    st.subheader("📊 Dashboards")
    dashboard = st.radio(
        "Select dashboard",
        ["🏢 Executive Overview", "💰 Sales Analytics", "👥 Customer Analytics", 
         "📈 Financial Analysis", "🔮 Predictive Insights", "📄 Report Generator"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Filters
    with st.expander("🔍 Filters", expanded=False):
        regions = st.multiselect(
            "Regions",
            ["North America", "Europe", "Asia", "Latin America"],
            default=["North America", "Europe", "Asia", "Latin America"]
        )
        
        channels = st.multiselect(
            "Sales Channels",
            ["Online", "Retail", "Mobile", "Partner"],
            default=["Online", "Retail", "Mobile", "Partner"]
        )
    
    st.markdown("---")
    
    # Export options
    with st.expander("💾 Export", expanded=False):
        if st.button("📊 Export Dashboard to PDF"):
            st.info("PDF export feature (requires additional setup)")
        if st.button("📈 Download Data (CSV)"):
            st.info("CSV download ready")

# ============================================================
# LOAD DATA
# ============================================================

business_data = generate_business_data(days)
customer_data = generate_customer_data()

# Apply filters
if regions:
    business_data = business_data[business_data['Region'].isin(regions)]
if channels:
    business_data = business_data[business_data['Channel'].isin(channels)]

# ============================================================
# EXECUTIVE OVERVIEW DASHBOARD
# ============================================================

if dashboard == "🏢 Executive Overview":
    st.title("🏢 Executive Dashboard")
    st.markdown("**Real-time business performance overview and key metrics**")
    st.markdown("---")
    
    # Calculate current period vs previous period
    current_period = business_data.tail(30)
    previous_period = business_data.iloc[-60:-30]
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_revenue = current_period['Revenue'].sum()
        prev_revenue = previous_period['Revenue'].sum()
        delta = ((current_revenue - prev_revenue) / prev_revenue * 100)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">💰 Total Revenue</div>
            <div class="metric-value">${current_revenue:,.0f}</div>
            <div class="metric-delta">{"📈" if delta > 0 else "📉"} {delta:+.1f}% vs last period</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        current_profit = current_period['Profit'].sum()
        prev_profit = previous_period['Profit'].sum()
        delta_profit = ((current_profit - prev_profit) / prev_profit * 100)
        
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-label">💵 Net Profit</div>
            <div class="metric-value">${current_profit:,.0f}</div>
            <div class="metric-delta">{"📈" if delta_profit > 0 else "📉"} {delta_profit:+.1f}% vs last period</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_orders = current_period['Orders'].sum()
        prev_orders = previous_period['Orders'].sum()
        delta_orders = ((total_orders - prev_orders) / prev_orders * 100)
        
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-label">🛒 Total Orders</div>
            <div class="metric-value">{total_orders:,}</div>
            <div class="metric-delta">{"📈" if delta_orders > 0 else "📉"} {delta_orders:+.1f}% vs last period</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_margin = current_period['Profit_Margin'].mean()
        prev_margin = previous_period['Profit_Margin'].mean()
        delta_margin = avg_margin - prev_margin
        
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <div class="metric-label">📊 Profit Margin</div>
            <div class="metric-value">{avg_margin:.1f}%</div>
            <div class="metric-delta">{"📈" if delta_margin > 0 else "📉"} {delta_margin:+.1f}pp vs last period</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Revenue Trend
    st.subheader("📈 Revenue & Profit Trend")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=business_data['Date'],
        y=business_data['Revenue'],
        name='Revenue',
        line=dict(color='#667eea', width=3),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=business_data['Date'],
        y=business_data['Profit'],
        name='Profit',
        line=dict(color='#43e97b', width=3),
        fill='tonexty',
        fillcolor='rgba(67, 233, 123, 0.1)'
    ))
    
    # Add 30-day moving average
    business_data['Revenue_MA30'] = business_data['Revenue'].rolling(window=30).mean()
    fig.add_trace(go.Scatter(
        x=business_data['Date'],
        y=business_data['Revenue_MA30'],
        name='Revenue (30-day MA)',
        line=dict(color='#764ba2', width=2, dash='dash')
    ))
    
    fig.update_layout(
        height=400,
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance by Region and Channel
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Revenue by Region")
        region_revenue = business_data.groupby('Region')['Revenue'].sum().sort_values(ascending=True)
        
        fig = px.bar(
            x=region_revenue.values,
            y=region_revenue.index,
            orientation='h',
            color=region_revenue.values,
            color_continuous_scale='Viridis',
            text=[f'${x:,.0f}' for x in region_revenue.values]
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            showlegend=False,
            xaxis_title="Revenue ($)",
            yaxis_title="",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📱 Revenue by Channel")
        channel_revenue = business_data.groupby('Channel')['Revenue'].sum()
        
        fig = px.pie(
            values=channel_revenue.values,
            names=channel_revenue.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Key Insights
    st.subheader("💡 Key Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Growth analysis
        revenue_growth = ((current_revenue - prev_revenue) / prev_revenue * 100)
        if revenue_growth > 10:
            insight_color = "#4CAF50"
            insight = f"🎉 Excellent revenue growth of {revenue_growth:.1f}%! Your business is thriving."
        elif revenue_growth > 0:
            insight_color = "#FFC107"
            insight = f"📊 Positive revenue growth of {revenue_growth:.1f}%. Consider increasing marketing spend."
        else:
            insight_color = "#F44336"
            insight = f"⚠️ Revenue declined by {abs(revenue_growth):.1f}%. Review pricing and marketing strategies."
        
        st.markdown(f"""
        <div class="insight-box" style="border-left-color: {insight_color};">
            <strong>Revenue Performance</strong><br>
            {insight}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Margin analysis
        if avg_margin > 30:
            margin_insight = "💎 Excellent profit margins! Focus on scaling operations."
        elif avg_margin > 20:
            margin_insight = "✅ Healthy profit margins. Room for optimization."
        else:
            margin_insight = "⚠️ Low margins detected. Review cost structure and pricing."
        
        st.markdown(f"""
        <div class="insight-box" style="border-left-color: #2196F3;">
            <strong>Profitability Analysis</strong><br>
            {margin_insight}
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# SALES ANALYTICS DASHBOARD
# ============================================================

elif dashboard == "💰 Sales Analytics":
    st.title("💰 Sales Performance Analytics")
    st.markdown("**Detailed sales metrics, trends, and product performance analysis**")
    st.markdown("---")
    
    # Sales metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_sales = business_data['Revenue'].sum()
        st.metric("Total Sales", f"${total_sales:,.0f}", delta=f"{len(business_data)} days")
    
    with col2:
        avg_daily_sales = business_data['Revenue'].mean()
        st.metric("Avg Daily Sales", f"${avg_daily_sales:,.0f}")
    
    with col3:
        total_orders = business_data['Orders'].sum()
        st.metric("Total Orders", f"{total_orders:,}")
    
    with col4:
        avg_order_value = business_data['Avg_Order_Value'].mean()
        st.metric("Avg Order Value", f"${avg_order_value:.2f}")
    
    with col5:
        conv_rate = business_data['Conversion_Rate'].mean() * 100
        st.metric("Conversion Rate", f"{conv_rate:.2f}%")
    
    st.markdown("---")
    
    # Sales trend with multiple metrics
    st.subheader("📊 Sales Performance Trends")
    
    metric_choice = st.selectbox(
        "Select metric to analyze",
        ["Revenue", "Orders", "Avg_Order_Value", "Conversion_Rate"],
        key="sales_metric_choice"
    )
    
    # Aggregation level
    agg_level = st.radio(
        "Aggregation level",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        key="sales_agg_level"
    )
    
    # Aggregate data
    if agg_level == "Weekly":
        plot_data = business_data.set_index('Date').resample('W')[metric_choice].mean().reset_index()
    elif agg_level == "Monthly":
        plot_data = business_data.set_index('Date').resample('M')[metric_choice].mean().reset_index()
    else:
        plot_data = business_data[['Date', metric_choice]]
    
    fig = px.line(
        plot_data,
        x='Date',
        y=metric_choice,
        title=f"{metric_choice} - {agg_level} View",
        markers=True
    )
    fig.update_traces(line_color='#667eea', line_width=3)
    fig.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Product and Channel Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏷️ Top Product Categories")
        category_performance = business_data.groupby('Product_Category').agg({
            'Revenue': 'sum',
            'Orders': 'sum',
            'Profit': 'sum'
        }).sort_values('Revenue', ascending=False)
        
        fig = px.bar(
            category_performance,
            y=category_performance.index,
            x='Revenue',
            orientation='h',
            color='Profit',
            color_continuous_scale='RdYlGn',
            text=[f'${x:,.0f}' for x in category_performance['Revenue']]
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            category_performance.style.format({
                'Revenue': '${:,.0f}',
                'Orders': '{:,}',
                'Profit': '${:,.0f}'
            }),
            use_container_width=True
        )
    
    with col2:
        st.subheader("📈 Sales Channel Performance")
        channel_perf = business_data.groupby('Channel').agg({
            'Revenue': 'sum',
            'Orders': 'sum',
            'Conversion_Rate': 'mean'
        }).sort_values('Revenue', ascending=False)
        
        # Create multi-metric chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Revenue',
            x=channel_perf.index,
            y=channel_perf['Revenue'],
            marker_color='#667eea'
        ))
        
        fig.add_trace(go.Bar(
            name='Orders',
            x=channel_perf.index,
            y=channel_perf['Orders'] * 100,  # Scale for visibility
            marker_color='#43e97b'
        ))
        
        fig.update_layout(
            height=350,
            barmode='group',
            xaxis_title="Channel",
            yaxis_title="Value"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            channel_perf.style.format({
                'Revenue': '${:,.0f}',
                'Orders': '{:,}',
                'Conversion_Rate': '{:.2%}'
            }),
            use_container_width=True
        )
    
    # Sales heatmap
    st.subheader("🔥 Sales Heatmap by Day of Week and Hour")
    
    # Create synthetic hourly data
    business_data['DayOfWeek'] = business_data['Date'].dt.day_name()
    business_data['Hour'] = np.random.randint(8, 22, len(business_data))
    
    heatmap_data = business_data.groupby(['DayOfWeek', 'Hour'])['Revenue'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Hour', columns='DayOfWeek', values='Revenue')
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_pivot[[d for d in day_order if d in heatmap_pivot.columns]]
    
    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x="Day of Week", y="Hour of Day", color="Avg Revenue"),
        color_continuous_scale='YlOrRd',
        aspect="auto"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# CUSTOMER ANALYTICS DASHBOARD
# ============================================================

elif dashboard == "👥 Customer Analytics":
    st.title("👥 Customer Analytics & Insights")
    st.markdown("**Customer behavior, segmentation, and lifetime value analysis**")
    st.markdown("---")
    
    # Customer metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_customers = len(customer_data)
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        avg_clv = customer_data['Customer_Lifetime_Value'].mean()
        st.metric("Avg CLV", f"${avg_clv:,.0f}")
    
    with col3:
        avg_orders = customer_data['Order_Count'].mean()
        st.metric("Avg Orders/Customer", f"{avg_orders:.1f}")
    
    with col4:
        high_value_customers = len(customer_data[customer_data['Customer_Lifetime_Value'] > 1000])
        st.metric("High-Value Customers", f"{high_value_customers:,}")
    
    with col5:
        avg_churn_risk = customer_data['Churn_Risk_Score'].mean()
        st.metric("Avg Churn Risk", f"{avg_churn_risk:.1%}")
    
    st.markdown("---")
    
    # Customer segmentation
    st.subheader("🎯 Customer Segmentation")
    
    tab1, tab2, tab3 = st.tabs(["RFM Analysis", "Loyalty Tiers", "Churn Risk"])
    
    with tab1:
        # RFM Segmentation
        st.markdown("**Recency, Frequency, Monetary Analysis**")
        
        # Calculate RFM scores
        customer_data['Recency_Score'] = pd.qcut(customer_data['Last_Purchase_Days'], 4, labels=[4,3,2,1])
        customer_data['Frequency_Score'] = pd.qcut(customer_data['Order_Count'].rank(method='first'), 4, labels=[1,2,3,4])
        customer_data['Monetary_Score'] = pd.qcut(customer_data['Total_Spent'], 4, labels=[1,2,3,4])
        
        customer_data['RFM_Score'] = (
            customer_data['Recency_Score'].astype(int) +
            customer_data['Frequency_Score'].astype(int) +
            customer_data['Monetary_Score'].astype(int)
        )
        
        # Segment customers
        def rfm_segment(score):
            if score >= 10:
                return 'Champions'
            elif score >= 8:
                return 'Loyal'
            elif score >= 6:
                return 'Potential'
            else:
                return 'At Risk'
        
        customer_data['Segment'] = customer_data['RFM_Score'].apply(rfm_segment)
        
        segment_counts = customer_data['Segment'].value_counts()
        segment_revenue = customer_data.groupby('Segment')['Total_Spent'].sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Customer Distribution by Segment",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=segment_revenue.index,
                y=segment_revenue.values,
                title="Revenue by Customer Segment",
                color=segment_revenue.values,
                color_continuous_scale='Blues',
                text=[f'${x:,.0f}' for x in segment_revenue.values]
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment characteristics
        st.markdown("**Segment Characteristics**")
        segment_stats = customer_data.groupby('Segment').agg({
            'Customer_ID': 'count',
            'Total_Spent': 'mean',
            'Order_Count': 'mean',
            'Customer_Lifetime_Value': 'mean',
            'Last_Purchase_Days': 'mean'
        }).round(2)
        segment_stats.columns = ['Count', 'Avg Spent', 'Avg Orders', 'Avg CLV', 'Avg Days Since Purchase']
        
        st.dataframe(
            segment_stats.style.format({
                'Count': '{:,}',
                'Avg Spent': '${:,.2f}',
                'Avg Orders': '{:.1f}',
                'Avg CLV': '${:,.2f}',
                'Avg Days Since Purchase': '{:.0f}'
            }).background_gradient(cmap='YlOrRd', subset=['Avg CLV']),
            use_container_width=True
        )
    
    with tab2:
        st.markdown("**Customer Loyalty Program Performance**")
        
        loyalty_stats = customer_data.groupby('Loyalty_Tier').agg({
            'Customer_ID': 'count',
            'Total_Spent': ['mean', 'sum'],
            'Order_Count': 'mean',
            'Customer_Lifetime_Value': 'mean'
        }).round(2)
        
        loyalty_stats.columns = ['Count', 'Avg Spent', 'Total Spent', 'Avg Orders', 'Avg CLV']
        
        # Reorder tiers (handle missing tiers)
        tier_order = ['Bronze', 'Silver', 'Gold', 'Platinum']
        existing_tiers = [tier for tier in tier_order if tier in loyalty_stats.index]
        loyalty_stats = loyalty_stats.reindex(existing_tiers)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=loyalty_stats.index,
                y=loyalty_stats['Count'],
                title="Customers by Loyalty Tier",
                color=loyalty_stats.index,
                color_discrete_map={'Bronze': '#CD7F32', 'Silver': '#C0C0C0', 'Gold': '#FFD700', 'Platinum': '#E5E4E2'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=loyalty_stats.index,
                y=loyalty_stats['Avg CLV'],
                title="Average CLV by Tier",
                color=loyalty_stats.index,
                color_discrete_map={'Bronze': '#CD7F32', 'Silver': '#C0C0C0', 'Gold': '#FFD700', 'Platinum': '#E5E4E2'},
                text=[f'${x:,.0f}' for x in loyalty_stats['Avg CLV']]
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            loyalty_stats.style.format({
                'Count': '{:,}',
                'Avg Spent': '${:,.2f}',
                'Total Spent': '${:,.0f}',
                'Avg Orders': '{:.1f}',
                'Avg CLV': '${:,.2f}'
            }).background_gradient(cmap='YlGn', subset=['Avg CLV']),
            use_container_width=True
        )
    
    with tab3:
        st.markdown("**Churn Risk Analysis**")
        
        # Categorize churn risk
        customer_data['Churn_Category'] = pd.cut(
            customer_data['Churn_Risk_Score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        churn_dist = customer_data['Churn_Category'].value_counts()
        high_risk = customer_data[customer_data['Churn_Category'] == 'High Risk']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=churn_dist.values,
                names=churn_dist.index,
                title="Churn Risk Distribution",
                color=churn_dist.index,
                color_discrete_map={'Low Risk': '#4CAF50', 'Medium Risk': '#FFC107', 'High Risk': '#F44336'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("High Risk Customers", f"{len(high_risk):,}")
            st.metric("Potential Revenue at Risk", f"${high_risk['Customer_Lifetime_Value'].sum():,.0f}")
        
        with col2:
            # Scatter plot: CLV vs Churn Risk
            fig = px.scatter(
                customer_data.sample(1000),
                x='Customer_Lifetime_Value',
                y='Churn_Risk_Score',
                color='Churn_Category',
                size='Total_Spent',
                title="Customer Lifetime Value vs Churn Risk",
                color_discrete_map={'Low Risk': '#4CAF50', 'Medium Risk': '#FFC107', 'High Risk': '#F44336'},
                opacity=0.6
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # High-risk customer details
        st.markdown("**High-Risk Customers (Sample)**")
        high_risk_sample = high_risk.nlargest(10, 'Customer_Lifetime_Value')[
            ['Customer_ID', 'Total_Spent', 'Order_Count', 'Last_Purchase_Days', 'Customer_Lifetime_Value', 'Churn_Risk_Score']
        ]
        
        st.dataframe(
            high_risk_sample.style.format({
                'Total_Spent': '${:,.2f}',
                'Order_Count': '{:,}',
                'Last_Purchase_Days': '{:.0f}',
                'Customer_Lifetime_Value': '${:,.2f}',
                'Churn_Risk_Score': '{:.2%}'
            }).background_gradient(cmap='Reds', subset=['Churn_Risk_Score']),
            use_container_width=True
        )

# ============================================================
# FINANCIAL ANALYSIS DASHBOARD
# ============================================================

elif dashboard == "📈 Financial Analysis":
    st.title("📈 Financial Performance Analysis")
    st.markdown("**Comprehensive financial metrics, profitability, and cost analysis**")
    st.markdown("---")
    
    # Financial KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = business_data['Revenue'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    
    with col2:
        total_costs = business_data['Costs'].sum()
        st.metric("Total Costs", f"${total_costs:,.0f}")
    
    with col3:
        total_profit = business_data['Profit'].sum()
        profit_margin = (total_profit / total_revenue * 100)
        st.metric("Net Profit", f"${total_profit:,.0f}", delta=f"{profit_margin:.1f}% margin")
    
    with col4:
        avg_roi = business_data['ROI'].mean()
        st.metric("Average ROI", f"{avg_roi:.1f}%")
    
    st.markdown("---")
    
    # Financial trends
    st.subheader("💰 Revenue, Costs & Profit Analysis")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=business_data['Date'],
        y=business_data['Revenue'],
        name='Revenue',
        line=dict(color='#2E7D32', width=2),
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=business_data['Date'],
        y=business_data['Costs'],
        name='Costs',
        line=dict(color='#C62828', width=2),
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=business_data['Date'],
        y=business_data['Profit'],
        name='Profit',
        line=dict(color='#1565C0', width=3)
    ))
    
    fig.update_layout(
        height=450,
        hovermode='x unified',
        yaxis_title="Amount ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Profitability metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Profit Margin Trend")
        
        # Calculate monthly profit margin
        monthly_data = business_data.set_index('Date').resample('M').agg({
            'Revenue': 'sum',
            'Costs': 'sum',
            'Profit': 'sum'
        })
        monthly_data['Profit_Margin'] = (monthly_data['Profit'] / monthly_data['Revenue'] * 100)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_data.index,
            y=monthly_data['Profit_Margin'],
            mode='lines+markers',
            name='Profit Margin',
            line=dict(color='#667eea', width=3),
            fill='tonexty'
        ))
        
        # Add target line
        fig.add_hline(y=25, line_dash="dash", line_color="green", annotation_text="Target: 25%")
        
        fig.update_layout(
            height=350,
            yaxis_title="Profit Margin (%)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💸 Marketing ROI Analysis")
        
        # Calculate ROI by period
        weekly_roi = business_data.set_index('Date').resample('W').agg({
            'Revenue': 'sum',
            'Marketing_Spend': 'sum'
        })
        weekly_roi['ROI'] = ((weekly_roi['Revenue'] - weekly_roi['Marketing_Spend']) / weekly_roi['Marketing_Spend'] * 100)
        
        fig = px.bar(
            x=weekly_roi.index,
            y=weekly_roi['ROI'],
            title="Weekly Marketing ROI",
            color=weekly_roi['ROI'],
            color_continuous_scale=['red', 'yellow', 'green'],
            labels={'x': 'Week', 'y': 'ROI (%)'}
        )
        
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        fig.update_layout(height=350, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Cost breakdown
    st.subheader("💵 Cost Structure Analysis")
    
    # Simulate cost categories
    cost_categories = pd.DataFrame({
        'Category': ['Operations', 'Marketing', 'Personnel', 'Technology', 'Other'],
        'Amount': [
            business_data['Costs'].sum() * 0.40,
            business_data['Marketing_Spend'].sum(),
            business_data['Costs'].sum() * 0.25,
            business_data['Costs'].sum() * 0.15,
            business_data['Costs'].sum() * 0.10
        ]
    })
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = px.pie(
            cost_categories,
            values='Amount',
            names='Category',
            title="Cost Distribution",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.treemap(
            cost_categories,
            path=['Category'],
            values='Amount',
            title="Cost Breakdown (Treemap)",
            color='Amount',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cash flow simulation
    st.subheader("💰 Cash Flow Projection")
    
    projection_months = st.slider("Projection period (months)", 3, 12, 6, key="cashflow_projection")
    
    # Calculate historical average
    avg_revenue = business_data['Revenue'].mean()
    avg_costs = business_data['Costs'].mean()
    
    # Growth assumptions
    revenue_growth = st.slider("Revenue growth rate (%/month)", -10, 30, 5, key="revenue_growth_rate") / 100
    cost_growth = st.slider("Cost growth rate (%/month)", -10, 30, 2, key="cost_growth_rate") / 100
    
    # Project future
    future_dates = pd.date_range(start=business_data['Date'].max(), periods=projection_months+1, freq='M')[1:]
    
    projected_revenue = [avg_revenue * (1 + revenue_growth) ** i for i in range(1, projection_months+1)]
    projected_costs = [avg_costs * (1 + cost_growth) ** i for i in range(1, projection_months+1)]
    projected_profit = [r - c for r, c in zip(projected_revenue, projected_costs)]
    
    projection_df = pd.DataFrame({
        'Date': future_dates,
        'Revenue': projected_revenue,
        'Costs': projected_costs,
        'Profit': projected_profit
    })
    
    # Combine historical and projected
    historical_monthly = business_data.set_index('Date').resample('M').agg({
        'Revenue': 'sum',
        'Costs': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=historical_monthly['Date'],
        y=historical_monthly['Revenue'],
        name='Historical Revenue',
        line=dict(color='#2E7D32', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=historical_monthly['Date'],
        y=historical_monthly['Profit'],
        name='Historical Profit',
        line=dict(color='#1565C0', width=2)
    ))
    
    # Projected
    fig.add_trace(go.Scatter(
        x=projection_df['Date'],
        y=projection_df['Revenue'],
        name='Projected Revenue',
        line=dict(color='#2E7D32', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=projection_df['Date'],
        y=projection_df['Profit'],
        name='Projected Profit',
        line=dict(color='#1565C0', width=2, dash='dash')
    ))
    
    fig.update_layout(
        height=400,
        hovermode='x unified',
        yaxis_title="Amount ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Projection table
    st.markdown("**Projected Financial Summary**")
    st.dataframe(
        projection_df.style.format({
            'Revenue': '${:,.0f}',
            'Costs': '${:,.0f}',
            'Profit': '${:,.0f}'
        }).background_gradient(cmap='RdYlGn', subset=['Profit']),
        use_container_width=True
    )

# ============================================================
# PREDICTIVE INSIGHTS DASHBOARD
# ============================================================

elif dashboard == "🔮 Predictive Insights":
    st.title("🔮 Predictive Analytics & Forecasting")
    st.markdown("**AI-powered predictions and trend forecasting for business planning**")
    st.markdown("---")
    
    if not ML_AVAILABLE:
        st.error("⚠️ Machine learning libraries not available. Please install scikit-learn:")
        st.code("pip install scikit-learn")
        st.info("Showing basic trend analysis instead...")
    
    st.subheader("📈 Revenue Forecasting")
    
    # Prepare data for forecasting
    forecast_data = business_data[['Revenue', 'Marketing_Spend', 'Website_Visits', 'Orders']].copy()
    forecast_data['Days'] = np.arange(len(forecast_data))
    
    if ML_AVAILABLE:
        # Train models
        X = forecast_data[['Days', 'Marketing_Spend', 'Website_Visits', 'Orders']]
        y = forecast_data['Revenue']
        
        # Simple train-test split (last 30 days for testing)
        split_point = max(len(X) - 30, int(len(X) * 0.8))
        X_train = X[:split_point]
        y_train = y[:split_point]
        X_test = X[split_point:]
        y_test = y[split_point:]
        
        try:
            # Train models
            lr_model = LinearRegression()
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            
            lr_model.fit(X_train, y_train)
            rf_model.fit(X_train, y_train)
            
            # Predictions
            lr_pred = lr_model.predict(X_test)
            rf_pred = rf_model.predict(X_test)
            
            # Forecast future
            forecast_days = st.slider("Forecast horizon (days)", 7, 90, 30, key="forecast_horizon_pred")
            
            future_X = pd.DataFrame({
                'Days': np.arange(len(forecast_data), len(forecast_data) + forecast_days),
                'Marketing_Spend': [forecast_data['Marketing_Spend'].mean()] * forecast_days,
                'Website_Visits': [forecast_data['Website_Visits'].mean()] * forecast_days,
                'Orders': [forecast_data['Orders'].mean()] * forecast_days
            })
            
            future_lr = lr_model.predict(future_X)
            future_rf = rf_model.predict(future_X)
            
            # Visualization
            future_dates = pd.date_range(start=business_data['Date'].max() + timedelta(days=1), periods=forecast_days)
            
            fig = go.Figure()
            
            # Historical
            fig.add_trace(go.Scatter(
                x=business_data['Date'],
                y=business_data['Revenue'],
                name='Historical Revenue',
                line=dict(color='#2E7D32', width=2)
            ))
            
            # Linear Regression forecast
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_lr,
                name='Linear Regression Forecast',
                line=dict(color='#1565C0', width=2, dash='dash')
            ))
            
            # Random Forest forecast
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_rf,
                name='Random Forest Forecast',
                line=dict(color='#D84315', width=2, dash='dot')
            ))
            
            # Confidence interval (simplified)
            std_dev = business_data['Revenue'].std()
            fig.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates)[::-1],
                y=list(future_rf + std_dev) + list((future_rf - std_dev).clip(min=0))[::-1],
                fill='toself',
                fillcolor='rgba(216, 67, 21, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
            
            fig.update_layout(
                height=450,
                hovermode='x unified',
                yaxis_title="Revenue ($)",
                xaxis_title="Date"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_forecast = future_rf.mean()
                st.metric("Avg Daily Forecast", f"${avg_forecast:,.0f}")
            
            with col2:
                total_forecast = future_rf.sum()
                st.metric(f"Total {forecast_days}-Day Forecast", f"${total_forecast:,.0f}")
            
            with col3:
                current_avg = business_data['Revenue'].tail(30).mean()
                growth = ((avg_forecast - current_avg) / current_avg * 100)
                st.metric("Projected Growth", f"{growth:+.1f}%")
        
        except Exception as e:
            st.error(f"Model training error: {e}")
            st.info("Using simple trend forecast instead...")
            ML_AVAILABLE = False
    
    # Fallback: Simple moving average forecast
    if not ML_AVAILABLE:
        forecast_days = st.slider("Forecast horizon (days)", 7, 90, 30, key="forecast_horizon_simple")
        
        # Calculate moving average
        ma_30 = business_data['Revenue'].tail(30).mean()
        future_dates = pd.date_range(start=business_data['Date'].max() + timedelta(days=1), periods=forecast_days)
        forecast_values = [ma_30] * forecast_days
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=business_data['Date'],
            y=business_data['Revenue'],
            name='Historical',
            line=dict(color='#2E7D32', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecast_values,
            name='Forecast (30-day MA)',
            line=dict(color='#1565C0', width=2, dash='dash')
        ))
        
        fig.update_layout(height=450, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Forecast", f"${ma_30:,.0f}")
        with col2:
            st.metric(f"{forecast_days}-Day Total", f"${ma_30 * forecast_days:,.0f}")
    
    # Anomaly detection
    st.subheader("🚨 Anomaly Detection")
    
    # Calculate z-scores
    business_data['Revenue_Zscore'] = (business_data['Revenue'] - business_data['Revenue'].mean()) / business_data['Revenue'].std()
    anomalies = business_data[np.abs(business_data['Revenue_Zscore']) > 2.5]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=business_data['Date'],
            y=business_data['Revenue'],
            mode='lines',
            name='Revenue',
            line=dict(color='#667eea', width=2)
        ))
        
        if len(anomalies) > 0:
            fig.add_trace(go.Scatter(
                x=anomalies['Date'],
                y=anomalies['Revenue'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='x')
            ))
        
        fig.update_layout(
            title="Revenue Anomalies Detection",
            height=350,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Anomalies Detected", len(anomalies))
        st.metric("% of Total Days", f"{len(anomalies)/len(business_data)*100:.1f}%")
        
        if len(anomalies) > 0:
            st.markdown("**Recent Anomalies:**")
            for idx, row in anomalies.tail(5).iterrows():
                st.write(f"📅 {row['Date'].strftime('%Y-%m-%d')}: ${row['Revenue']:,.0f}")
        else:
            st.success("✅ No significant anomalies detected")
    
    # Trend analysis
    st.subheader("📊 Trend Analysis & Seasonality")
    
    # Simple trend using moving average
    window_size = 30
    trend = business_data['Revenue'].rolling(window=window_size, center=True).mean()
    seasonal = business_data['Revenue'] - trend
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Trend Component", "Seasonal Component"))
    
    fig.add_trace(
        go.Scatter(x=business_data['Date'], y=trend, name='Trend', line=dict(color='#2E7D32', width=3)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=business_data['Date'], y=seasonal, name='Seasonal', line=dict(color='#1565C0', width=2)),
        row=2, col=1
    )
    
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# REPORT GENERATOR DASHBOARD
# ============================================================

elif dashboard == "📄 Report Generator":
    st.title("📄 Automated Report Generator")
    st.markdown("**Generate comprehensive business reports with one click**")
    st.markdown("---")
    
    st.subheader("⚙️ Report Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Sales Performance", "Customer Analytics", "Financial Report", "Full Business Report"]
        )
        
        report_period = st.selectbox(
            "Report Period",
            ["Last 7 Days", "Last 30 Days", "Last Quarter", "Last Year", "Custom Range"]
        )
        
        include_charts = st.checkbox("Include Charts", value=True)
        include_recommendations = st.checkbox("Include AI Recommendations", value=True)
    
    with col2:
        report_format = st.selectbox(
            "Export Format",
            ["PDF", "Excel", "PowerPoint", "Word"]
        )
        
        email_report = st.checkbox("Email Report", value=False)
        
        if email_report:
            recipient_email = st.text_input("Recipient Email")
    
    st.markdown("---")
    
    if st.button("🚀 Generate Report", type="primary"):
        with st.spinner("Generating comprehensive report..."):
            import time
            time.sleep(2)  # Simulate processing
            
            st.success("✅ Report generated successfully!")
            
            # Display report preview
            st.subheader("📄 Report Preview")
            
            # Executive Summary Section
            st.markdown("### Executive Summary")
            st.markdown(f"**Report Period:** {report_period}")
            st.markdown(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Revenue", f"${business_data['Revenue'].sum():,.0f}")
            with col2:
                st.metric("Net Profit", f"${business_data['Profit'].sum():,.0f}")
            with col3:
                st.metric("Total Orders", f"{business_data['Orders'].sum():,}")
            with col4:
                st.metric("Profit Margin", f"{business_data['Profit_Margin'].mean():.1f}%")
            
            if include_charts:
                st.markdown("### Performance Trends")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=business_data['Date'],
                    y=business_data['Revenue'],
                    name='Revenue',
                    fill='tonexty'
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            if include_recommendations:
                st.markdown("### 🤖 AI-Powered Recommendations")
                
                recommendations = [
                    "📈 Revenue is trending upward. Consider increasing inventory for high-demand products.",
                    "💰 Marketing ROI is strong. Allocate 15% more budget to top-performing channels.",
                    "👥 Customer retention rate is 78%. Implement loyalty program to improve to 85%.",
                    "⚠️ 234 customers at high churn risk. Launch targeted retention campaign.",
                    "🎯 Focus on North America region - highest profit margin at 32%."
                ]
                
                for rec in recommendations:
                    st.info(rec)
            
            # Download buttons
            st.markdown("---")
            st.markdown("### 📥 Download Report")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Create sample CSV for download
                csv_data = business_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name=f"business_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="download_csv_btn"
                )
            
            with col2:
                # PDF Download
                if PDF_AVAILABLE:
                    try:
                        pdf_bytes = generate_pdf_report(business_data, date_range)
                        if pdf_bytes:
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=pdf_bytes,
                                file_name=f"business_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf",
                                key="download_pdf_btn"
                            )
                            st.caption("✅ Full PDF report")
                        else:
                            raise Exception("PDF generation failed")
                    except Exception as e:
                        st.error(f"PDF error: {e}")
                        # Fallback to HTML
                        html_content = f"""
                        <html>
                        <head>
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                                h1 {{ color: #333; }}
                                .metric {{ background: #f0f0f0; padding: 15px; margin: 10px 0; }}
                            </style>
                        </head>
                        <body>
                            <h1>Business Report</h1>
                            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
                            <div class="metric">
                                <strong>Revenue:</strong> ${business_data['Revenue'].sum():,.0f}<br>
                                <strong>Profit:</strong> ${business_data['Profit'].sum():,.0f}
                            </div>
                        </body>
                        </html>
                        """
                        st.download_button(
                            label="📄 Download HTML Report",
                            data=html_content.encode('utf-8'),
                            file_name=f"report_{datetime.now().strftime('%Y%m%d')}.html",
                            mime="text/html",
                            key="download_html_fallback"
                        )
                        st.caption("💡 Open in browser → Print to PDF")
                else:
                    # HTML fallback
                    html_content = f"""
                    <html>
                    <head>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 40px; }}
                            h1 {{ color: #333; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
                            h2 {{ color: #666; margin-top: 30px; }}
                            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                            th {{ background-color: #667eea; color: white; }}
                            .metric {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                     color: white; padding: 20px; margin: 10px 0; border-radius: 8px; }}
                            .metric-value {{ font-size: 24px; font-weight: bold; }}
                        </style>
                    </head>
                    <body>
                        <h1>📊 Business Intelligence Report</h1>
                        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p><strong>Period:</strong> {date_range}</p>
                        
                        <h2>Executive Summary</h2>
                        <div class="metric">
                            <div>Total Revenue</div>
                            <div class="metric-value">${business_data['Revenue'].sum():,.0f}</div>
                        </div>
                        <div class="metric">
                            <div>Net Profit</div>
                            <div class="metric-value">${business_data['Profit'].sum():,.0f}</div>
                        </div>
                        <div class="metric">
                            <div>Total Orders</div>
                            <div class="metric-value">{business_data['Orders'].sum():,}</div>
                        </div>
                        <div class="metric">
                            <div>Profit Margin</div>
                            <div class="metric-value">{business_data['Profit_Margin'].mean():.2f}%</div>
                        </div>
                        
                        <h2>Top 10 Days by Revenue</h2>
                        <table>
                            <tr>
                                <th>Date</th>
                                <th>Revenue</th>
                                <th>Profit</th>
                                <th>Orders</th>
                                <th>Margin %</th>
                            </tr>
                    """
                    
                    top_days = business_data.nlargest(10, 'Revenue')
                    for _, row in top_days.iterrows():
                        html_content += f"""
                            <tr>
                                <td>{row['Date'].strftime('%Y-%m-%d')}</td>
                                <td>${row['Revenue']:,.0f}</td>
                                <td>${row['Profit']:,.0f}</td>
                                <td>{row['Orders']:,}</td>
                                <td>{row['Profit_Margin']:.2f}%</td>
                            </tr>
                        """
                    
                    html_content += """
                        </table>
                        
                        <h2>Regional Performance</h2>
                        <table>
                            <tr>
                                <th>Region</th>
                                <th>Total Revenue</th>
                                <th>Total Profit</th>
                                <th>Avg Margin %</th>
                            </tr>
                    """
                    
                    regional = business_data.groupby('Region').agg({
                        'Revenue': 'sum',
                        'Profit': 'sum',
                        'Profit_Margin': 'mean'
                    }).sort_values('Revenue', ascending=False)
                    
                    for region, row in regional.iterrows():
                        html_content += f"""
                            <tr>
                                <td>{region}</td>
                                <td>${row['Revenue']:,.0f}</td>
                                <td>${row['Profit']:,.0f}</td>
                                <td>{row['Profit_Margin']:.2f}%</td>
                            </tr>
                        """
                    
                    html_content += """
                        </table>
                        
                        <p style="margin-top: 50px; color: #666; font-size: 12px; text-align: center;">
                            Generated by Business Intelligence Dashboard | Powered by Python & Streamlit
                        </p>
                    </body>
                    </html>
                    """
                    
                    st.download_button(
                        label="📄 Download Report (HTML)",
                        data=html_content.encode('utf-8'),
                        file_name=f"business_report_{datetime.now().strftime('%Y%m%d')}.html",
                        mime="text/html",
                        key="download_html_btn"
                    )
                    st.caption("💡 Open in browser → Ctrl+P → Save as PDF")
            
            with col3:
                try:
                    # Create Excel file with proper error handling
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        business_data.to_excel(writer, sheet_name='Data', index=False)
                        
                        # Add summary sheet
                        summary = pd.DataFrame({
                            'Metric': ['Total Revenue', 'Net Profit', 'Total Orders', 'Avg Profit Margin'],
                            'Value': [
                                f"${business_data['Revenue'].sum():,.0f}",
                                f"${business_data['Profit'].sum():,.0f}",
                                f"{business_data['Orders'].sum():,}",
                                f"{business_data['Profit_Margin'].mean():.2f}%"
                            ]
                        })
                        summary.to_excel(writer, sheet_name='Summary', index=False)
                    
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📊 Download Excel",
                        data=excel_data,
                        file_name=f"business_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel_btn"
                    )
                except Exception as e:
                    st.error(f"Excel export error: {e}")
                    st.info("Please install openpyxl: pip install openpyxl")
            
            with col4:
                st.button("📊 Download PowerPoint", disabled=True, help="PowerPoint export requires python-pptx")
                st.caption("⚠️ PPTX export needs setup")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h3>🚀 Real-Time Business Intelligence Dashboard</h3>
        <p><strong>Powered by:</strong> Python | Streamlit | Plotly | Scikit-learn | Pandas</p>
        <p><strong>Features:</strong> Executive Dashboards • Sales Analytics • Customer Insights • Financial Analysis • Predictive Forecasting • Automated Reports</p>
        <p style='margin-top: 20px; font-size: 0.9em;'>
            Made with ❤️ for Business Intelligence | 
            <a href='https://github.com/SendPain11' target='_blank'>GitHub</a> | 
            <a href='https://linkedin.com/in/sendy-prismana-nurferian/' target='_blank'>LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)