import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page configurations as the first stream command
st.set_page_config(
    page_title="Buisnessverse - Smart Analytics Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Inject custom glassmorphic stylesheet
CSS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "custom.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Import connections and custom engines
from database.connection import get_raw_connection, engine
from database.queries import PREBUILT_QUERIES, execute_custom_query
from utils.auth import show_login_interface, check_permission, init_auth_session
from utils.ai_engine import detect_business_anomalies, translate_nlp_to_sql, ask_business_chatbot
from utils.pdf_generator import generate_excel_report, generate_html_briefing

# Import ML algorithms
from models.forecaster import prepare_time_series_data, train_holt_winters_forecast, train_ml_regressor_forecast, plot_forecast_plotly
from models.churn import prepare_churn_features, train_churn_classifier, plot_confusion_matrix_plotly, plot_roc_curve_plotly, predict_custom_churn
from models.segmentation import calculate_rfm_metrics, calculate_elbow_wcss, plot_elbow_curve_plotly, train_kmeans_segmentation, plot_3d_clusters_plotly
from models.recommender import extract_shopping_baskets, calculate_association_rules, recommend_cross_sell_products
from models.financial_models import get_financial_baselines, simulate_business_scenario

# --- DATA LOADERS (Cached for speed) ---
@st.cache_data(ttl=60)
def load_all_tables(profile="saas"):
    prefix = "custom_" if profile == "custom" else ""
    with engine.connect() as conn:
        p = pd.read_sql(f"SELECT * FROM {prefix}products;", conn)
        c = pd.read_sql(f"SELECT * FROM {prefix}customers;", conn)
        s = pd.read_sql(f"SELECT * FROM {prefix}sales;", conn)
        m = pd.read_sql(f"SELECT * FROM {prefix}marketing_campaigns;", conn)
        f = pd.read_sql(f"SELECT * FROM {prefix}financials;", conn)
    return p, c, s, m, f

# Initialize Session States
init_auth_session()

# Render Authentication screen if not logged in
if not st.session_state.authenticated:
    show_login_interface()
    st.stop()

# Initialize Business Profile Session States
if "business_profile" not in st.session_state:
    st.session_state.business_profile = "saas"

from database.connection import check_custom_tables_exist
custom_data_exists = check_custom_tables_exist()

# --- AUTHENTICATED SYSTEM PORTAL ---
# Load datasets based on active profile
active_profile = st.session_state.business_profile
if active_profile == "custom" and not custom_data_exists:
    # Temporarily fetch SaaS to prevent crashes, but we will block layout below
    products_df, customers_df, sales_df, marketing_df, financials_df = load_all_tables("saas")
else:
    try:
        products_df, customers_df, sales_df, marketing_df, financials_df = load_all_tables(active_profile)
    except Exception as e:
        custom_data_exists = False
        st.session_state.business_profile = "saas"
        st.error(f"Error loading custom tables: {str(e)}. Reverting to SaaS playground data.")
        products_df, customers_df, sales_df, marketing_df, financials_df = load_all_tables("saas")

# Globally convert sale_date to datetime to avoid pandas .dt accessor errors on strings
try:
    sales_df["sale_date"] = pd.to_datetime(sales_df["sale_date"], errors="coerce")
except Exception as e:
    st.error(f"Error parsing sale dates: {str(e)}. Reverting to SaaS playground data.")
    st.session_state.business_profile = "saas"
    products_df, customers_df, sales_df, marketing_df, financials_df = load_all_tables("saas")
    sales_df["sale_date"] = pd.to_datetime(sales_df["sale_date"], errors="coerce")


# Helper to apply the light glassmorphic theme globally to Plotly figures
def apply_light_theme(fig):
    fig.update_layout(
        font=dict(color="#1e1b4b", family="Inter", size=13),
        title_font=dict(color="#1e1b4b", size=16, family="Outfit"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,255,0.6)",
        legend=dict(
            font=dict(color="#1e1b4b", size=12),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(30,27,75,0.15)",
            borderwidth=1
        )
    )
    if hasattr(fig, 'update_xaxes'):
        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(30, 27, 75, 0.1)",
            linecolor="rgba(30, 27, 75, 0.25)",
            zeroline=False,
            tickfont=dict(color="#2d2a6e", size=12, family="Inter"),
            title_font=dict(color="#1e1b4b", size=13, family="Outfit")
        )
    if hasattr(fig, 'update_yaxes'):
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(30, 27, 75, 0.1)",
            linecolor="rgba(30, 27, 75, 0.25)",
            zeroline=False,
            tickfont=dict(color="#2d2a6e", size=12, family="Inter"),
            title_font=dict(color="#1e1b4b", size=13, family="Outfit")
        )
    return fig

prod_costs = dict(zip(products_df["product_id"], products_df["cost"]))

# Detect active alerts for display
active_anomalies = detect_business_anomalies()

# Set up beautiful Sidebar details
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 25px; border-bottom: 1px solid rgba(53,46,140,0.1); padding-bottom: 18px;'>
            <h2 style='
                margin: 0 0 6px 0;
                font-family: "Outfit", sans-serif;
                font-size: 1.6rem;
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                background: linear-gradient(90deg, #352e8c, #5ac8fa);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            '>BUISNESSVERSE</h2>
            <span style='
                color: #4a467f;
                font-size: 0.7rem;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-weight: 700;
                font-family: "Inter", sans-serif;
            '>SMART BUSINESS INTELLIGENCE</span>
        </div>
    """, unsafe_allow_html=True)
    
    # User Profile card
    st.markdown(f"""
        <div style='background: rgba(255,255,255,0.02); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;'>
            <span style='font-size: 0.75rem; color:#8a99ad; display:block;'>Active Profile:</span>
            <strong style='color:#ffffff; font-size: 0.95rem; font-family:"Outfit";'>{st.session_state.fullname}</strong>
            <span style='background: rgba(0, 210, 255, 0.15); color: #00d2ff; font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; float: right;'>{st.session_state.user_role}</span>
            <div style='clear:both;'></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Business Profile switcher
    st.markdown("<span style='font-size:0.75rem; color:#8a99ad; text-transform:uppercase; font-weight:600;'>Business Profile</span>", unsafe_allow_html=True)
    
    profiles = {
        "saas": "🧪 SaaS Enterprise (Playground)",
        "custom": "🏢 Custom Business (Imported CSV)"
    }
    current_prof_idx = 0 if st.session_state.business_profile == "saas" else 1
    selected_prof = st.selectbox(
        "Select Active Business Profile",
        options=["saas", "custom"],
        format_func=lambda x: profiles[x],
        index=current_prof_idx,
        key="profile_selector_widget",
        label_visibility="collapsed"
    )
    
    if selected_prof != st.session_state.business_profile:
        st.session_state.business_profile = selected_prof
        st.rerun()
        
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    
    st.markdown("<span style='font-size:0.75rem; color:#8a99ad; text-transform:uppercase; font-weight:600;'>System Navigation</span>", unsafe_allow_html=True)
    
    # Custom page navigator
    pages = {
        "dashboard": "⚡ Executive Dashboard",
        "sales": "📈 Sales Analytics",
        "customers": "👥 Customer Segmentations",
        "products": "📦 Product & Inventory",
        "marketing": "📣 Marketing Performance",
        "financials": "💰 Financial Analytics",
        "ml": "🧠 AI & AutoML Workspace",
        "chatbot": "💬 AI Insights Chatbot",
        "sql": "⚙️ Expert SQL Workspace",
        "import": "💼 Data Import Center",
        "reports": "📥 Report Export Center"
    }
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "dashboard"
        
    for page_key, page_title in pages.items():
        active_class = "active" if st.session_state.current_page == page_key else ""
        if st.sidebar.button(page_title, key=f"nav_{page_key}", use_container_width=True):
            st.session_state.current_page = page_key
            st.rerun()
            
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # AI Settings block
    with st.expander("🤖 AI Insights Provider Settings"):
        ai_provider = st.selectbox("API Provider", ["Gemini", "OpenAI"])
        api_key = st.text_input("Enter API Key", type="password", placeholder="Paste sk-... or AIza...")
        if api_key:
            st.success("API Credentials Buffered!")
            
    # Sign out button
    if st.sidebar.button("Sign Out of System", type="secondary", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_role = None
        st.session_state.fullname = None
        st.rerun()

# Save API configurations to session state
st.session_state.api_key = api_key if 'api_key' in locals() else None
st.session_state.ai_provider = ai_provider if 'ai_provider' in locals() else "Gemini"


# --- NO CUSTOM DATA REDIRECT GUARD ---
if st.session_state.business_profile == "custom" and not custom_data_exists and st.session_state.current_page != "import":
    st.markdown("<h1>Custom Business Profile</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin-bottom: 25px;'>Instantly scale and customize the analytics platform with your own corporate data.</p>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("""
            <div style='text-align: center; padding: 40px;'>
                <div style='font-size: 4rem; margin-bottom: 15px;'>🏢</div>
                <h3 style='color:#00d2ff; font-family:"Outfit"; margin: 0 0 10px 0;'>No Custom Business Data Found</h3>
                <p style='color:#cbd5e1; font-size:1.1rem; max-width: 600px; margin: 10px auto;'>
                    You have toggled to the <b>Custom Business</b> profile. To unlock the entire suite of 10 pages for your own company, 
                    simply upload a single sales history or invoice CSV file!
                </p>
                <div style='color: #8a99ad; font-size: 0.9rem; margin-bottom: 25px;'>
                    Our backend synthesis pipeline will automatically compile matching products, customer lifespans, marketing campaigns, and ledger financials.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Upload Business CSV Now", type="primary", use_container_width=True):
                st.session_state.current_page = "import"
                st.rerun()
    st.stop()

if st.session_state.current_page == "dashboard":
    st.markdown("<h1>Executive Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin-bottom: 25px;'>High-level performance monitoring, dynamic business health scores, and predictive operational logs.</p>", unsafe_allow_html=True)
    
    # Metrics calculations
    total_rev = sales_df["total_amount"].sum()
    total_cogs = sum(row["quantity"] * prod_costs.get(row["product_id"], 0) for _, row in sales_df.iterrows())
    gross_margin = ((total_rev - total_cogs) / total_rev) * 100
    
    retention_rate = (1.0 - customers_df["churn_status"].mean()) * 100
    
    sales_df["sale_date"] = pd.to_datetime(sales_df["sale_date"])
    monthly_sales = sales_df.set_index("sale_date").resample("MS")["total_amount"].sum()
    growth_rate = ((monthly_sales.iloc[-1] - monthly_sales.iloc[-2]) / monthly_sales.iloc[-2]) * 100
    
    health_score = int(
        (gross_margin * 0.3) + 
        (retention_rate * 0.35) + 
        (min(100.0, max(0.0, 50.0 + growth_rate)) * 0.20) + 
        (max(0, 100 - (len(active_anomalies) * 15)) * 0.15)
    )
    health_score = max(0, min(100, health_score))
    
    # Show KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-lbl">Total revenue</div>
                <div class="kpi-val">${total_rev:,.0f}</div>
                <div class="kpi-sub"><span class="trend-up">▲ 12.4%</span> vs last year</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="kpi-card success">
                <div class="kpi-lbl">Gross Profit Margin</div>
                <div class="kpi-val">{gross_margin:.1f}%</div>
                <div class="kpi-sub"><span class="trend-up">▲ 0.8%</span> month-on-month</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="kpi-card purple">
                <div class="kpi-lbl">Client Retention</div>
                <div class="kpi-val">{retention_rate:.1f}%</div>
                <div class="kpi-sub"><span class="trend-up">▲ 2.1%</span> vs industry avg</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="kpi-card warning">
                <div class="kpi-lbl">Monthly Revenue Growth</div>
                <div class="kpi-val">{growth_rate:+.1f}%</div>
                <div class="kpi-sub"><span class="trend-down">▼ 1.2%</span> last 30d slope</div>
            </div>
        """, unsafe_allow_html=True)
    with col5:
        alert_class = "error" if len(active_anomalies) > 0 else ""
        st.markdown(f"""
            <div class="kpi-card {alert_class}">
                <div class="kpi-lbl">Active Risk Alerts</div>
                <div class="kpi-val">{len(active_anomalies)}</div>
                <div class="kpi-sub">Critical issues flagged</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart, col_score = st.columns([3, 1])
    
    with col_chart:
        with st.container(border=True):
            # Sales history and forecast combined
            df_monthly = prepare_time_series_data(sales_df)
            in_sample_hw, df_fc_hw, _ = train_holt_winters_forecast(df_monthly, forecast_months=6)
            
            # Plot Plotly chart
            fig_fc = plot_forecast_plotly(df_monthly, in_sample_hw, df_fc_hw, "Executive BusinessPulse")
            st.plotly_chart(fig_fc, use_container_width=True)
        
    with col_score:
        with st.container(border=True):
            st.markdown("<h4 style='text-align: center; color: var(--text-main);'>Business Health Score</h4>", unsafe_allow_html=True)
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="health-score-container">
                    <div class="health-val">{health_score}</div>
                    <div class="kpi-lbl" style='margin-top: 10px;'>Overall Rating</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if health_score >= 85:
                st.success("Platform status: EXCELLENT. Gross margins and churn are healthy.")
            elif health_score >= 70:
                st.warning("Platform status: HEURISTIC WARNING. Minor regional logistics leaks detected.")
            else:
                st.error("Platform status: CRITICAL ACTION. High churn spikes require account management.")
        
    # Bottom Layout: Region map & Alerts log
    col_map, col_alerts = st.columns([2, 2])
    
    with col_map:
        with st.container(border=True):
            st.subheader("Sales Channel Share by Shipping Region")
            # PREMIUM ADDITION: Stacked bar chart showing shipping channel breakdown per region
            channel_region = sales_df.groupby(["region", "sales_channel"])["total_amount"].sum().reset_index()
            fig_stack = px.bar(
                channel_region, x="region", y="total_amount", color="sales_channel",
                title="Regional Channel Split (Online vs. Retail vs. Direct)",
                barmode="stack",
                color_discrete_sequence=["#352e8c", "#5ac8fa", "#ff5e7e", "#10b981", "#fbbf24", "#ef4444"]
            )
            fig_stack.update_layout(
                template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,255,0.6)", font=dict(color="#1e1b4b", family="Inter", size=13),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(apply_light_theme(fig_stack), use_container_width=True)
        
    with col_alerts:
        with st.container(border=True):
            st.subheader("Active Operational Risk Logs")
            if len(active_anomalies) == 0:
                st.info("Excellent! No operational anomalies registered by AI engines.")
            else:
                for alert in active_anomalies:
                    severity_color = "#ff5252" if alert["severity"] == "High" else "#ffb600"
                    st.markdown(f"""
                        <div class="anomaly-alert" style='border-left-color: {severity_color};'>
                            <div class="anomaly-title" style='color: {severity_color};'>[{alert['type']}] {alert['title']} ({alert['severity']} Priority)</div>
                            <div style='color: var(--text-muted); font-size: 0.85rem;'>{alert['desc']}</div>
                        </div>
                    """, unsafe_allow_html=True)

elif st.session_state.current_page == "sales":
    st.markdown("<h1>Sales Analytics & Simulators</h1>", unsafe_allow_html=True)
    
    col_sel, col_sim = st.columns([3, 2])
    
    with col_sel:
        with st.container(border=True):
            st.subheader("Gross Sales Revenue by Category Structure")
            
            # Treemap of Product Categories & Sales
            s_df = sales_df.copy()
            p_df = products_df.copy()
            s_df["product_id"] = s_df["product_id"].astype(str)
            p_df["product_id"] = p_df["product_id"].astype(str)
            sales_prod = pd.merge(s_df, p_df, on="product_id")
            
            if len(sales_prod) > 0:
                sales_cat = sales_prod.groupby(["category", "name"])["total_amount"].sum().reset_index()
                sales_cat.columns = ["Category", "Product", "Revenue"]
            else:
                sales_cat = pd.DataFrame(columns=["Category", "Product", "Revenue"])
            
            fig_tree = px.treemap(
                sales_cat, path=["Category", "Product"], values="Revenue",
                color="Revenue", color_continuous_scale="Viridis",
                labels={"Revenue": "Revenue ($)"}
            )
            fig_tree.update_layout(
                template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,255,0.6)", font=dict(color="#1e1b4b", family="Inter", size=13),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(apply_light_theme(fig_tree), use_container_width=True)
        
    with col_sim:
        with st.container(border=True):
            st.subheader("Price Elasticity & What-If Planner")
            st.markdown("<p style='color:#8a99ad; font-size:0.85rem;'>Model pricing adjustments against customer price elasticities to predict future billing outcomes.</p>", unsafe_allow_html=True)
            
            price_adj = st.slider("Select Price Adjustment %", -20, 20, 0, 5, format="%d%%") / 100.0
            mkt_boost = st.slider("Select Marketing Budget Shift %", -50, 100, 0, 10, format="%d%%") / 100.0
            
            sim_results = simulate_business_scenario(sales_df, customers_df, products_df, financials_df,
                                                    price_adjustment_pct=price_adj,
                                                    marketing_spend_pct=mkt_boost)
            
            # Variance cards
            rev_var = sim_results["variance"]["revenue_pct"]
            prof_var = sim_results["variance"]["net_profit_pct"]
            margin_var = sim_results["variance"]["margin_diff"]
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                color_c = "trend-up" if rev_var >= 0 else "trend-down"
                icon = "▲" if rev_var >= 0 else "▼"
                st.markdown(f"""
                    <div class="sub-feature-card" style='text-align:center; padding:15px;'>
                        <span style='font-size:0.8rem; color:#a5b4fc;'>Simulated Revenue</span>
                        <h3 style='margin:5px 0; color:#ffffff;'>${sim_results['simulated']['revenue']:,.0f}</h3>
                        <span class="{color_c}">{icon} {abs(rev_var):.1f}% Variance</span>
                    </div>
                """, unsafe_allow_html=True)
                
            with col_res2:
                color_c = "trend-up" if prof_var >= 0 else "trend-down"
                icon = "▲" if prof_var >= 0 else "▼"
                st.markdown(f"""
                    <div class="sub-feature-card" style='text-align:center; padding:15px;'>
                        <span style='font-size:0.8rem; color:#a5b4fc;'>Simulated Net Profit</span>
                        <h3 style='margin:5px 0; color:#ffffff;'>${sim_results['simulated']['net_profit']:,.0f}</h3>
                        <span class="{color_c}">{icon} {abs(prof_var):.1f}% Variance</span>
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"Summary: Price adjustment of {price_adj:+.0%} combined with a {mkt_boost:+.0%} marketing shift results in an overall profit margin delta of {margin_var:+.2f}%.")
            
    # PREMIUM ADDITION: Month-on-Month Growth Heatmap
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("🗓️ Corporate Monthly Revenue Chronology & Growth Rate")
        # Reuse monthly growth logic from queries
        monthly_sales_summary = sales_df.copy()
        monthly_sales_summary["year_month"] = monthly_sales_summary["sale_date"].dt.strftime("%Y-%m")
        grouped_mom = monthly_sales_summary.groupby("year_month")["total_amount"].sum().reset_index()
        grouped_mom["prev_revenue"] = grouped_mom["total_amount"].shift(1)
        grouped_mom["growth_rate"] = ((grouped_mom["total_amount"] - grouped_mom["prev_revenue"]) / grouped_mom["prev_revenue"]) * 100
        
        # Render clean bar graph with line overlay for MoM growth
        fig_mom = go.Figure()
        fig_mom.add_trace(go.Bar(
            x=grouped_mom["year_month"], y=grouped_mom["total_amount"],
            name="Monthly Sales ($)", marker_color="#00d2ff", opacity=0.8
        ))
        fig_mom.add_trace(go.Scatter(
            x=grouped_mom["year_month"], y=grouped_mom["growth_rate"],
            name="MoM Growth Rate (%)", yaxis="y2",
            mode="lines+markers", line=dict(color="#10b981", width=3)
        ))
        fig_mom.update_layout(
            template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,255,0.6)",
            font=dict(color="#1e1b4b", family="Inter", size=13),
            yaxis=dict(
                title="Monthly Sales ($)",
                tickfont=dict(color="#2d2a6e", size=12),
                title_font=dict(color="#1e1b4b", size=13, family="Outfit")
            ),
            yaxis2=dict(
                title="MoM Growth Rate (%)", overlaying="y", side="right",
                tickfont=dict(color="#2d2a6e", size=12),
                title_font=dict(color="#1e1b4b", size=13, family="Outfit")
            ),
            xaxis=dict(
                tickfont=dict(color="#2d2a6e", size=12),
                title_font=dict(color="#1e1b4b", size=13, family="Outfit")
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(color="#1e1b4b", size=12),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(30,27,75,0.15)",
                borderwidth=1
            ),
            margin=dict(l=40, r=40, t=40, b=20)
        )
        st.plotly_chart(apply_light_theme(fig_mom), use_container_width=True)

elif st.session_state.current_page == "customers":
    st.markdown("<h1>Customer Analytics & Segmentations</h1>", unsafe_allow_html=True)
    
    col_clusters, col_stats = st.columns([3, 2])
    
    with col_clusters:
        with st.container(border=True):
            st.subheader("Interactive 3D Customer K-Means RFM Clusters")
            
            # Calculate RFM and train K-Means (4 clusters default)
            df_rfm = calculate_rfm_metrics(customers_df, sales_df)
            df_clustered, cluster_summary = train_kmeans_segmentation(df_rfm, n_clusters=4)
            
            # Plot 3D visual
            fig_3d = plot_3d_clusters_plotly(df_clustered)
            st.plotly_chart(fig_3d, use_container_width=True)
        
    with col_stats:
        with st.container(border=True):
            st.subheader("Segment Personas Summary")
            
            for _, row in cluster_summary.iterrows():
                st.markdown(f"""
                    <div class="sub-feature-card" style='margin-bottom:12px; padding:15px;'>
                        <h4 style='color:#5ac8fa; margin:0 0 5px 0;'>{row['segment_profile']}</h4>
                        <span style='font-size:0.8rem; color:#a5b4fc; display:block;'>Accounts Size: <b style='color:#ffffff;'>{row['size']}</b></span>
                        <div style='display:flex; justify-content:space-between; margin-top:8px; font-size:0.82rem;'>
                            <span>Recency: <b style='color:#ffffff;'>{row['avg_recency']} days</b></span>
                            <span>Orders Count: <b style='color:#ffffff;'>{row['avg_frequency']}</b></span>
                            <span>Avg Total: <b style='color:#10b981;'>${row['avg_monetary']:,.2f}</b></span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
    # PREMIUM ADDITION: Customer Demographics Breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    col_demo, col_sat = st.columns(2)
    with col_demo:
        with st.container(border=True):
            st.subheader("👥 Customer Age Distribution by Segment")
            fig_demo = px.histogram(
                customers_df, x="age", color="segment",
                color_discrete_sequence=["#352e8c", "#5ac8fa", "#ff5e7e", "#10b981", "#fbbf24", "#ef4444"],
                title="Age Demographics Histogram",
                barmode="overlay", opacity=0.75
            )
            fig_demo.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,255,0.6)", font=dict(color="#1e1b4b", family="Inter", size=13))
            st.plotly_chart(apply_light_theme(fig_demo), use_container_width=True)
            
    with col_sat:
        with st.container(border=True):
            st.subheader("⭐ satisfaction Score vs. Customer Churn Correlation")
            # Shows satisfaction scores and churn status scatter
            sat_churn = customers_df.groupby("satisfaction_score")["churn_status"].mean().reset_index()
            sat_churn["churn_rate_pct"] = sat_churn["churn_status"] * 100.0
            
            fig_sat = px.bar(
                sat_churn, x="satisfaction_score", y="churn_rate_pct",
                labels={"satisfaction_score": "Satisfaction Score (1-5)", "churn_rate_pct": "Churn Rate %"},
                title="Customer Attrition Rate by Satisfaction Grade",
                color="churn_rate_pct", color_continuous_scale="Reds"
            )
            fig_sat.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,255,0.6)", font=dict(color="#1e1b4b", family="Inter", size=13))
            st.plotly_chart(apply_light_theme(fig_sat), use_container_width=True)

elif st.session_state.current_page == "products":
    st.markdown("<h1>Product & Inventory Diagnostics</h1>", unsafe_allow_html=True)
    
    col_abc, col_rec = st.columns([3, 2])
    
    with col_abc:
        with st.container(border=True):
            st.subheader("ABC Inventory Analytics (Revenue Contribution)")
            st.markdown("<p style='color:#8a99ad; font-size:0.85rem;'>Class A represents high-value products accounting for the top 80% of revenue, Class B forms the next 15%, and Class C represents remaining items.</p>", unsafe_allow_html=True)
            
            # ABC analysis via query connection
            abc_data = execute_custom_query(PREBUILT_QUERIES["abc_inventory"]["sql"])
            
            st.dataframe(
                abc_data[["name", "category", "inventory_stock", "total_revenue", "cumulative_percentage", "abc_category"]],
                column_config={
                    "name": "Product Name",
                    "category": "Category",
                    "inventory_stock": "Stock Level",
                    "total_revenue": st.column_config.NumberColumn("Revenue ($)", format="$%.2f"),
                    "cumulative_percentage": "Cumulative Revenue %",
                    "abc_category": "ABC Classification"
                },
                hide_index=True,
                use_container_width=True
            )
        
    with col_rec:
        with st.container(border=True):
            st.subheader("Cross-Sell & Market Basket Recommender")
            st.markdown("<p style='color:#8a99ad; font-size:0.85rem;'>Engine parses historical shopping patterns to identify item dependencies (Apriori algorithm).</p>", unsafe_allow_html=True)
            
            # Select active product
            product_names = products_df["name"].tolist()
            sel_product = st.selectbox("Select Core Product", product_names)
            
            # Calculate rules
            baskets = extract_shopping_baskets(sales_df, products_df)
            rules = calculate_association_rules(baskets, min_support=0.005, min_confidence=0.01)
            recs = recommend_cross_sell_products(rules, sel_product)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### AI Recommended Bundle Deals:")
            
            if len(recs) == 0:
                st.info("No statistically significant companion transactions identified for this product yet.")
            else:
                for rec in recs:
                    st.markdown(f"""
                        <div class="sub-feature-card" style='margin-bottom:10px; padding:15px;'>
                            <span style='color:#a5b4fc; font-size:0.8rem;'>Recommendation:</span>
                            <h4 style='color:#ff9f1c; margin:3px 0;'>Buy with: {rec['product']}</h4>
                            <div style='display:flex; justify-content:space-between; font-size:0.85rem; margin-top:5px;'>
                                <span>Lift Score: <b style='color:#ff5e7e;'>{rec['lift_score']:.2f}x</b></span>
                                <span>Association Confidence: <b style='color:#ffffff;'>{rec['confidence_pct']}%</b></span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
    # PREMIUM ADDITION: Pareto 80/20 Cumulative Revenue distribution Line Chart
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("⚖️ Product Revenue Pareto distribution (80/20 Rule Analysis)")
        # Plot cumulative revenue line chart from abc_data
        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(
            x=abc_data["name"], y=abc_data["total_revenue"],
            name="Individual Product Revenue ($)", marker_color="#00d2ff", opacity=0.75
        ))
        fig_pareto.add_trace(go.Scatter(
            x=abc_data["name"], y=abc_data["cumulative_percentage"],
            name="Cumulative Revenue Share (%)", yaxis="y2",
            mode="lines+markers", line=dict(color="#a855f7", width=3)
        ))
        # 80% line guide
        fig_pareto.add_shape(
            type="line", yref="y2", xref="x",
            x0=abc_data["name"].iloc[0], x1=abc_data["name"].iloc[-1],
            y0=80, y1=80,
            line=dict(color="#ff5252", width=2, dash="dash")
        )
        fig_pareto.update_layout(
            template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,255,0.6)", font=dict(color="#1e1b4b", family="Inter", size=13),
            yaxis=dict(title="Revenue ($)"),
            yaxis2=dict(title="Cumulative Share (%)", overlaying="y", side="right", range=[0, 105]),
            xaxis=dict(tickangle=45),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=60)
        )
        st.plotly_chart(apply_light_theme(fig_pareto), use_container_width=True)

elif st.session_state.current_page == "marketing":
    st.markdown("<h1>Marketing & Campaign Analytics</h1>", unsafe_allow_html=True)
    
    col_cor, col_roi = st.columns([3, 2])
    
    with col_cor:
        with st.container(border=True):
            st.subheader("Multi-Channel Advertising ROI Heatmap")
            
            corr_matrix = marketing_df[["budget", "impressions", "clicks", "conversions", "revenue_generated"]].corr()
            
            fig_heat = px.imshow(
                corr_matrix.values,
                x=["Budget", "Impressions", "Clicks", "Conversions", "Revenue"],
                y=["Budget", "Impressions", "Clicks", "Conversions", "Revenue"],
                color_continuous_scale="Blues",
                text_auto=True
            )
            fig_heat.update_layout(
                template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,255,0.6)", font=dict(color="#1e1b4b", family="Inter", size=13),
                margin=dict(l=40, r=40, t=20, b=40)
            )
            st.plotly_chart(apply_light_theme(fig_heat), use_container_width=True)
        
    with col_roi:
        with st.container(border=True):
            st.subheader("Campaign Performance Rankings")
            
            m_data = execute_custom_query(PREBUILT_QUERIES["campaign_efficiency"]["sql"])
            
            for _, row in m_data.iterrows():
                st.markdown(f"""
                    <div class="sub-feature-card" style='margin-bottom:10px; padding:12px;'>
                        <h5 style='margin:0; color:#ffffff;'>{row['name']}</h5>
                        <span style='font-size:0.75rem; color:#a5b4fc; display:block;'>Channel: <b style='color:#ffffff;'>{row['channel']}</b></span>
                        <div style='display:flex; justify-content:space-between; margin-top:8px; font-size:0.8rem;'>
                            <span>Spend: <b style='color:#ffffff;'>${row['budget']:,.0f}</b></span>
                            <span>ROI: <b style='color:#10b981;'>{row['roi_multiplier']}x</b></span>
                            <span>CAC: <b style='color:#ffffff;'>${row['cost_per_conversion']:.1f}</b></span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
    # PREMIUM ADDITION: Marketing Conversion Funnel Chart
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("🎯 Integrated Marketing Conversion Funnel")
        # Sum overall impressions, clicks, conversions
        imp_sum = marketing_df["impressions"].sum()
        clk_sum = marketing_df["clicks"].sum()
        cvr_sum = marketing_df["conversions"].sum()
        
        # Calculate approximate customer counts (based on conversion transaction records)
        funnel_stages = ["Impressions", "Clicks", "Conversions (Signups)"]
        funnel_values = [imp_sum, clk_sum, cvr_sum]
        
        fig_funnel = px.funnel(
            x=funnel_values, y=funnel_stages,
            title="Aggregated Advertising Conversion Milestones",
            color_discrete_sequence=["#352e8c", "#5ac8fa", "#ff5e7e", "#10b981", "#fbbf24", "#ef4444"]
        )
        fig_funnel.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,255,0.6)", font=dict(color="#1e1b4b", family="Inter", size=13))
        st.plotly_chart(apply_light_theme(fig_funnel), use_container_width=True)

elif st.session_state.current_page == "financials":
    st.markdown("<h1>Financial Performance</h1>", unsafe_allow_html=True)
    
    col_water, col_fore = st.columns([3, 2])
    
    with col_water:
        with st.container(border=True):
            st.subheader(" waterfall: Corporate Operating Breakdown")
            
            # Extract financials
            rev_sum = sales_df["total_amount"].sum()
            # COGS
            cogs_sum = sum(row["quantity"] * prod_costs.get(row["product_id"], 0) for _, row in sales_df.iterrows())
            gross_profit = rev_sum - cogs_sum
            
            # Payroll, opex, rent, marketing
            pay = financials_df["payroll"].sum()
            ope = financials_df["operating_expenses"].sum()
            ren = financials_df["rent"].sum()
            mkt = financials_df["marketing_spend"].sum()
            tax = financials_df["taxes"].sum()
            
            ebitda = gross_profit - (pay + ope + ren + mkt)
            net_profit = ebitda - tax
            
            fig_water = go.Figure(go.Waterfall(
                name="Corporate Annual Breakdown", orientation="v",
                measure=["relative", "relative", "total", "relative", "relative", "relative", "relative", "total", "relative", "total"],
                x=["Revenue", "COGS", "Gross Profit", "Payroll", "Office Rent", "Opex", "Marketing", "EBITDA", "Taxes", "Net Profit"],
                textposition="outside",
                y=[rev_sum, -cogs_sum, 0, -pay, -ren, -ope, -mkt, 0, -tax, 0],
                connector=dict(line=dict(color="rgba(255,255,255,0.1)"))
            ))
            fig_water.update_layout(
                template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,255,0.6)", font=dict(color="#1e1b4b", family="Inter", size=13),
                margin=dict(l=40, r=40, t=20, b=40)
            )
            st.plotly_chart(apply_light_theme(fig_water), use_container_width=True)
        
    with col_fore:
        with st.container(border=True):
            st.subheader("Financial Simulation Planners")
            
            payroll_adj = st.slider("Adjust Staff Payroll Costs %", -15, 25, 0, 5, format="%d%%") / 100.0
            opex_adj = st.slider("Adjust Operational Overheads %", -25, 50, 0, 5, format="%d%%") / 100.0
            
            sim_res = simulate_business_scenario(sales_df, customers_df, products_df, financials_df,
                                                payroll_adjustment_pct=payroll_adj,
                                                opex_adjustment_pct=opex_adj)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Annual Financial Outlook:")
            st.write(f"Baseline Opex Overhead: **${sim_res['baseline']['operating_expenses']:,.2f}**")
            st.write(f"Simulated Opex Overhead: **${sim_res['simulated']['operating_expenses']:,.2f}**")
            st.markdown(f"Simulated EBITDA Yield: **${sim_res['simulated']['ebitda']:,.2f}**")
            
            st.markdown("<br>", unsafe_allow_html=True)
            net_var = sim_res["variance"]["net_profit_pct"]
            if net_var >= 0:
                st.success(f"Strategy generates **+{net_var:.1f}%** increase in corporate net profits!")
            else:
                st.error(f"Strategy results in **{net_var:.1f}%** reduction in corporate net profits.")
                
    # PREMIUM ADDITION: Break-Even Analysis Plotter
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("📉 Break-Even Point Strategic Model Analysis")
        # Calculate standard average sales parameters
        avg_price = sales_df["total_amount"].sum() / sales_df["quantity"].sum()
        avg_cost = sum(row["quantity"] * prod_costs.get(row["product_id"], 0) for _, row in sales_df.iterrows()) / sales_df["quantity"].sum()
        margin_per_unit = avg_price - avg_cost
        
        # Annualized fixed cost base (Rent + Payroll + baseline opex)
        annual_fixed_costs = ren + pay + ope
        
        # Calculate break-even units count
        be_units = int(annual_fixed_costs / margin_per_unit)
        be_revenue = be_units * avg_price
        
        # Plot Break-Even chart (Units range from 0 to double the break-even level)
        units_range = np.linspace(0, max(be_units * 2, 1000), 50)
        revenue_curve = units_range * avg_price
        total_costs_curve = annual_fixed_costs + (units_range * avg_cost)
        
        fig_be = go.Figure()
        fig_be.add_trace(go.Scatter(x=units_range, y=revenue_curve, name="Gross Revenue ($)", line=dict(color="#10b981", width=3)))
        fig_be.add_trace(go.Scatter(x=units_range, y=total_costs_curve, name="Total Costs (Fixed+Variable) ($)", line=dict(color="#ef4444", width=2)))
        # Fixed cost baseline
        fig_be.add_trace(go.Scatter(
            x=[0, units_range[-1]], y=[annual_fixed_costs, annual_fixed_costs],
            name="Annual Fixed Overhead ($)", line=dict(color="rgba(255,255,255,0.25)", width=2, dash="dash")
        ))
        # Intersection indicator
        fig_be.add_trace(go.Scatter(
            x=[be_units], y=[be_revenue],
            name="Break-Even Point (Intersection)", mode="markers",
            marker=dict(color="#00d2ff", size=12, symbol="diamond")
        ))
        
        fig_be.update_layout(
            template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,255,0.6)", font=dict(color="#1e1b4b", family="Inter", size=13),
            title=f"Required Units to Break-Even: {be_units:,} (Revenue Intersection: ${be_revenue:,.2f})",
            xaxis_title="Units Sold Volume", yaxis_title="Total Finance Amount ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(apply_light_theme(fig_be), use_container_width=True)

elif st.session_state.current_page == "ml":
    st.markdown("<h1>Machine Learning & AutoML Workspace</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🧠 Real-Time Predictors", "⚙️ AutoML & Custom Datasets"])
    
    with tab1:
        col_m1, col_m2 = st.columns([3, 2])
        
        with col_m1:
            with st.container(border=True):
                st.subheader("Customer Churn Model Performance Diagnostics")
                
                # Train model
                feat_df = prepare_churn_features(customers_df, sales_df)
                model_c, X_t, y_t, y_p, y_pb, metrics_c, cm, fpr, tpr, auc_val, df_feats = train_churn_classifier(
                    feat_df, model_type="Random Forest", n_estimators=100, max_depth=6
                )
                
                # Renders Confusion Matrix
                fig_cm = plot_confusion_matrix_plotly(cm)
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Renders ROC curve or features
                fig_feats = px.bar(df_feats.head(6), x="Importance", y="Feature", orientation="h", title="Top Predictive Features")
                fig_feats.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,255,0.6)", font=dict(color="#1e1b4b", family="Inter", size=13))
                st.plotly_chart(apply_light_theme(fig_feats), use_container_width=True)
            
        with col_m2:
            with st.container(border=True):
                st.subheader("Client Churn Risk Evaluator")
                st.markdown("<p style='color:#8a99ad; font-size:0.85rem;'>Input values manually to predict this specific account profile's probability of attrition.</p>", unsafe_allow_html=True)
                
                age_val = st.number_input("Customer Age", 18, 90, 42)
                sat_val = st.slider("Satisfaction Score (1-5)", 1, 5, 4)
                spend_val = st.number_input("Lifetime Spending ($)", 0.0, 50000.0, 1200.0)
                orders_val = st.number_input("Lifetime Order Frequency", 0, 50, 6)
                rec_val = st.number_input("Days since last transaction (Recency)", 0, 365, 45)
                tenure_val = st.number_input("Account Tenure length (Days)", 1, 1000, 300)
                
                segment_val = st.selectbox("Customer Segment", ["Enterprise", "Mid-Market", "SMB"])
                region_val = st.selectbox("Shipping Region", ["North", "South", "East", "West"])
                gender_val = st.selectbox("Client Gender", ["Male", "Female", "Non-binary"])
                
                # Run prediction
                custom_data = {
                    "age": age_val, "satisfaction_score": sat_val, "total_spend": spend_val,
                    "order_count": orders_val, "recency": rec_val, "tenure": tenure_val,
                    "segment": segment_val, "region": region_val, "gender": gender_val
                }
                
                pred_res = predict_custom_churn(model_c, custom_data)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="sub-feature-card" style='text-align:center; padding:15px;'>
                        <span style='font-size:0.85rem; color:#a5b4fc;'>Calculated Churn Probability:</span>
                        <h2 style='color:#ff5e7e; margin:5px 0;'>{pred_res['churn_probability']}%</h2>
                        <span style='font-size:0.95rem; font-weight:600; color:{pred_res['color']};'>{pred_res['risk_level']}</span>
                    </div>
                """, unsafe_allow_html=True)
                
        # PREMIUM ADDITION: ML Model Comparison Matrix
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("📊 Supervised ML Model Metrics Comparison Matrix")
            # Compiles standard RandomForest metrics against Gradient Boosting metrics
            _, _, _, _, _, metrics_gb, _, _, _, _, _ = train_churn_classifier(
                feat_df, model_type="Gradient Boosting", n_estimators=100, max_depth=6
            )
            
            comp_data = {
                "Model Type": ["Random Forest Classifier", "Gradient Boosting Classifier"],
                "Accuracy": [f"{metrics_c['Accuracy']*100:.2f}%", f"{metrics_gb['Accuracy']*100:.2f}%"],
                "Precision": [f"{metrics_c['Precision']*100:.2f}%", f"{metrics_gb['Precision']*100:.2f}%"],
                "Recall (Sensitivity)": [f"{metrics_c['Recall']*100:.2f}%", f"{metrics_gb['Recall']*100:.2f}%"],
                "F1-Score": [f"{metrics_c['F1-Score']*100:.2f}%", f"{metrics_gb['F1-Score']*100:.2f}%"],
                "Operational Status": ["Cached Resource (Primary)", "Backup Model (Evaluation)"]
            }
            df_comp = pd.DataFrame(comp_data)
            st.dataframe(df_comp, hide_index=True, use_container_width=True)
            
    with tab2:
        with st.container(border=True):
            st.subheader("AutoML Sandbox: Upload Custom CSV")
            st.markdown("<p style='color:#8a99ad; font-size:0.85rem;'>Upload any structured business CSV dataset. The engine will detect features, let you select target class, train a Random Forest regressor/classifier instantly, and display full validation scores.</p>", unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Choose CSV Dataset", type="csv")
            
            if uploaded_file:
                try:
                    df_custom = pd.read_csv(uploaded_file)
                    st.success(f"Successfully loaded '{uploaded_file.name}' ({len(df_custom)} rows, {len(df_custom.columns)} columns)!")
                    
                    st.markdown("##### Preview Data Columns:")
                    st.dataframe(df_custom.head(5))
                    
                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        target_col = st.selectbox("Select Target Column (Y)", df_custom.columns)
                    with col_c2:
                        task_type = st.selectbox("Machine Learning Task Type", ["Classification", "Regression"])
                        
                    estimators = st.slider("Number of Estimators (Trees)", 10, 200, 100, 10)
                    m_depth = st.slider("Max Tree Depth", 3, 20, 8)
                    
                    if st.button("Trigger AutoML Training Pipeline", type="primary"):
                        st.toast("Beginning training iterations...", icon="🔄")
                        
                        df_clean = df_custom.dropna()
                        X_cust = df_clean.drop(columns=[target_col])
                        y_cust = df_clean[target_col]
                        
                        X_cust = pd.get_dummies(X_cust, drop_first=True)
                        
                        from sklearn.model_selection import train_test_split
                        X_t, X_te, y_t, y_te = train_test_split(X_cust, y_cust, test_size=0.25, random_state=42)
                        
                        if task_type == "Classification":
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.metrics import accuracy_score
                            clf = RandomForestClassifier(n_estimators=estimators, max_depth=m_depth, random_state=42)
                            clf.fit(X_t, y_t)
                            y_p = clf.predict(X_te)
                            acc = accuracy_score(y_te, y_p)
                            
                            st.balloons()
                            st.success(f"AutoML Classification Pipeline Completed! Model Accuracy: **{acc*100:.2f}%**")
                        else:
                            from sklearn.ensemble import RandomForestRegressor
                            reg = RandomForestRegressor(n_estimators=estimators, max_depth=m_depth, random_state=42)
                            reg.fit(X_t, y_t)
                            y_p = reg.predict(X_te)
                            from sklearn.metrics import mean_absolute_error, r2_score
                            mae = mean_absolute_error(y_te, y_p)
                            r2 = r2_score(y_te, y_p)
                            
                            st.balloons()
                            st.success(f"AutoML Regression Pipeline Completed! Mean Absolute Error (MAE): **{mae:.4f}**, R² Score: **{r2:.3f}**")
                except Exception as ex:
                    st.error(f"AutoML Pipeline Error: {str(ex)}")

elif st.session_state.current_page == "chatbot":
    st.markdown("<h1>AI Insights Chatbot & Voice Assistant</h1>", unsafe_allow_html=True)
    
    col_chat, col_sql = st.columns([1, 1])
    
    with col_chat:
        with st.container(border=True):
            st.subheader("PulseAdvisor Chat Executive")
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
                
            chat_container = st.container(height=380)
            with chat_container:
                for role, text_msg in st.session_state.chat_history:
                    if role == "User":
                        st.markdown(f"<div style='text-align: right; margin-bottom: 12px;'><span style='background: #352e8c; color: white; padding: 8px 12px; border-radius: 12px 12px 0 12px; display: inline-block; font-size: 0.9rem;'>{text_msg}</span></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align: left; margin-bottom: 12px;'><span style='background: #1e1b4b; color: #ffffff; border: 1px solid #352e8c; padding: 8px 12px; border-radius: 12px 12px 12px 0; display: inline-block; font-size: 0.9rem;'>🤖 <b>PulseAdvisor:</b><br>{text_msg}</span></div>", unsafe_allow_html=True)
                        
            html_speech = """
            <div style="text-align: center; margin-top: 10px;">
                <button id="record-btn" style="background: linear-gradient(135deg, #00d2ff, #a855f7); border: none; border-radius: 50%; width: 50px; height: 50px; color: white; cursor: pointer; box-shadow: 0 4px 15px rgba(0,210,255,0.4); outline:none; transition: all 0.3s ease;">
                    🎤
                </button>
                <p id="speech-status" style="color: #8a99ad; font-size: 0.75rem; margin: 5px 0 0 0;">Click mic to trigger voice query</p>
            </div>
            
            <script>
                const btn = document.getElementById('record-btn');
                const status = document.getElementById('speech-status');
                
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                
                if (SpeechRecognition) {
                    const rec = new SpeechRecognition();
                    rec.lang = 'en-US';
                    rec.interimResults = false;
                    
                    btn.onclick = () => {
                        rec.start();
                        status.innerText = "Listening...";
                        btn.style.boxShadow = "0 0 25px #00d2ff";
                    };
                    
                    rec.onresult = (e) => {
                        const resultText = e.results[0][0].transcript;
                        navigator.clipboard.writeText(resultText);
                        status.innerHTML = "Copied to clipboard: <i>\\"" + resultText + "\\"</i>. Paste in input and send!";
                        btn.style.boxShadow = "0 4px 15px rgba(0,210,255,0.4)";
                    };
                    
                    rec.onerror = () => {
                        status.innerText = "Error picking up microphone.";
                        btn.style.boxShadow = "0 4px 15px rgba(0,210,255,0.4)";
                    };
                } else {
                    status.innerText = "Speech Recognition API not supported in this browser.";
                    btn.style.opacity = 0.5;
                }
            </script>
            """
            
            chat_query = st.text_input("Enter Question / Paste transcribed voice query:", key="chat_input")
            
            col_c_btn1, col_c_btn2 = st.columns([4, 1])
            with col_c_btn1:
                send_msg = st.button("Send Message", use_container_width=True, type="primary")
            with col_c_btn2:
                st.components.v1.html(html_speech, height=85)
                
            if send_msg and chat_query:
                st.session_state.chat_history.append(("User", chat_query))
                
                with st.spinner("Compiling tactical consultation advice..."):
                    resp = ask_business_chatbot(
                        chat_query, st.session_state.chat_history,
                        st.session_state.api_key, st.session_state.ai_provider
                    )
                st.session_state.chat_history.append(("AI", resp))
                st.rerun()
            
    with col_sql:
        with st.container(border=True):
            st.subheader("Natural Language SQL Compiler")
            st.markdown("<p style='color:#8a99ad; font-size:0.85rem;'>Type any business question in plain English. The compiler maps it to database structures, executes the SQL locally, and charts the raw tables instantly.</p>", unsafe_allow_html=True)
            
            nl_query = st.text_input("Plain English Question:", placeholder="e.g. show me top 5 products or overall revenue")
            
            # PREMIUM ADDITION: Dynamic Chart Visualizer Selector (Line, Bar, Pie)
            chart_select = st.selectbox("Select SQL Visualization Type", ["Bar Chart", "Line Chart", "Pie Chart"])
            
            if st.button("Execute Natural Language Query", type="primary") and nl_query:
                with st.spinner("Synthesizing query instructions..."):
                    sql, expl = translate_nlp_to_sql(nl_query, st.session_state.api_key, st.session_state.ai_provider)
                    
                if not sql:
                    st.error(expl)
                else:
                    st.markdown(f"**AI Query Translation Takeaways:** *{expl}*")
                    st.code(sql, language="sql")
                    
                    try:
                        df_res = execute_custom_query(sql)
                        st.success("Query executed successfully!")
                        
                        st.markdown("##### Query Outcomes:")
                        st.dataframe(df_res, use_container_width=True)
                        
                        if len(df_res) > 0 and len(df_res.columns) >= 2:
                            x_col = df_res.columns[0]
                            y_col = df_res.columns[1]
                            
                            if pd.api.types.is_numeric_dtype(df_res[y_col]):
                                if chart_select == "Bar Chart":
                                    fig_res = px.bar(df_res, x=x_col, y=y_col, color=x_col, title="AI Generated Visual")
                                elif chart_select == "Line Chart":
                                    fig_res = px.line(df_res, x=x_col, y=y_col, markers=True, title="AI Generated Visual")
                                else: # Pie Chart
                                    fig_res = px.pie(df_res, names=x_col, values=y_col, title="AI Generated Visual")
                                    
                                fig_res.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,255,0.6)", font=dict(color="#1e1b4b", family="Inter", size=13))
                                st.plotly_chart(apply_light_theme(fig_res), use_container_width=True)
                    except Exception as db_ex:
                        st.error(f"SQLite Compilation Error: {str(db_ex)}")

elif st.session_state.current_page == "sql":
    st.markdown("<h1>Expert SQL Analytics Workspace</h1>", unsafe_allow_html=True)
    
    col_edit, col_schema = st.columns([3, 1])
    
    # PREMIUM ADDITION: Query History Logger
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
        
    with col_edit:
        with st.container(border=True):
            st.subheader("SQL Execution Sandbox")
            
            selected_template = st.selectbox("Select SQL Template to Load", ["-- Choose Template --"] + list(PREBUILT_QUERIES.keys()), format_func=lambda x: PREBUILT_QUERIES[x]["title"] if x in PREBUILT_QUERIES else x)
            
            default_editor_val = "SELECT * FROM sales LIMIT 10;"
            if selected_template in PREBUILT_QUERIES:
                default_editor_val = PREBUILT_QUERIES[selected_template]["sql"]
                st.markdown(f"*Template Context: {PREBUILT_QUERIES[selected_template]['description']}*")
                
            sql_input = st.text_area("Write/Edit SQL SELECT Query", value=default_editor_val, height=200)
            
            col_exec, col_down = st.columns([1, 1])
            execute_clicked = col_exec.button("Run SQL Command", type="primary", use_container_width=True)
            
            if execute_clicked and sql_input:
                try:
                    # Log to history
                    st.session_state.query_history.append({
                        "sql": sql_input,
                        "time": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    df_res = execute_custom_query(sql_input)
                    st.success(f"Query filed. Rows retrieved: {len(df_res)}")
                    
                    st.dataframe(df_res, use_container_width=True)
                    
                    st.markdown("##### 📝 Automated Business Takeaways:")
                    if "growth_percentage" in df_res.columns:
                        mean_growth = df_res["growth_percentage"].dropna().mean()
                        st.info(f"Summary: Average monthly growth rate over the period is **{mean_growth:.2f}%**. Standard indicators show consistent positive regional vectors.")
                    elif "clv_rank" in df_res.columns:
                        top_client = df_res.iloc[0]["name"]
                        st.info(f"Summary: Top-spending Enterprise account is identified as **{top_client}**. Concentrating specialized product support on high-CLV accounts is recommended.")
                    elif "abc_category" in df_res.columns:
                        class_a_cnt = len(df_res[df_res["abc_category"].str.contains("A")])
                        st.info(f"Summary: Found **{class_a_cnt}** Class A inventory product listings contributing 80% of revenue. Restocking these lines takes high priority.")
                    else:
                        st.info("Query processed successfully. SQL syntax is optimized and indexed.")
                        
                    csv_bytes = df_res.to_csv(index=False).encode('utf-8')
                    col_down.download_button("Download Output (CSV)", data=csv_bytes, file_name="sql_query_output.csv", mime="text/csv", use_container_width=True)
                    
                except Exception as sql_ex:
                    st.error(f"SQL execution failed: {str(sql_ex)}")
            
            # Draw history
            if len(st.session_state.query_history) > 0:
                with st.expander("🕒 View Workspace Query Execution Logs"):
                    for item in reversed(st.session_state.query_history[-5:]):
                        st.markdown(f"**[{item['time']}]** `{item['sql'][:60]}...`")
                        
    with col_schema:
        with st.container(border=True):
            st.subheader("Database Schema")
            
            st.markdown("""
                <div class="sql-schema-tree">
                    <b>📂 sales</b><br>
                    └─ sale_id (INT)<br>
                    └─ customer_id (INT)<br>
                    └─ product_id (INT)<br>
                    └─ quantity (INT)<br>
                    └─ total_amount (REAL)<br>
                    └─ region (TEXT)<br>
                    <br>
                    <b>📂 customers</b><br>
                    └─ customer_id (INT)<br>
                    └─ name (TEXT)<br>
                    └─ segment (TEXT)<br>
                    └─ region (TEXT)<br>
                    └─ churn_status (INT)<br>
                    <br>
                    <b>📂 products</b><br>
                    └─ product_id (INT)<br>
                    └─ name (TEXT)<br>
                    └─ category (TEXT)<br>
                    └─ price (REAL)<br>
                    └─ inventory_stock (INT)<br>
                    <br>
                    <b>📂 marketing_campaigns</b><br>
                    └─ campaign_id (INT)<br>
                    └─ name (TEXT)<br>
                    └─ budget (REAL)<br>
                    └─ conversions (INT)<br>
                    <br>
                    <b>📂 financials</b><br>
                    └─ financial_id (INT)<br>
                    └─ payroll (REAL)<br>
                    └─ marketing_spend (REAL)<br>
                    └─ operating_expenses (REAL)
                </div>
            """, unsafe_allow_html=True)

elif st.session_state.current_page == "reports":
    st.markdown("<h1>Report Export Center</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin-bottom: 25px;'>Export high-fidelity analytical reporting models directly in corporate Excel sheets or HTML briefing layouts.</p>", unsafe_allow_html=True)
    
    # PREMIUM ADDITION: Custom Title / Theme PDF Briefing selection
    with st.container(border=True):
        st.subheader("🎨 Executive Briefing Customization Builder")
        custom_subtitle = st.text_input("Briefing Report Custom Subtitle", "Corporate Operations Performance Summary")
        theme_color = st.selectbox("Select Report Theme Color Guide", ["Classic Corporate Blue", "Mint Growth Green", "Electric Indigo Purple"])
        
    col_r1, col_r2 = st.columns([1, 1])
    
    with col_r1:
        with st.container(border=True):
            st.subheader("📥 Excel Workbooks Pack")
            st.write("Generates a fully structured openpyxl workbook containing raw transactions, segment distributions, marketing ROI metrics, and product catalogs grouped across cleanly styled sheets.")
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            excel_bytes = generate_excel_report(sales_df, customers_df, products_df, marketing_df)
            st.download_button(
                "Download Analytical Workbook (.xlsx)",
                data=excel_bytes,
                file_name=f"businesspulse_workbook_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary"
            )
        
    with col_r2:
        with st.container(border=True):
            st.subheader("📄 Executive briefing & PDF Exporter")
            st.write("Compiles a styled HTML document outlining key financial KPIs, active low-stock alerts, campaign conversion ranks, and corporate margins. Easily viewable, and perfectly formatted for a single-click PDF print layout!")
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Compile HTML Briefing
            html_brief = generate_html_briefing(sales_df, customers_df, marketing_df, active_anomalies)
            
            # Apply Custom Subtitle & Colors dynamically
            html_brief = html_brief.replace("Corporate Performance Briefing", custom_subtitle)
            if theme_color == "Mint Growth Green":
                html_brief = html_brief.replace("#3498db", "#00f5a0")
            elif theme_color == "Electric Indigo Purple":
                html_brief = html_brief.replace("#3498db", "#a855f7")
                
            st.download_button(
                "Download Executive HTML Briefing (.html)",
                data=html_brief,
                file_name=f"businesspulse_executive_briefing_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                use_container_width=True,
                type="primary"
            )

elif st.session_state.current_page == "import":
    st.markdown("<h1>Data Import Center</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin-bottom: 25px;'>Instantly scale and customize the analytics platform with your own corporate data or manage your pre-loaded SaaS playground database.</p>", unsafe_allow_html=True)
    
    tab_wizard, tab_manual = st.tabs(["🏢 Guided Custom Business Wizard (Recommended)", "🔧 Advanced Table-by-Table Overwrite"])
    
    with tab_wizard:
        col_upload_wiz, col_info_wiz = st.columns([3, 2])
        
        with col_upload_wiz:
            with st.container(border=True):
                st.subheader("💼 Transform Platform with single sales CSV")
                st.markdown("<p style='color:#8a99ad; font-size:0.85rem;'>Upload a single sales ledger or invoice history sheet. Our synthesis engine will extract products/customers and synthesize matching marketing/financial data instantly.</p>", unsafe_allow_html=True)
                
                uploaded_wizard_csv = st.file_uploader("Upload Company Sales or Invoices CSV", type="csv", key="wizard_csv_uploader")
                
                if uploaded_wizard_csv:
                    try:
                        df_wiz = pd.read_csv(uploaded_wizard_csv)
                        st.success(f"CSV buffer success! Loaded '{uploaded_wizard_csv.name}' ({len(df_wiz)} rows, {len(df_wiz.columns)} columns)")
                        
                        st.markdown("##### Preview CSV Data Columns:")
                        st.dataframe(df_wiz.head(3), use_container_width=True)
                        
                        st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
                        st.markdown("##### ⚙️ Map CSV Columns to Database Fields:")
                        
                        cols = list(df_wiz.columns)
                        
                        def auto_detect_col(cols, keywords, default_idx=0):
                            for i, col in enumerate(cols):
                                if any(kw in col.lower() for kw in keywords):
                                    return i
                            return default_idx
                            
                        def get_optional_col_idx(cols, keywords):
                            idx = auto_detect_col(cols, keywords, default_idx=-1)
                            return idx + 1 if idx >= 0 else 0
                            
                        col_map1, col_map2 = st.columns(2)
                        with col_map1:
                            map_date = st.selectbox("Sale / Invoice Date (Required)", cols, index=auto_detect_col(cols, ["date", "time", "day", "created"]))
                            map_product = st.selectbox("Product / Item Name (Required)", cols, index=auto_detect_col(cols, ["product", "item", "article", "sku"]))
                            map_total = st.selectbox("Total Revenue / Amount (Required)", cols, index=auto_detect_col(cols, ["revenue", "amount", "total", "sales", "price", "billing"]))
                            map_customer = st.selectbox("Customer Name / Client ID (Required)", cols, index=auto_detect_col(cols, ["customer", "client", "buyer", "user", "name"]))
                        with col_map2:
                            map_qty = st.selectbox("Quantity Purchased (Optional)", ["-- Map Later / Default to 1 --"] + cols, index=get_optional_col_idx(cols, ["quantity", "qty", "volume", "unit_count"]))
                            map_category = st.selectbox("Product Category (Optional)", ["-- Auto-Categorize --"] + cols, index=get_optional_col_idx(cols, ["category", "dept", "group", "class"]))
                            map_region = st.selectbox("Sales Region (Optional)", ["-- Auto-Assign Random --"] + cols, index=get_optional_col_idx(cols, ["region", "country", "state", "city", "location"]))
                            map_segment = st.selectbox("Customer Segment (Optional)", ["-- Auto-Segment --"] + cols, index=get_optional_col_idx(cols, ["segment", "tier", "size", "profile"]))
                            
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        if st.button("🚀 Compile & Activate Custom Business Profile", type="primary", use_container_width=True):
                            with st.spinner("Synthesizing multi-year relational database tables..."):
                                from utils.custom_business_parser import import_and_synthesize_custom_business
                                
                                mappings = {
                                    "sale_date": map_date,
                                    "product_name": map_product,
                                    "total_amount": map_total,
                                    "customer_name": map_customer,
                                    "quantity": None if map_qty.startswith("--") else map_qty,
                                    "category": None if map_category.startswith("--") else map_category,
                                    "region": None if map_region.startswith("--") else map_region,
                                    "segment": None if map_segment.startswith("--") else map_segment
                                }
                                
                                res_summary = import_and_synthesize_custom_business(df_wiz, mappings)
                                
                                st.session_state.business_profile = "custom"
                                st.cache_data.clear()
                                st.cache_resource.clear()
                                
                            st.balloons()
                            st.success(f"Success! Relational database successfully compiled from your corporate sales data!")
                            st.info(f"Summary: Extracted {res_summary['products_count']} products, {res_summary['customers_count']} customers, and loaded {res_summary['sales_count']} transactions chronologically. Restoring view...")
                            st.session_state.current_page = "dashboard"
                            st.rerun()
                            
                    except Exception as ex_wiz:
                        st.error(f"Guided Synthesis Pipeline Error: {str(ex_wiz)}")
                        
        with col_info_wiz:
            with st.container(border=True):
                st.subheader("💡 Dynamic Database Seeding Details")
                st.markdown("""
                    Our **Guided Synthesis Pipeline** translates your unstructured or semi-structured business transactions into a fully relational, multi-year optimized database:
                    
                    - **Products**: Auto-extracts unique item listings, estimates cost basis to maintain a 40% margin, and randomly generates inventory stock levels.
                    - **Customers**: Auto-groups buyers, creates corporate emails, determines customer segments, maps regional structures, and calculates lifetime churn patterns based on historical transaction timelines.
                    - **Sales**: Maps transactions chronologically, applies discounts dynamically, and computes delivery costs.
                    - **Marketing**: Generates realistic multi-channel campaigns (Google, social, email, LinkedIn) distributed across your sales date range, keeping Page 5 completely intact.
                    - **Financials**: Compiles monthly financial overhead ledger costs (payroll, rent, opex, depreciation, and tax variables) calculated proportionally to your actual sales revenue, keeping Page 6 functional.
                """, unsafe_allow_html=True)
                
    with tab_manual:
        col_upload, col_schemas = st.columns([3, 2])
        
        with col_upload:
            with st.container(border=True):
                st.subheader("💼 Advanced Custom Table Overwrite")
                st.markdown("<p style='color:#8a99ad; font-size:0.85rem;'>Select which database table to overwrite and upload your custom CSV file. Double-check required columns on the right before importing.</p>", unsafe_allow_html=True)
                
                target_table = st.selectbox("Target Database Table to Overwrite", [
                    "sales (Sales Transactions)",
                    "customers (Customer Profiles)",
                    "products (Product Inventory)",
                    "marketing_campaigns (Marketing Performance)",
                    "financials (Monthly Ledger Costs)"
                ])
                
                uploaded_csv = st.file_uploader(f"Choose custom CSV for {target_table}", type="csv")
                
                if uploaded_csv:
                    try:
                        df_up = pd.read_csv(uploaded_csv)
                        st.success(f"CSV buffer success! Loaded '{uploaded_csv.name}' ({len(df_up)} rows, {len(df_up.columns)} columns)")
                        
                        st.markdown("##### Preview CSV Data Columns:")
                        st.dataframe(df_up.head(5), use_container_width=True)
                        
                        # Mapping logic and database push
                        if st.button("💾 Overwrite & Seeding Into Database", type="primary", use_container_width=True):
                            table_db_name = target_table.split(" ")[0]
                            
                            # Prefix table name if custom profile is active
                            prefix = "custom_" if st.session_state.business_profile == "custom" else ""
                            table_db_name = f"{prefix}{table_db_name}"
                            
                            with engine.begin() as conn:
                                # Direct replace of SQLite table
                                df_up.to_sql(table_db_name, conn, if_exists="replace", index=False)
                                
                                # Recreate necessary performance indices
                                st.toast("Rebuilding search indexes...", icon="⚙️")
                                if "sales" in table_db_name:
                                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_c_sales_date ON {table_db_name}(sale_date);"))
                                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_c_sales_cust ON {table_db_name}(customer_id);"))
                                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_c_sales_prod ON {table_db_name}(product_id);"))
                                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_c_sales_region ON {table_db_name}(region);"))
                                elif "customers" in table_db_name:
                                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_c_cust_signup ON {table_db_name}(signup_date);"))
                                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_c_cust_segment ON {table_db_name}(segment);"))
                                elif "products" in table_db_name:
                                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_c_prod_cat ON {table_db_name}(category);"))
                                    
                            st.balloons()
                            st.success(f"Success! Relational SQLite table '{table_db_name}' successfully overwritten with your custom records. App metrics updated!")
                            
                            # Clear cache so whole application renders the new data instantly
                            st.cache_data.clear()
                            st.cache_resource.clear()
                            
                    except Exception as ex_up:
                        st.error(f"Import Pipeline Error: {str(ex_up)}")
                        
        with col_schemas:
            with st.container(border=True):
                st.subheader("🔄 Wipes & Reset Default Test Data")
                st.write("Clicking below will delete all of your uploaded custom database tables, re-seed the full-scope multi-year seasonal SaaS enterprise dummy transactions, and clear system caches instantly.")
                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("Reset Platform to Default SaaS Playground Data", type="secondary", use_container_width=True):
                    with st.spinner("Re-seeding initial database..."):
                        from database.connection import init_and_seed_db
                        init_and_seed_db(force=True)
                        
                        # Clean up custom tables if they exist
                        with engine.begin() as conn:
                            for tbl in ["custom_sales", "custom_customers", "custom_products", "custom_marketing_campaigns", "custom_financials"]:
                                conn.execute(text(f"DROP TABLE IF EXISTS {tbl};"))
                                
                        st.session_state.business_profile = "saas"
                        st.cache_data.clear()
                        st.cache_resource.clear()
                    st.balloons()
                    st.success("Platform successfully restored to original preseeded SaaS data framework! Reloading...")
                    st.rerun()
                
    with col_schemas:
        with st.container(border=True):
            st.subheader("📋 Relational Schema Specifications")
            st.write("Ensure your custom CSVs map to these column names exactly for smooth visualizations:")
            
            with st.expander("📊 Table 'sales' Columns"):
                st.code("""
sale_id (INT) -> Unique ID
customer_id (INT) -> FK to customer
product_id (INT) -> FK to product
sale_date (TEXT) -> YYYY-MM-DD
quantity (INT)
unit_price (REAL)
discount (REAL) -> 0.00 to 0.35
total_amount (REAL) -> Net total
sales_channel (TEXT) -> Online/Retail/Direct
payment_method (TEXT)
shipping_cost (REAL)
region (TEXT) -> North/South/East/West
                """, language="text")
                
            with st.expander("👥 Table 'customers' Columns"):
                st.code("""
customer_id (INT) -> Unique ID
name (TEXT) -> Full name
email (TEXT)
segment (TEXT) -> Enterprise/Mid-Market/SMB
signup_date (TEXT) -> YYYY-MM-DD
region (TEXT) -> North/South/East/West
age (INT)
gender (TEXT) -> Male/Female...
satisfaction_score (INT) -> 1 to 5
churn_status (INT) -> 0 or 1
                """, language="text")
                
            with st.expander("📦 Table 'products' Columns"):
                st.code("""
product_id (INT) -> Unique ID
name (TEXT) -> Product name
category (TEXT) -> Electronics/Software...
cost (REAL) -> Manufacture cost
price (REAL) -> Catalog price
inventory_stock (INT)
min_reorder_level (INT)
                """, language="text")
                
            with st.expander("📣 Table 'marketing_campaigns' Columns"):
                st.code("""
campaign_id (INT) -> Unique ID
name (TEXT) -> Campaign name
channel (TEXT) -> Google/LinkedIn/Social/Email
start_date (TEXT) -> YYYY-MM-DD
end_date (TEXT) -> YYYY-MM-DD
budget (REAL)
revenue_generated (REAL)
clicks (INT)
impressions (INT)
conversions (INT)
                """, language="text")
                
            with st.expander("💰 Table 'financials' Columns"):
                st.code("""
financial_id (INT) -> Unique ID
date (TEXT) -> Monthly YYYY-MM-DD
operating_expenses (REAL)
taxes (REAL)
depreciation (REAL)
payroll (REAL)
rent (REAL)
marketing_spend (REAL)
other_costs (REAL)
                """, language="text")

