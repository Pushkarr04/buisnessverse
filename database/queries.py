import pandas as pd
from database.connection import engine

# Prebuilt advanced SQL queries with clear explanations of their analytical utility
PREBUILT_QUERIES = {
    "clv_ranking": {
        "title": "Customer Lifetime Value (CLV) & Rankings",
        "description": "Calculates the total orders, total spend, average order size, and dense ranks customers globally based on lifetime purchase value. Useful for identifying 'Champions' segment.",
        "sql": """WITH CustomerSpending AS (
    SELECT 
        c.customer_id,
        c.name,
        c.segment,
        c.region,
        COUNT(s.sale_id) as total_orders,
        SUM(s.total_amount) as total_spent,
        ROUND(AVG(s.total_amount), 2) as avg_order_value,
        MIN(s.sale_date) as first_order_date,
        MAX(s.sale_date) as last_order_date
    FROM customers c
    JOIN sales s ON c.customer_id = s.customer_id
    GROUP BY c.customer_id, c.name, c.segment, c.region
)
SELECT 
    customer_id,
    name,
    segment,
    region,
    total_orders,
    ROUND(total_spent, 2) as total_spent,
    avg_order_value,
    DENSE_RANK() OVER (ORDER BY total_spent DESC) as clv_rank
FROM CustomerSpending
ORDER BY total_spent DESC
LIMIT 50;"""
    },
    
    "monthly_growth": {
        "title": "Monthly Revenue & Sales Growth Rate (CTE + Lag)",
        "description": "Uses a CTE to group sales by month, and a Window Function (LAG) to compute the growth rate percentage compared to the previous month. Identifies growth spikes or negative trends.",
        "sql": """WITH MonthlySales AS (
    SELECT 
        strftime('%Y-%m', sale_date) as sales_month,
        SUM(total_amount) as monthly_revenue
    FROM sales
    GROUP BY sales_month
),
MonthlyGrowth AS (
    SELECT 
        sales_month,
        monthly_revenue,
        LAG(monthly_revenue, 1) OVER (ORDER BY sales_month) as prev_month_revenue
    FROM MonthlySales
)
SELECT 
    sales_month,
    ROUND(monthly_revenue, 2) as monthly_revenue,
    ROUND(prev_month_revenue, 2) as prev_month_revenue,
    ROUND(((monthly_revenue - prev_month_revenue) / prev_month_revenue) * 100, 2) as growth_percentage
FROM MonthlyGrowth
ORDER BY sales_month;"""
    },
    
    "running_total_region": {
        "title": "Regional Sales Running Total (Window Function)",
        "description": "Utilizes a window function to partition transactions by region and calculate the running cumulative revenue sum over chronological dates. Essential for visualizing geographical momentum.",
        "sql": """SELECT 
    sale_id,
    sale_date,
    region,
    total_amount,
    ROUND(SUM(total_amount) OVER (
        PARTITION BY region 
        ORDER BY sale_date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ), 2) as running_total_revenue
FROM sales
ORDER BY region, sale_date
LIMIT 100;"""
    },
    
    "abc_inventory": {
        "title": "ABC Inventory Classification (Pareto Rule)",
        "description": "Uses a CTE to calculate each product's lifetime revenue share, aggregates a running sum percentage, and ranks items into Class A (high contribution: top 80%), Class B (medium: 80-95%), and Class C (low: 95-100%).",
        "sql": """WITH ProductRevenue AS (
    SELECT 
        p.product_id,
        p.name,
        p.category,
        p.inventory_stock,
        COALESCE(SUM(s.total_amount), 0) as total_revenue
    FROM products p
    LEFT JOIN sales s ON p.product_id = s.product_id
    GROUP BY p.product_id
),
CumulativeRevenue AS (
    SELECT 
        product_id,
        name,
        category,
        inventory_stock,
        total_revenue,
        SUM(total_revenue) OVER (ORDER BY total_revenue DESC) as running_total_revenue,
        SUM(total_revenue) OVER () as grand_total_revenue
    FROM ProductRevenue
),
Percentages AS (
    SELECT 
        product_id,
        name,
        category,
        inventory_stock,
        total_revenue,
        ROUND((running_total_revenue / grand_total_revenue) * 100, 2) as cumulative_percentage
    FROM CumulativeRevenue
)
SELECT 
    product_id,
    name,
    category,
    inventory_stock,
    ROUND(total_revenue, 2) as total_revenue,
    cumulative_percentage,
    CASE 
        WHEN cumulative_percentage <= 80.0 THEN 'A (High Revenue)'
        WHEN cumulative_percentage <= 95.0 THEN 'B (Medium Revenue)'
        ELSE 'C (Low Revenue)'
    END as abc_category
FROM Percentages
ORDER BY total_revenue DESC;"""
    },
    
    "campaign_efficiency": {
        "title": "Marketing Campaign Efficiency & ROI Analysis",
        "description": "Analyzes the ROI of marketing campaigns, computing the Customer Acquisition Cost (CAC) based on conversion counts, and evaluating return multiplier relative to budget allocations.",
        "sql": """SELECT 
    campaign_id,
    name,
    channel,
    budget,
    revenue_generated,
    ROUND(revenue_generated - budget, 2) as net_profit,
    ROUND((revenue_generated / budget), 2) as roi_multiplier,
    conversions,
    ROUND(budget / conversions, 2) as cost_per_conversion
FROM marketing_campaigns
ORDER BY roi_multiplier DESC;"""
    }
}

def execute_custom_query(sql_query: str) -> pd.DataFrame:
    """
    Executes a custom SQL query string against the business_pulse SQLite database.
    Validates safety patterns before execution.
    """
    # Simple security safety check
    forbidden_keywords = ["drop", "delete", "truncate", "update", "insert", "alter", "create table", "drop table"]
    cleaned_query = sql_query.lower().strip()
    
    for kw in forbidden_keywords:
        if kw in cleaned_query:
            raise ValueError(f"Security Alert: Execution of write/structure modifying commands is restricted. Found '{kw}' keyword.")
            
    # Dynamic table translation for custom business profile
    try:
        import streamlit as st
        import re
        from database.connection import check_custom_tables_exist
        if st.session_state.get("business_profile") == "custom" and check_custom_tables_exist():
            for tbl in ["customers", "sales", "products", "marketing_campaigns", "financials"]:
                # Replace whole word matches only to avoid partial replacing inside names
                sql_query = re.sub(rf"\b{tbl}\b", f"custom_{tbl}", sql_query, flags=re.IGNORECASE)
    except Exception:
        pass

    with engine.connect() as conn:
        df = pd.read_sql_query(sql_query, conn)
    return df
