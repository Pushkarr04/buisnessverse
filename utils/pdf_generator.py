import io
import pandas as pd
from datetime import datetime

def generate_excel_report(df_sales, df_customers, df_products, df_marketing):
    """
    Creates a styled multi-sheet Excel workbook.
    Sheets:
      1. Sales Performance (grouped sales, total revenues)
      2. Customer Insights (demographics, segmentation)
      3. Marketing Campaigns (ROI allocations)
      4. Inventory Catalog (stock status, prices)
    Returns raw bytes ready for streamlit file download.
    """
    output = io.BytesIO()
    
    # Process some summaries for Excel
    df_sales_summary = df_sales.copy()
    df_sales_summary["sale_date"] = pd.to_datetime(df_sales_summary["sale_date"])
    df_sales_summary["year_month"] = df_sales_summary["sale_date"].dt.strftime("%Y-%m")
    
    monthly_sales = df_sales_summary.groupby("year_month").agg(
        revenue=("total_amount", "sum"),
        orders=("sale_id", "count"),
        avg_order=("total_amount", "mean")
    ).reset_index()
    
    # Build Excel tabs
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        monthly_sales.to_excel(writer, sheet_name="Monthly Performance", index=False)
        df_sales.head(1000).to_excel(writer, sheet_name="Sales Details (Sample)", index=False)
        df_customers.to_excel(writer, sheet_name="Customer Roster", index=False)
        df_marketing.to_excel(writer, sheet_name="Marketing Campaigns", index=False)
        df_products.to_excel(writer, sheet_name="Product Catalog", index=False)
        
    processed_data = output.getvalue()
    return processed_data

def generate_html_briefing(df_sales, df_customers, df_marketing, anomalies):
    """
    Generates a beautifully styled corporate briefing document in HTML format.
    Styled with professional fonts, structural borders, grids, and KPI callouts.
    Can be easily printed to PDF by the user's web browser in one-click.
    """
    # 1. Compile high-level metrics
    total_rev = df_sales["total_amount"].sum()
    total_sales = len(df_sales)
    avg_order = df_sales["total_amount"].mean()
    
    churn_rate = round(df_customers["churn_status"].mean() * 100, 1)
    satisfaction = round(df_customers["satisfaction_score"].mean(), 2)
    
    marketing_budget = df_marketing["budget"].sum()
    marketing_rev = df_marketing["revenue_generated"].sum()
    marketing_roi = round(marketing_rev / marketing_budget, 2)
    
    # 2. Top-selling products table
    prod_counts = df_sales.groupby("product_id")["quantity"].sum().reset_index()
    # Map to product name (dummy map or logic)
    
    # HTML template
    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Buisnessverse - Executive Performance Briefing</title>
<style>
    body {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2c3e50;
        background-color: #ffffff;
        margin: 40px;
        line-height: 1.5;
    }}
    .header {{
        border-bottom: 3px solid #3498db;
        padding-bottom: 20px;
        margin-bottom: 30px;
    }}
    .logo {{
        font-size: 28px;
        font-weight: bold;
        color: #2c3e50;
        letter-spacing: -1px;
    }}
    .logo span {{
        color: #3498db;
    }}
    .meta {{
        float: right;
        text-align: right;
        font-size: 13px;
        color: #7f8c8d;
        margin-top: 10px;
    }}
    .title {{
        font-size: 24px;
        margin: 20px 0 10px 0;
        color: #2c3e50;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-bottom: 30px;
    }}
    .card {{
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        background-color: #f8fafc;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .card-val {{
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin: 5px 0;
    }}
    .card-lbl {{
        font-size: 11px;
        text-transform: uppercase;
        color: #7f8c8d;
        font-weight: 600;
        letter-spacing: 1px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 14px;
    }}
    th {{
        background-color: #f1f5f9;
        color: #475569;
        font-weight: bold;
        text-align: left;
        padding: 12px;
        border-bottom: 2px solid #cbd5e1;
    }}
    td {{
        padding: 12px;
        border-bottom: 1px solid #e2e8f0;
    }}
    tr:nth-child(even) td {{
        background-color: #f8fafc;
    }}
    .alert-box {{
        padding: 15px;
        background-color: #fffaf0;
        border-left: 4px solid #dd6b20;
        border-radius: 4px;
        margin-bottom: 15px;
        font-size: 14px;
    }}
    .alert-title {{
        font-weight: bold;
        color: #dd6b20;
        margin-bottom: 4px;
    }}
    .footer {{
        margin-top: 50px;
        border-top: 1px solid #cbd5e1;
        padding-top: 15px;
        text-align: center;
        font-size: 12px;
        color: #94a3b8;
    }}
    @media print {{
        body {{ margin: 20px; }}
        .card {{ background-color: #f8fafc !important; -webkit-print-color-adjust: exact; }}
    }}
</style>
</head>
<body>
    <div class="header">
        <div class="meta">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}<br>
            System Status: ACTIVE
        </div>
        <div class="logo">Business<span>Pulse</span> AI</div>
        <div style="clear:both;"></div>
    </div>
    
    <div class="title">Corporate Performance Briefing</div>
    <p>This automated intelligence report outlines the financial health, sales trends, inventory anomalies, and marketing efficiency indexes compiled for the current reporting term. Use this report for strategic resource allocation planning.</p>
    
    <div class="grid">
        <div class="card">
            <div class="card-lbl">Gross Revenue</div>
            <div class="card-val">${total_rev:,.2f}</div>
            <div class="card-lbl">{total_sales:,} orders filed</div>
        </div>
        <div class="card">
            <div class="card-lbl">Customer Churn Rate</div>
            <div class="card-val">{churn_rate}%</div>
            <div class="card-lbl">Satisfaction: {satisfaction}/5.0</div>
        </div>
        <div class="card">
            <div class="card-lbl">Marketing ROI</div>
            <div class="card-val">{marketing_roi}x</div>
            <div class="card-lbl">${marketing_budget:,.0f} total budget</div>
        </div>
    </div>
    
    <div class="title" style="font-size: 18px; margin-top: 40px; border-bottom: 1px solid #cbd5e1; padding-bottom: 5px;">Active Operational Warnings & Anomalies</div>
"""
    
    if len(anomalies) == 0:
        html_content += "<p>No active anomalies or operational risks are currently identified by the AI engines.</p>"
    else:
        for idx, alert in enumerate(anomalies[:4]):
            html_content += f"""
            <div class="alert-box">
                <div class="alert-title">[{alert['type']}] {alert['title']} (Severity: {alert['severity']})</div>
                <div>{alert['desc']}</div>
            </div>
            """
            
    html_content += """
    <div class="title" style="font-size: 18px; margin-top: 40px; border-bottom: 1px solid #cbd5e1; padding-bottom: 5px;">Recent Campaign Allocations</div>
    <table>
        <thead>
            <tr>
                <th>Campaign Name</th>
                <th>Channel</th>
                <th>Budget</th>
                <th>Revenue Generated</th>
                <th>ROI</th>
                <th>Conversions</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for _, row in df_marketing.head(6).iterrows():
        roi = round(row["revenue_generated"] / row["budget"], 2)
        html_content += f"""
            <tr>
                <td>{row['name']}</td>
                <td>{row['channel']}</td>
                <td>${row['budget']:,.2f}</td>
                <td>${row['revenue_generated']:,.2f}</td>
                <td><b>{roi}x</b></td>
                <td>{row['conversions']:,}</td>
            </tr>
        """
        
    html_content += """
        </tbody>
    </table>
    
    <div class="footer">
        Buisnessverse Intelligence Briefing &bull; Proprietary Internal Document &bull; Confidential
    </div>
</body>
</html>
"""
    return html_content
