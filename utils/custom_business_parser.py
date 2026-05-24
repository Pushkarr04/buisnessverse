import os
import re
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from sqlalchemy import text
from database.connection import engine

def import_and_synthesize_custom_business(df_up, mappings):
    """
    Cleans uploaded custom CSV business transactions, extracts relational entities 
    (products, customers, sales), and auto-synthesizes corresponding marketing 
    campaigns and monthly financials. Overwrites or creates custom_ tables in SQLite.
    
    mappings = {
        "sale_date": col_name,
        "product_name": col_name,
        "category": col_name or None,
        "quantity": col_name or None,
        "total_amount": col_name,
        "customer_name": col_name,
        "region": col_name or None,
        "segment": col_name or None
    }
    """
    # 1. Clean dates and build clean base transactions
    df_base = pd.DataFrame()
    
    # Map Sale Date
    date_col = mappings["sale_date"]
    df_base["sale_date_raw"] = pd.to_datetime(df_up[date_col], errors="coerce")
    # Drop rows with NaT dates
    df_base = df_base.dropna(subset=["sale_date_raw"]).copy()
    if len(df_base) == 0:
        raise ValueError("Could not parse any valid transaction dates from the mapped date column.")
        
    df_base["sale_date"] = df_base["sale_date_raw"].dt.strftime("%Y-%m-%d")
    
    # Map Product Name
    prod_col = mappings["product_name"]
    df_base["product_name"] = df_up.loc[df_base.index, prod_col].fillna("General Product").astype(str).str.strip()
    
    # Map Category
    cat_col = mappings.get("category")
    if cat_col and cat_col in df_up.columns:
        df_base["category"] = df_up.loc[df_base.index, cat_col].fillna("General").astype(str).str.strip()
    else:
        # Auto-detect category from product name keyword
        categories = []
        for name in df_base["product_name"]:
            name_l = name.lower()
            if any(k in name_l for k in ["laptop", "computer", "monitor", "headphone", "audio", "tv", "vr", "device"]):
                categories.append("Electronics")
            elif any(k in name_l for k in ["software", "cloud", "saas", "premium", "licence", "license", "api", "sub"]):
                categories.append("Software")
            elif any(k in name_l for k in ["desk", "chair", "paper", "marker", "pen", "notebook", "office", "supplies"]):
                categories.append("Office Supplies")
            elif any(k in name_l for k in ["server", "hub", "cable", "router", "switch", "camera", "hardware"]):
                categories.append("Hardware")
            else:
                categories.append("General Goods")
        df_base["category"] = categories
        
    # Map Quantity
    qty_col = mappings.get("quantity")
    if qty_col and qty_col in df_up.columns:
        df_base["quantity"] = pd.to_numeric(df_up.loc[df_base.index, qty_col], errors="coerce").fillna(1).astype(int)
        df_base.loc[df_base["quantity"] <= 0, "quantity"] = 1
    else:
        df_base["quantity"] = 1
        
    # Map Total Amount
    amt_col = mappings["total_amount"]
    df_base["total_amount"] = pd.to_numeric(df_up.loc[df_base.index, amt_col], errors="coerce").fillna(100.0).astype(float)
    df_base.loc[df_base["total_amount"] < 0, "total_amount"] = 0.0
    
    # Map Customer Name
    cust_col = mappings["customer_name"]
    df_base["customer_name"] = df_up.loc[df_base.index, cust_col].fillna("Guest Buyer").astype(str).str.strip()
    
    # Map Region
    reg_col = mappings.get("region")
    if reg_col and reg_col in df_up.columns:
        df_base["region"] = df_up.loc[df_base.index, reg_col].fillna("Global").astype(str).str.strip()
    else:
        df_base["region"] = [random.choice(["North", "South", "East", "West"]) for _ in range(len(df_base))]
        
    # Map Segment
    seg_col = mappings.get("segment")
    if seg_col and seg_col in df_up.columns:
        df_base["segment"] = df_up.loc[df_base.index, seg_col].fillna("SMB").astype(str).str.strip()
    else:
        df_base["segment"] = [np.random.choice(["Enterprise", "Mid-Market", "SMB"], p=[0.15, 0.35, 0.50]) for _ in range(len(df_base))]
        
    # 2. Extract and synthesize custom_products
    unique_prods = df_base.groupby("product_name").agg({
        "category": "first",
        "total_amount": "sum",
        "quantity": "sum"
    }).reset_index()
    
    products_list = []
    prod_id_map = {}
    next_prod_id = 2001
    
    for _, row in unique_prods.iterrows():
        p_name = row["product_name"]
        p_cat = row["category"]
        total_rev = row["total_amount"]
        total_qty = row["quantity"]
        
        # Calculate sensible price per unit
        avg_price = round(total_rev / total_qty, 2) if total_qty > 0 else 99.99
        if avg_price <= 0:
            avg_price = 49.99
            
        cost = round(avg_price * 0.6, 2) # Clean 40% margin
        stock = random.randint(30, 400)
        reorder = random.choice([15, 25, 40])
        
        products_list.append({
            "product_id": next_prod_id,
            "name": p_name,
            "category": p_cat,
            "cost": cost,
            "price": avg_price,
            "inventory_stock": stock,
            "min_reorder_level": reorder
        })
        
        prod_id_map[p_name] = next_prod_id
        next_prod_id += 1
        
    df_products_custom = pd.DataFrame(products_list)
    
    # 3. Extract and synthesize custom_customers
    unique_custs = df_base.groupby("customer_name").agg({
        "segment": "first",
        "region": "first",
        "sale_date_raw": ["min", "max"]
    }).reset_index()
    unique_custs.columns = ["customer_name", "segment", "region", "first_purchase", "last_purchase"]
    
    customers_list = []
    cust_id_map = {}
    next_cust_id = 6001
    
    max_dataset_date = df_base["sale_date_raw"].max()
    
    for _, row in unique_custs.iterrows():
        c_name = row["customer_name"]
        segment = row["segment"]
        region = row["region"]
        first_p = row["first_purchase"]
        last_p = row["last_purchase"]
        
        # Parse clean email
        clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', c_name)
        parts = clean_name.lower().split()
        if len(parts) >= 2:
            email = f"{parts[0]}.{parts[1]}{random.randint(10,99)}@example-business.com"
        elif len(parts) == 1:
            email = f"{parts[0]}{random.randint(10,99)}@example-business.com"
        else:
            email = f"buyer{random.randint(100,999)}@example-business.com"
            
        # Signup date is 10 to 60 days before first purchase date
        signup_dt = first_p - timedelta(days=random.randint(10, 60))
        signup_str = signup_dt.strftime("%Y-%m-%d")
        
        # Demographics
        age = random.randint(23, 62)
        gender = np.random.choice(["Male", "Female", "Non-binary"], p=[0.47, 0.48, 0.05])
        satisfaction = random.choice([3, 4, 5])
        
        # Churn logic: if last purchase is > 90 days ago, higher probability
        days_dormant = (max_dataset_date - last_p).days
        churn_prob = 0.05
        if days_dormant > 180:
            churn_prob = 0.70
        elif days_dormant > 90:
            churn_prob = 0.40
            
        churn_status = 1 if np.random.rand() < churn_prob else 0
        
        customers_list.append({
            "customer_id": next_cust_id,
            "name": c_name,
            "email": email,
            "segment": segment,
            "signup_date": signup_str,
            "region": region,
            "age": age,
            "gender": gender,
            "satisfaction_score": satisfaction,
            "churn_status": churn_status
        })
        
        cust_id_map[c_name] = next_cust_id
        next_cust_id += 1
        
    df_customers_custom = pd.DataFrame(customers_list)
    
    # 4. Map transactions for custom_sales
    sales_list = []
    next_sale_id = 80001
    
    for idx, row in df_base.iterrows():
        p_name = row["product_name"]
        c_name = row["customer_name"]
        s_date = row["sale_date"]
        qty = row["quantity"]
        total = row["total_amount"]
        region = row["region"]
        
        p_id = prod_id_map.get(p_name, 2001)
        c_id = cust_id_map.get(c_name, 6001)
        
        # Find unit price
        p_price = next(p["price"] for p in products_list if p["product_id"] == p_id)
        
        # Back-calculate discount if total amount differs from catalog sum
        expected_total = qty * p_price
        discount = 0.0
        if expected_total > total and expected_total > 0:
            discount = (expected_total - total) / expected_total
            discount = min(0.35, max(0.0, round(discount, 2)))
            
        channel = np.random.choice(["Online", "Retail", "Direct"], p=[0.55, 0.25, 0.20])
        pmt = np.random.choice(["Credit Card", "Bank Transfer", "PayPal", "Purchase Order"], p=[0.45, 0.25, 0.15, 0.15])
        ship = round(random.uniform(5.0, 30.0) if qty < 10 else 0.0, 2)
        
        sales_list.append({
            "sale_id": next_sale_id,
            "customer_id": c_id,
            "product_id": p_id,
            "sale_date": s_date,
            "quantity": qty,
            "unit_price": p_price,
            "discount": discount,
            "total_amount": total,
            "sales_channel": channel,
            "payment_method": pmt,
            "shipping_cost": ship,
            "region": region
        })
        next_sale_id += 1
        
    df_sales_custom = pd.DataFrame(sales_list)
    df_sales_custom = df_sales_custom.sort_values(by="sale_date").reset_index(drop=True)
    df_sales_custom["sale_id"] = range(80001, 80001 + len(df_sales_custom))
    
    # 5. Synthesize custom_marketing_campaigns
    min_date = df_base["sale_date_raw"].min()
    max_date = df_base["sale_date_raw"].max()
    days_range = (max_date - min_date).days
    
    # Total revenue baseline
    total_revenue_custom = df_base["total_amount"].sum()
    marketing_budget = max(5000.0, total_revenue_custom * 0.05) # 5% budget
    
    campaigns_list = []
    channels = ["Google Ads", "Social Media", "Email", "LinkedIn Ads"]
    campaign_names = [
        ("Google Ads - Growth Catalyst", "Google Ads"),
        ("Social Booster Drive", "Social Media"),
        ("Executive Email Nurture", "Email"),
        ("LinkedIn Enterprise Funnel", "LinkedIn Ads"),
        ("Flash Sales Campaign", "Social Media"),
        ("Search Engine Optimization Campaign", "Google Ads")
    ]
    
    budget_per_camp = round(marketing_budget / len(campaign_names), 2)
    
    for idx, (name, chan) in enumerate(campaign_names):
        # Evenly distribute start/end dates
        start_offset = int((days_range / len(campaign_names)) * idx)
        start_dt = min_date + timedelta(days=start_offset)
        duration = random.randint(15, 60)
        end_dt = min_date + timedelta(days=min(days_range, start_offset + duration))
        
        # Calculate click/conversion metrics
        impressions = int(budget_per_camp * random.randint(22, 35))
        
        if chan == "Email":
            ctr = random.uniform(0.12, 0.16)
            cvr = random.uniform(0.05, 0.09)
        elif chan == "LinkedIn Ads":
            ctr = random.uniform(0.009, 0.014)
            cvr = random.uniform(0.015, 0.03)
        else:
            ctr = random.uniform(0.025, 0.05)
            cvr = random.uniform(0.02, 0.04)
            
        clicks = int(impressions * ctr)
        conversions = int(clicks * cvr)
        
        # Realistic positive ROI
        roi = random.uniform(1.2, 2.3)
        rev_gen = round(budget_per_camp * roi, 2)
        
        campaigns_list.append({
            "campaign_id": idx + 1,
            "name": name,
            "channel": chan,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "budget": budget_per_camp,
            "revenue_generated": rev_gen,
            "clicks": clicks,
            "impressions": impressions,
            "conversions": conversions
        })
        
    df_marketing_custom = pd.DataFrame(campaigns_list)
    
    # 6. Synthesize custom_financials (Monthly ledger records)
    monthly_dates = pd.date_range(start=min_date.replace(day=1), end=max_date, freq="MS")
    financials_list = []
    f_id = 1
    
    avg_monthly_rev = df_sales_custom["total_amount"].sum() / max(1, len(monthly_dates))
    
    for dt in monthly_dates:
        # Find actual monthly sales revenue
        sales_m = df_sales_custom[
            (pd.to_datetime(df_sales_custom["sale_date"]).dt.year == dt.year) &
            (pd.to_datetime(df_sales_custom["sale_date"]).dt.month == dt.month)
        ]
        rev_sum = sales_m["total_amount"].sum()
        
        # COGS
        cogs = 0.0
        for _, sale in sales_m.iterrows():
            prod_c = next(p["cost"] for p in products_list if p["product_id"] == sale["product_id"])
            cogs += sale["quantity"] * prod_c
        cogs = round(cogs, 2)
        
        # Financial scaling factors
        operating_expenses = round((rev_sum * 0.15) + (avg_monthly_rev * 0.02) * random.uniform(0.9, 1.1), 2)
        payroll = round((rev_sum * 0.25) + (avg_monthly_rev * 0.03) * random.uniform(0.95, 1.05), 2)
        rent = round(avg_monthly_rev * 0.08, 2) # fixed base rent
        
        # Amortized campaign spend
        marketing_spend = round(df_marketing_custom[
            (pd.to_datetime(df_marketing_custom["start_date"]) <= dt) &
            (pd.to_datetime(df_marketing_custom["end_date"]) >= dt)
        ]["budget"].sum() / 2.0, 2)
        if marketing_spend == 0:
            marketing_spend = round(avg_monthly_rev * 0.05, 2)
            
        depreciation = round(avg_monthly_rev * 0.02, 2)
        other_costs = round(random.uniform(200, 1500), 2)
        
        # Taxes
        ebitda = rev_sum - cogs - (payroll + rent + operating_expenses + marketing_spend)
        taxes = round(max(0.0, ebitda * 0.22), 2)
        
        financials_list.append({
            "financial_id": f_id,
            "date": dt.strftime("%Y-%m-%d"),
            "operating_expenses": operating_expenses,
            "taxes": taxes,
            "depreciation": depreciation,
            "payroll": payroll,
            "rent": rent,
            "marketing_spend": marketing_spend,
            "other_costs": other_costs
        })
        f_id += 1
        
    df_financials_custom = pd.DataFrame(financials_list)
    
    # 7. Write synthesized DataFrames into database custom_ tables
    with engine.begin() as conn:
        df_products_custom.to_sql("custom_products", conn, if_exists="replace", index=False)
        df_customers_custom.to_sql("custom_customers", conn, if_exists="replace", index=False)
        df_sales_custom.to_sql("custom_sales", conn, if_exists="replace", index=False)
        df_marketing_custom.to_sql("custom_marketing_campaigns", conn, if_exists="replace", index=False)
        df_financials_custom.to_sql("custom_financials", conn, if_exists="replace", index=False)
        
        # Recreate essential indexes on custom tables
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_c_sales_date ON custom_sales(sale_date);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_c_sales_cust ON custom_sales(customer_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_c_sales_prod ON custom_sales(product_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_c_sales_region ON custom_sales(region);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_c_cust_signup ON custom_customers(signup_date);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_c_cust_segment ON custom_customers(segment);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_c_prod_cat ON custom_products(category);"))
        
    return {
        "sales_count": len(df_sales_custom),
        "customers_count": len(df_customers_custom),
        "products_count": len(df_products_custom),
        "campaigns_count": len(df_marketing_custom),
        "financials_count": len(df_financials_custom)
    }
