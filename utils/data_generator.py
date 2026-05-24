import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

def generate_business_data():
    """
    Generates highly realistic relational business data spanning approximately 3 years (2023-2026).
    Includes seasonal trends, segment correlations, promotional anomalies, and realistic churn signals.
    """
    # 1. Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2026, 5, 1)
    date_range_days = (end_date - start_date).days
    
    # --- 1. PRODUCT CATALOG ---
    categories = {
        "Electronics": [
            ("Apex Laptop Pro", 800, 1200, 120, 20),
            ("Quantum Monitor 27\"", 180, 299, 150, 25),
            ("SmartHub Station", 60, 99, 300, 50),
            ("Vector VR Headset", 250, 450, 80, 15),
            ("AudioStream Headphones", 45, 89, 400, 60)
        ],
        "Software": [
            ("CloudSync SaaS Core (Annual)", 120, 499, 9999, 0), # virtual inventory
            ("Pulse Analytics Premium (Annual)", 180, 799, 9999, 0),
            ("SecureShield Enterprise (Annual)", 300, 1200, 9999, 0),
            ("DesignFlow Plus (Monthly)", 15, 39, 9999, 0)
        ],
        "Office Supplies": [
            ("ErgoDesk Stand-Up", 150, 299, 60, 10),
            ("Aura Mesh Task Chair", 90, 179, 85, 15),
            ("EcoWhite Paper Case", 15, 29, 600, 100),
            ("ProMarker Pack 24-Color", 4, 12, 1000, 200)
        ],
        "Hardware": [
            ("Titan Server Rack", 1200, 2400, 15, 3),
            ("OptiCable 10Gb Hub", 75, 149, 180, 30),
            ("SecurCam Outdoor 4K", 110, 199, 120, 25)
        ]
    }
    
    products_list = []
    prod_id = 101
    for cat, items in categories.items():
        for name, cost, price, stock, reorder in items:
            products_list.append({
                "product_id": prod_id,
                "name": name,
                "category": cat,
                "cost": cost,
                "price": price,
                "inventory_stock": stock,
                "min_reorder_level": reorder
            })
            prod_id += 1
    
    df_products = pd.DataFrame(products_list)
    
    # --- 2. CUSTOMERS ---
    num_customers = 1200
    regions = ["North", "South", "East", "West"]
    segments = ["Enterprise", "Mid-Market", "SMB"]
    genders = ["Male", "Female", "Non-binary", "Prefer not to say"]
    
    # Weighted probabilities for segments & regions
    seg_weights = [0.15, 0.35, 0.50]  # Enterprise: 15%, Mid-Market: 35%, SMB: 50%
    reg_weights = [0.25, 0.20, 0.30, 0.25]  # East is largest, South is smallest
    
    customers_list = []
    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth",
                   "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen",
                   "Christopher", "Nancy", "Daniel", "Lisa", "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
                  "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
    
    for c_id in range(1001, 1001 + num_customers):
        segment = np.random.choice(segments, p=seg_weights)
        region = np.random.choice(regions, p=reg_weights)
        gender = np.random.choice(genders, p=[0.46, 0.48, 0.04, 0.02])
        
        # Correlate age with segment (Enterprise decision makers tend to be slightly older)
        if segment == "Enterprise":
            age = int(np.random.normal(48, 8))
        else:
            age = int(np.random.normal(36, 10))
        age = max(21, min(70, age))
        
        # Signup date spread out over the 3 years
        signup_days_ago = np.random.randint(0, date_range_days)
        signup_date = start_date + timedelta(days=signup_days_ago)
        
        # Correlate satisfaction score with churn
        # Baseline satisfaction score
        sat_score = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.08, 0.15, 0.42, 0.30])
        
        # Higher churn probability for low satisfaction, and SMB segment
        churn_prob = 0.05
        if sat_score == 1:
            churn_prob += 0.65
        elif sat_score == 2:
            churn_prob += 0.40
        elif sat_score == 3:
            churn_prob += 0.15
            
        if segment == "SMB":
            churn_prob += 0.10
        elif segment == "Enterprise":
            churn_prob -= 0.03
            
        churn_status = 1 if np.random.rand() < max(0.01, min(0.95, churn_prob)) else 0
        
        # Generate names
        fname = random.choice(first_names)
        lname = random.choice(last_names)
        name = f"{fname} {lname}"
        email = f"{fname.lower()}.{lname.lower()}{random.randint(10,99)}@example.com"
        
        customers_list.append({
            "customer_id": c_id,
            "name": name,
            "email": email,
            "segment": segment,
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "region": region,
            "age": age,
            "gender": gender,
            "satisfaction_score": sat_score,
            "churn_status": churn_status
        })
        
    df_customers = pd.DataFrame(customers_list)
    
    # --- 3. SALES TRANSACTIONS ---
    sales_list = []
    sale_id = 50001
    
    channels = ["Online", "Retail", "Direct"]
    pmt_methods = ["Credit Card", "Bank Transfer", "PayPal", "Purchase Order"]
    
    # Seasonal multipliers (Nov/Dec holiday spikes, Jan slump)
    season_multipliers = {
        1: 0.75, 2: 0.82, 3: 0.95, 4: 1.02, 5: 1.05, 6: 1.00,
        7: 0.92, 8: 0.94, 9: 1.08, 10: 1.12, 11: 1.35, 12: 1.50
    }
    
    # Generate sales
    for _, cust in df_customers.iterrows():
        cust_id = cust["customer_id"]
        segment = cust["segment"]
        region = cust["region"]
        signup_dt = datetime.strptime(cust["signup_date"], "%Y-%m-%d")
        
        # Decide transaction frequency based on segment
        if segment == "Enterprise":
            # Enterprise has fewer, high-value orders
            num_purchases = random.choice([1, 2, 3, 4, 5])
        elif segment == "Mid-Market":
            num_purchases = random.randint(2, 8)
        else:
            # SMB purchases more frequently but smaller totals
            num_purchases = random.randint(3, 12)
            
        # Create transactions
        for i in range(num_purchases):
            # Purchase date must be after signup date
            days_after_signup = np.random.randint(0, max(1, (end_date - signup_dt).days))
            sale_date = signup_dt + timedelta(days=days_after_signup)
            
            # Month seasonal multiplier
            month = sale_date.month
            multiplier = season_multipliers[month]
            
            # Regional Logistics Anomaly: East Region drop in Q3 2024 (Jul, Aug, Sep)
            if region == "East" and sale_date.year == 2024 and month in [7, 8, 9]:
                multiplier *= 0.45 # drop sales by 55% in East region
                
            # Promotional Spike Anomaly: March 2025 (Global marketing campaign)
            if sale_date.year == 2025 and month == 3:
                multiplier *= 1.40
                
            # Filter products catalog based on segment suitability
            # Enterprise buys high-end Hardware/Software, SMB buys Office Supplies/Software/lower Electronics
            if segment == "Enterprise":
                prod_choices = df_products[df_products["category"].isin(["Software", "Electronics", "Hardware"])]
            elif segment == "SMB":
                prod_choices = df_products[df_products["category"].isin(["Office Supplies", "Software", "Electronics"])]
            else:
                prod_choices = df_products
                
            product = prod_choices.sample(1).iloc[0]
            prod_id = product["product_id"]
            price = product["price"]
            
            # Select realistic quantity based on segment and product cost
            if segment == "Enterprise":
                # High quantities for licenses, moderate for hardware
                quantity = random.randint(5, 25) if product["category"] in ["Software", "Office Supplies"] else random.randint(1, 8)
            elif segment == "Mid-Market":
                quantity = random.randint(2, 10) if product["category"] in ["Software", "Office Supplies"] else random.randint(1, 4)
            else:
                quantity = random.randint(1, 4)
                
            # Apply multiplier check (chance of buying)
            if np.random.rand() > multiplier * 0.8:
                continue
                
            # Apply logical discounts
            # High quantity and Enterprise gets larger discounts
            discount = 0.0
            if quantity >= 10:
                discount = random.choice([0.10, 0.15, 0.20])
            elif quantity >= 5:
                discount = random.choice([0.05, 0.10])
                
            if segment == "Enterprise":
                discount += 0.05
            discount = min(0.35, discount) # capped at 35%
            
            total_amount = round(quantity * price * (1.0 - discount), 2)
            
            # Core channels and payment methods based on segment
            if segment == "Enterprise":
                channel = "Direct"
                pmt_method = "Purchase Order"
            elif segment == "SMB":
                channel = np.random.choice(channels, p=[0.70, 0.20, 0.10])
                pmt_method = np.random.choice(pmt_methods, p=[0.40, 0.10, 0.50, 0.00])
            else:
                channel = np.random.choice(channels, p=[0.45, 0.35, 0.20])
                pmt_method = np.random.choice(pmt_methods, p=[0.50, 0.20, 0.20, 0.10])
                
            shipping_cost = round(random.uniform(5.0, 50.0) if product["category"] != "Software" else 0.0, 2)
            
            sales_list.append({
                "sale_id": sale_id,
                "customer_id": cust_id,
                "product_id": prod_id,
                "sale_date": sale_date.strftime("%Y-%m-%d"),
                "quantity": quantity,
                "unit_price": price,
                "discount": round(discount, 2),
                "total_amount": total_amount,
                "sales_channel": channel,
                "payment_method": pmt_method,
                "shipping_cost": shipping_cost,
                "region": region
            })
            sale_id += 1
            
    df_sales = pd.DataFrame(sales_list)
    # Sort sales chronologically
    df_sales = df_sales.sort_values(by="sale_date").reset_index(drop=True)
    df_sales["sale_id"] = range(50001, 50001 + len(df_sales))
    
    # --- 4. MARKETING CAMPAIGNS ---
    marketing_list = []
    m_id = 1
    
    campaign_names = [
        ("Google Search Ads - Q1", "Google Ads", datetime(2023, 1, 15), datetime(2023, 3, 30), 12000, 1.8),
        ("Social Spring Booster", "Social Media", datetime(2023, 3, 1), datetime(2023, 4, 30), 8000, 2.2),
        ("LinkedIn Enterprise Outreach", "LinkedIn Ads", datetime(2023, 5, 10), datetime(2023, 8, 15), 25000, 1.4),
        ("Nurture Email Campaign", "Email", datetime(2023, 6, 1), datetime(2023, 6, 30), 1500, 6.5),
        ("Google Search Ads - Q3", "Google Ads", datetime(2023, 7, 10), datetime(2023, 9, 30), 15000, 1.9),
        ("Holiday Sale Bonanza", "Social Media", datetime(2023, 11, 1), datetime(2023, 12, 25), 20000, 2.8),
        
        ("Google Search Ads - Q1 (24)", "Google Ads", datetime(2024, 1, 15), datetime(2024, 3, 30), 14000, 1.7),
        ("Social Spring Booster (24)", "Social Media", datetime(2024, 3, 1), datetime(2024, 4, 30), 9500, 2.4),
        ("LinkedIn Enterprise Outreach (24)", "LinkedIn Ads", datetime(2024, 5, 10), datetime(2024, 8, 15), 28000, 1.5),
        ("East Logistics Recovery", "Email", datetime(2024, 10, 1), datetime(2024, 10, 31), 2500, 4.0),
        ("Holiday Sale Bonanza (24)", "Social Media", datetime(2024, 11, 1), datetime(2024, 12, 25), 24000, 3.0),
        
        ("Spring Awakening Promo", "Social Media", datetime(2025, 2, 15), datetime(2025, 3, 31), 18000, 3.5), # Large promo spike
        ("Google Search Ads - Q2 (25)", "Google Ads", datetime(2025, 4, 1), datetime(2025, 6, 30), 16000, 2.0),
        ("LinkedIn Enterprise Outreach (25)", "LinkedIn Ads", datetime(2025, 5, 10), datetime(2025, 8, 15), 32000, 1.6),
        ("Black Friday Blitz", "Social Media", datetime(2025, 11, 1), datetime(2025, 11, 30), 22000, 3.8),
        ("Year End Corporate Drive", "LinkedIn Ads", datetime(2025, 12, 1), datetime(2025, 12, 31), 15000, 2.1)
    ]
    
    for name, channel, start_d, end_d, budget, roi_multiplier in campaign_names:
        days = (end_d - start_d).days
        impressions = int(budget * np.random.uniform(20, 35))
        
        # CTR vary by channel
        if channel == "Email":
            ctr = np.random.uniform(0.12, 0.18)
        elif channel == "LinkedIn Ads":
            ctr = np.random.uniform(0.008, 0.015)
        elif channel == "Google Ads":
            ctr = np.random.uniform(0.03, 0.06)
        else: # Social Media
            ctr = np.random.uniform(0.015, 0.04)
            
        clicks = int(impressions * ctr)
        
        # Conversion rate
        cvr = np.random.uniform(0.015, 0.035) if channel != "Email" else np.random.uniform(0.05, 0.10)
        conversions = int(clicks * cvr)
        
        revenue_generated = round(budget * roi_multiplier * np.random.normal(1.0, 0.08), 2)
        
        marketing_list.append({
            "campaign_id": m_id,
            "name": name,
            "channel": channel,
            "start_date": start_d.strftime("%Y-%m-%d"),
            "end_date": end_d.strftime("%Y-%m-%d"),
            "budget": budget,
            "revenue_generated": revenue_generated,
            "clicks": clicks,
            "impressions": impressions,
            "conversions": conversions
        })
        m_id += 1
        
    df_marketing = pd.DataFrame(marketing_list)
    
    # --- 5. FINANCIALS ---
    # Create monthly financial records
    financial_list = []
    f_id = 1
    
    # Months spanning the period
    monthly_dates = pd.date_range(start="2023-01-01", end="2026-05-01", freq="MS")
    
    for dt in monthly_dates:
        # Base operating costs with realistic monthly variations and a slight growth trend
        months_elapsed = (dt.year - 2023) * 12 + (dt.month - 1)
        growth_factor = 1.0 + (months_elapsed * 0.008) # +0.8% expense growth per month
        
        rent = round(12000 * growth_factor, 2)
        payroll = round(65000 * growth_factor * np.random.uniform(0.97, 1.03), 2)
        marketing_spend = round(df_marketing[
            (pd.to_datetime(df_marketing["start_date"]) <= dt) & 
            (pd.to_datetime(df_marketing["end_date"]) >= dt)
        ]["budget"].sum() / 3.0, 2) # amortize campaign costs
        
        if marketing_spend == 0:
            marketing_spend = round(5000 * np.random.uniform(0.8, 1.2), 2)
            
        operating_expenses = round(15000 * growth_factor * np.random.uniform(0.9, 1.1), 2)
        depreciation = round(2500, 2)
        
        # Calculate revenue from sales table for this month
        sales_this_month = df_sales[
            (pd.to_datetime(df_sales["sale_date"]).dt.year == dt.year) &
            (pd.to_datetime(df_sales["sale_date"]).dt.month == dt.month)
        ]
        
        revenue_sum = sales_this_month["total_amount"].sum()
        
        # Approximate COGS
        # Link sale quantities back to product costs
        cogs = 0.0
        for _, sale in sales_this_month.iterrows():
            prod_cost = df_products[df_products["product_id"] == sale["product_id"]]["cost"].values[0]
            cogs += sale["quantity"] * prod_cost
            
        cogs = round(cogs, 2)
        gross_profit = revenue_sum - cogs
        
        # Taxes
        ebitda = gross_profit - (payroll + rent + operating_expenses + marketing_spend)
        taxes = round(max(0, ebitda * 0.22), 2) # 22% corporate tax rate on positive EBITDA
        
        other_costs = round(np.random.uniform(500, 2000), 2)
        
        financial_list.append({
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
        
    df_financials = pd.DataFrame(financial_list)
    
    return df_products, df_customers, df_sales, df_marketing, df_financials

if __name__ == "__main__":
    p, c, s, m, f = generate_business_data()
    print("Generated Products:", len(p))
    print("Generated Customers:", len(c))
    print("Generated Sales Transactions:", len(s))
    print("Generated Marketing Campaigns:", len(m))
    print("Generated Financial Records:", len(f))
    print("Total revenue:", s["total_amount"].sum())
