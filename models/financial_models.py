import numpy as np
import pandas as pd

def get_financial_baselines(df_sales, df_financials):
    """
    Extracts the current baseline totals from the database to compare against simulator outcomes.
    """
    total_revenue = df_sales["total_amount"].sum()
    
    # Get last 12 months financial summary
    df_f = df_financials.copy()
    df_f["date"] = pd.to_datetime(df_f["date"])
    df_f_last_year = df_f.sort_values(by="date").tail(12)
    
    baseline_payroll = df_f_last_year["payroll"].sum()
    baseline_rent = df_f_last_year["rent"].sum()
    baseline_opex = df_f_last_year["operating_expenses"].sum()
    baseline_marketing = df_f_last_year["marketing_spend"].sum()
    baseline_taxes = df_f_last_year["taxes"].sum()
    
    return {
        "revenue": round(total_revenue, 2),
        "payroll": round(baseline_payroll, 2),
        "rent": round(baseline_rent, 2),
        "opex": round(baseline_opex, 2),
        "marketing": round(baseline_marketing, 2),
        "taxes": round(baseline_taxes, 2)
    }

def simulate_business_scenario(df_sales, df_customers, df_products, df_financials,
                              price_adjustment_pct=0.0,
                              marketing_spend_pct=0.0,
                              payroll_adjustment_pct=0.0,
                              opex_adjustment_pct=0.0):
    """
    Runs a detailed macroeconomic simulation based on elasticity models:
    - Segment Price Elasticity of Demand:
      - SMB: -2.2 (Highly price sensitive)
      - Mid-Market: -1.4 (Moderately sensitive)
      - Enterprise: -0.6 (Highly price inelastic)
    - Marketing ROI scaling: Logarithmic diminishing returns on campaign budgets.
    - Financial variables: recodes COGS, margins, and payroll opex to yield simulated EBITDA, Taxes, and Net Income.
    """
    # 1. Baseline metrics
    df_s = df_sales.copy()
    df_c = df_customers.copy()
    df_p = df_products.copy()
    
    baseline_revenue = df_s["total_amount"].sum()
    
    # Calculate baseline COGS
    # Coerce product_id to string to avoid float/int type mismatch
    prod_costs = {str(k): v for k, v in zip(df_p["product_id"], df_p["cost"])}
    baseline_cogs = sum(row["quantity"] * prod_costs.get(str(row["product_id"]), 0) for _, row in df_s.iterrows())
    
    # 2. Simulate Sales Price & Elasticity adjustments
    # Elasticity coefficients per segment
    elasticity = {
        "SMB": -2.2,
        "Mid-Market": -1.4,
        "Enterprise": -0.6
    }
    
    simulated_sales = []
    
    # Pre-map customer segment by string customer_id to avoid repeated filtering crashes and type mismatches
    cust_segments = {str(k): v for k, v in zip(df_c["customer_id"], df_c["segment"])}
    
    for _, sale in df_s.iterrows():
        cust_id = str(sale["customer_id"])
        # Find customer segment with safe default
        segment = cust_segments.get(cust_id, "SMB")
        
        # Apply elasticity
        e_coeff = elasticity.get(segment, -1.4)
        qty_change_pct = price_adjustment_pct * e_coeff
        
        # Calculate new quantity (cannot go below 0)
        new_qty = max(0.0, sale["quantity"] * (1.0 + qty_change_pct))
        
        # New unit price
        new_unit_price = sale["unit_price"] * (1.0 + price_adjustment_pct)
        
        # Keep same discount structure
        discount = sale["discount"]
        
        # New total amount
        new_total = round(new_qty * new_unit_price * (1.0 - discount), 2)
        
        simulated_sales.append({
            "product_id": str(sale["product_id"]),
            "qty": new_qty,
            "total": new_total
        })
        
    df_sim_sales = pd.DataFrame(simulated_sales)
    simulated_revenue = df_sim_sales["total"].sum()
    
    # Simulated COGS
    simulated_cogs = sum(row["qty"] * prod_costs.get(str(row["product_id"]), 0) for _, row in df_sim_sales.iterrows())
    
    # 3. Simulate Marketing Budget Spends & Revenues
    # Marketing increases customer acquisition and frequency
    # We model a diminishing log returns curve: Revenue multiplier = log(1 + spend_multiplier)
    m_multiplier = 1.0
    if marketing_spend_pct > 0.0:
        # Increasing budget increases sales volume
        # If budget grows by 100%, volume grows by log(2) * coefficient = 0.69 * 0.25 = ~17%
        m_multiplier = 1.0 + (np.log(1.0 + marketing_spend_pct) * 0.25)
    elif marketing_spend_pct < 0.0:
        # Decreasing marketing drops sales volume
        m_multiplier = max(0.4, 1.0 + (marketing_spend_pct * 0.40))
        
    simulated_revenue *= m_multiplier
    simulated_cogs *= m_multiplier
    
    # 4. Compile Financial Summaries
    df_f = df_financials.copy()
    df_f["date"] = pd.to_datetime(df_f["date"])
    df_f_last_year = df_f.sort_values(by="date").tail(12) # Use last 12 months as annual opex base
    
    base_payroll = df_f_last_year["payroll"].sum()
    base_rent = df_f_last_year["rent"].sum()
    base_opex = df_f_last_year["operating_expenses"].sum()
    base_marketing = df_f_last_year["marketing_spend"].sum()
    base_depreciation = df_f_last_year["depreciation"].sum()
    
    # Apply opex, payroll, and marketing spend sliders
    sim_payroll = base_payroll * (1.0 + payroll_adjustment_pct)
    sim_rent = base_rent # rent remains constant typically
    sim_opex = base_opex * (1.0 + opex_adjustment_pct)
    sim_marketing = base_marketing * (1.0 + marketing_spend_pct)
    
    # Calculate EBITDA, Profit, Taxes
    sim_gross_profit = simulated_revenue - simulated_cogs
    sim_ebitda = sim_gross_profit - (sim_payroll + sim_rent + sim_opex + sim_marketing)
    
    # Corporate tax 22%
    sim_taxes = max(0.0, (sim_ebitda - base_depreciation) * 0.22)
    sim_net_profit = sim_ebitda - base_depreciation - sim_taxes
    
    # Baseline comparison values (annualized)
    base_cogs = baseline_cogs
    base_gross_profit = baseline_revenue - base_cogs
    base_ebitda = base_gross_profit - (base_payroll + base_rent + base_opex + base_marketing)
    base_taxes = max(0.0, (base_ebitda - base_depreciation) * 0.22)
    base_net_profit = base_ebitda - base_depreciation - base_taxes
    
    return {
        "baseline": {
            "revenue": round(baseline_revenue, 2),
            "cogs": round(base_cogs, 2),
            "gross_profit": round(base_gross_profit, 2),
            "operating_expenses": round(base_payroll + base_rent + base_opex + base_marketing, 2),
            "ebitda": round(base_ebitda, 2),
            "taxes": round(base_taxes, 2),
            "net_profit": round(base_net_profit, 2),
            "margin_pct": round((base_net_profit / baseline_revenue) * 100, 2)
        },
        "simulated": {
            "revenue": round(simulated_revenue, 2),
            "cogs": round(simulated_cogs, 2),
            "gross_profit": round(sim_gross_profit, 2),
            "operating_expenses": round(sim_payroll + sim_rent + sim_opex + sim_marketing, 2),
            "ebitda": round(sim_ebitda, 2),
            "taxes": round(sim_taxes, 2),
            "net_profit": round(sim_net_profit, 2),
            "margin_pct": round((sim_net_profit / simulated_revenue) * 100, 2) if simulated_revenue > 0 else 0.0
        },
        "variance": {
            "revenue_pct": round(((simulated_revenue - baseline_revenue) / baseline_revenue) * 100, 2),
            "net_profit_pct": round(((sim_net_profit - base_net_profit) / base_net_profit) * 100, 2) if base_net_profit > 0 else 0.0,
            "margin_diff": round(((sim_net_profit / simulated_revenue) * 100) - ((base_net_profit / baseline_revenue) * 100), 2) if simulated_revenue > 0 and baseline_revenue > 0 else 0.0
        }
    }
