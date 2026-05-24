import numpy as np
import pandas as pd
from itertools import combinations
import streamlit as st

@st.cache_data(show_spinner="Extracting shopping baskets...")
def extract_shopping_baskets(df_sales, df_products):
    """
    Groups individual transactions by (customer_id, sale_date) to form shopping baskets.
    Maps product_ids to names for readable analytical results.
    """
    df = df_sales.copy()
    
    # Map product names
    # Coerce product_id to string to avoid float/int type mismatch
    prod_map = {str(k): v for k, v in zip(df_products["product_id"], df_products["name"])}
    df["product_name"] = df["product_id"].astype(str).map(prod_map)
    
    # Group by customer and date to represent a single shopping basket event
    baskets = df.groupby(["customer_id", "sale_date"])["product_name"].apply(set).reset_index()
    return baskets["product_name"].tolist()

@st.cache_data(show_spinner="Compiling Market Basket association rules...")
def calculate_association_rules(baskets, min_support=0.01, min_confidence=0.05):
    """
    An elegant, high-performance custom implementation of the Apriori Association Rules algorithm
    using pure pandas/numpy. Avoids external compilation issues while remaining fully transparent.
    Calculates Support, Confidence, and Lift for all product pairs (A -> B).
    """
    total_transactions = len(baskets)
    if total_transactions == 0:
        return pd.DataFrame()
        
    # 1. Count individual item frequencies
    item_counts = {}
    for basket in baskets:
        for item in basket:
            item_counts[item] = item_counts.get(item, 0) + 1
            
    # Calculate single item supports
    item_supports = {item: count / total_transactions for item, count in item_counts.items()}
    
    # Filter items by minimum support
    frequent_items = {item for item, sup in item_supports.items() if sup >= min_support}
    
    # 2. Count pair frequencies
    pair_counts = {}
    for basket in baskets:
        # Keep only frequent items in basket
        filtered_basket = [item for item in basket if item in frequent_items]
        if len(filtered_basket) >= 2:
            # Generate all combinations of pairs (sorted to avoid duplicates)
            for combo in combinations(sorted(filtered_basket), 2):
                pair_counts[combo] = pair_counts.get(combo, 0) + 1
                
    # Calculate pair metrics and compile rules
    rules_list = []
    
    for (item_A, item_B), count in pair_counts.items():
        support_AB = count / total_transactions
        
        if support_AB >= min_support:
            support_A = item_supports[item_A]
            support_B = item_supports[item_B]
            
            # Rule 1: A -> B
            conf_A_to_B = support_AB / support_A
            lift_A_to_B = conf_A_to_B / support_B
            
            if conf_A_to_B >= min_confidence:
                rules_list.append({
                    "antecedent": item_A,
                    "consequent": item_B,
                    "support_A": round(support_A, 4),
                    "support_B": round(support_B, 4),
                    "support_AB": round(support_AB, 4),
                    "confidence": round(conf_A_to_B, 4),
                    "lift": round(lift_A_to_B, 3)
                })
                
            # Rule 2: B -> A
            conf_B_to_A = support_AB / support_B
            lift_B_to_A = conf_B_to_A / support_A
            
            if conf_B_to_A >= min_confidence:
                rules_list.append({
                    "antecedent": item_B,
                    "consequent": item_A,
                    "support_A": round(support_B, 4),
                    "support_B": round(support_A, 4),
                    "support_AB": round(support_AB, 4),
                    "confidence": round(conf_B_to_A, 4),
                    "lift": round(lift_B_to_A, 3)
                })
                
    df_rules = pd.DataFrame(rules_list)
    if not df_rules.empty:
        df_rules = df_rules.sort_values(by="lift", ascending=False).reset_index(drop=True)
        
    return df_rules

def recommend_cross_sell_products(df_rules, antecedent_product_name, top_n=3):
    """
    Searches the calculated rules for products frequently purchased together with the input product.
    Returns the top associated products sorted by Lift and Confidence.
    """
    if df_rules.empty:
        return []
        
    # Filter rules where antecedent is the product in question
    recommendations = df_rules[df_rules["antecedent"] == antecedent_product_name].copy()
    
    if recommendations.empty:
        # Fall back to recommending products with high lift overall or empty list
        return []
        
    recommendations = recommendations.sort_values(by=["lift", "confidence"], ascending=[False, False])
    
    recs = []
    for _, row in recommendations.head(top_n).iterrows():
        recs.append({
            "product": row["consequent"],
            "support": row["support_AB"],
            "confidence_pct": round(row["confidence"] * 100, 1),
            "lift_score": row["lift"]
        })
        
    return recs
