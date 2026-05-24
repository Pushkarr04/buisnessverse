import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

@st.cache_data(show_spinner=False)
def calculate_rfm_metrics(df_customers, df_sales):
    """
    Computes RFM (Recency, Frequency, Monetary) data per customer based on historical sales.
    """
    df_s = df_sales.copy()
    df_c = df_customers.copy()
    df_s["customer_id"] = df_s["customer_id"].astype(str)
    df_c["customer_id"] = df_c["customer_id"].astype(str)
    
    df_s["sale_date"] = pd.to_datetime(df_s["sale_date"])
    
    max_db_date = df_s["sale_date"].max()
    
    # Aggregate transaction levels
    rfm = df_s.groupby("customer_id").agg(
        recency=("sale_date", lambda x: (max_db_date - x.max()).days),
        frequency=("sale_id", "count"),
        monetary=("total_amount", "sum")
    ).reset_index()
    
    # Merge with original customer lists to preserve demographic columns if needed
    df_rfm = pd.merge(df_c[["customer_id", "name", "segment", "region"]], rfm, on="customer_id", how="inner")
    
    return df_rfm

@st.cache_data(show_spinner="Evaluating cluster elbow optimizations...")
def calculate_elbow_wcss(df_rfm, max_clusters=8):
    """
    Computes the Within-Cluster Sum of Squares (WCSS) for multiple clusters to draw an Elbow curve.
    """
    features = df_rfm[["recency", "frequency", "monetary"]]
    
    # Scale features
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(features)
    
    wcss = []
    cluster_range = range(1, max_clusters + 1)
    
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(scaled_feats)
        wcss.append(kmeans.inertia_)
        
    return list(cluster_range), wcss

def plot_elbow_curve_plotly(cluster_range, wcss):
    """
    Renders an elegant Plotly line plot for Elbow visual optimization.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cluster_range,
        y=wcss,
        mode="lines+markers",
        line=dict(color="#5ac8fa", width=3),
        marker=dict(size=8, color="#352e8c")
    ))
    
    fig.update_layout(
        title=dict(text="K-Means Cluster Optimization (Elbow Method)", font=dict(color="#1e1b4b", size=16, family="Outfit")),
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Within-Cluster Sum of Squares (WCSS / Inertia)",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,255,0.6)",
        font=dict(color="#1e1b4b", family="Inter", size=13),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(30, 27, 75, 0.1)",
            linecolor="rgba(30, 27, 75, 0.25)", zeroline=False,
            tickfont=dict(color="#2d2a6e", size=12, family="Inter"),
            title_font=dict(color="#1e1b4b", size=13, family="Outfit")
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(30, 27, 75, 0.1)",
            linecolor="rgba(30, 27, 75, 0.25)", zeroline=False,
            tickfont=dict(color="#2d2a6e", size=12, family="Inter"),
            title_font=dict(color="#1e1b4b", size=13, family="Outfit")
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

@st.cache_data(show_spinner="Fitting K-Means segmentation clusters...")
def train_kmeans_segmentation(df_rfm, n_clusters=4):
    """
    Standardizes RFM scales, fits K-Means clustering, assigns labels,
    calculates cluster stats, and auto-profiles segments with relatable personas.
    """
    df = df_rfm.copy()
    features = df[["recency", "frequency", "monetary"]]
    
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    df["cluster"] = kmeans.fit_predict(scaled_feats)
    
    # Compute centroids and statistics
    cluster_stats = df.groupby("cluster").agg(
        size=("customer_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean")
    ).reset_index()
    
    # Auto-map segments based on centroid relationships
    # 1. High monetary & frequency with low recency = Champions
    # 2. High recency with low frequency/monetary = Churn Risk / Hibernating
    # 3. Moderate levels = Loyalists / Mid-Tier
    # 4. Low recency, low frequency, low monetary = New Signup / Potential Spender
    
    cluster_stats = cluster_stats.sort_values(by="avg_monetary", ascending=False).reset_index(drop=True)
    
    persona_mapping = {}
    personas = [
        "Champions (High Value, Active)",
        "Core Loyalists (Frequent, Moderate Value)",
        "Potential Spenders (New, Active)",
        "At Risk (Infrequent, High Recency)",
        "Hibernating (Inactive, Low Value)",
        "Sparsely Engaged"
    ]
    
    for i, row in cluster_stats.iterrows():
        c_id = row["cluster"]
        p_name = personas[min(i, len(personas)-1)]
        persona_mapping[c_id] = p_name
        
    df["segment_profile"] = df["cluster"].map(persona_mapping)
    cluster_stats["segment_profile"] = cluster_stats["cluster"].map(persona_mapping)
    
    # Make stats prettier
    cluster_stats["avg_recency"] = cluster_stats["avg_recency"].round(1)
    cluster_stats["avg_frequency"] = cluster_stats["avg_frequency"].round(1)
    cluster_stats["avg_monetary"] = cluster_stats["avg_monetary"].round(2)
    
    return df, cluster_stats

def plot_3d_clusters_plotly(df_clustered):
    """
    Generates a stunning interactive 3D Scatter plot mapping customers by RFM axes
    colored dynamically by cluster assignment.
    """
    # Clip extreme outlier monetary values for better graph readability
    df_plot = df_clustered.copy()
    m_cap = df_plot["monetary"].quantile(0.98)
    df_plot["monetary_trimmed"] = df_plot["monetary"].clip(upper=m_cap)
    
    fig = px.scatter_3d(
        df_plot,
        x="recency",
        y="frequency",
        z="monetary_trimmed",
        color="segment_profile",
        hover_name="name",
        hover_data={"recency": True, "frequency": True, "monetary": ":.2f", "segment_profile": False},
        color_discrete_sequence=["#352e8c", "#5ac8fa", "#ff5e7e", "#10b981", "#ff9f1c", "#ef4444"],
        labels={"recency": "Recency (Days)", "frequency": "Frequency (Orders)", "monetary_trimmed": "Monetary ($)"}
    )
    
    fig.update_layout(
        title=dict(text="Interactive 3D Customer RFM Space", font=dict(color="#1e1b4b", size=16, family="Outfit")),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e1b4b", family="Inter", size=13),
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(248,249,255,0.6)", gridcolor="rgba(30, 27, 75, 0.15)",
                showbackground=True, tickfont=dict(color="#2d2a6e", size=11),
                title_font=dict(color="#1e1b4b", size=12)
            ),
            yaxis=dict(
                backgroundcolor="rgba(248,249,255,0.6)", gridcolor="rgba(30, 27, 75, 0.15)",
                showbackground=True, tickfont=dict(color="#2d2a6e", size=11),
                title_font=dict(color="#1e1b4b", size=12)
            ),
            zaxis=dict(
                backgroundcolor="rgba(248,249,255,0.6)", gridcolor="rgba(30, 27, 75, 0.15)",
                showbackground=True, tickfont=dict(color="#2d2a6e", size=11),
                title_font=dict(color="#1e1b4b", size=12)
            )
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.9, xanchor="center", x=0.5,
            font=dict(color="#1e1b4b", size=12),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(30,27,75,0.15)",
            borderwidth=1
        )
    )
    return fig
