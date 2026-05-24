import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import streamlit as st

@st.cache_data(show_spinner=False)
def prepare_churn_features(df_customers, df_sales):
    """
    Integrates customer demo stats with transaction metrics to engineer churn predictor features:
    RFM factors, tenure length, satisfaction score, segment/region dummy indexes.
    """
    df_c = df_customers.copy()
    df_s = df_sales.copy()
    
    # Parse dates
    df_s["sale_date"] = pd.to_datetime(df_s["sale_date"])
    df_c["signup_date"] = pd.to_datetime(df_c["signup_date"])
    
    # Baseline benchmark date (max date in database)
    max_db_date = df_s["sale_date"].max()
    
    # Calculate RFM per customer
    df_s["customer_id"] = df_s["customer_id"].astype(str)
    df_c["customer_id"] = df_c["customer_id"].astype(str)
    
    customer_rfm = df_s.groupby("customer_id").agg(
        total_spend=("total_amount", "sum"),
        order_count=("sale_id", "count"),
        last_purchase_date=("sale_date", "max")
    ).reset_index()
    
    customer_rfm["recency"] = (max_db_date - customer_rfm["last_purchase_date"]).dt.days
    
    # Merge RFM with customer records
    df_features = pd.merge(df_c, customer_rfm, on="customer_id", how="left")
    
    # Fill missing values for customers without transactions (safety check)
    df_features["total_spend"] = df_features["total_spend"].fillna(0)
    df_features["order_count"] = df_features["order_count"].fillna(0)
    df_features["recency"] = df_features["recency"].fillna((max_db_date - df_features["signup_date"]).dt.days)
    
    # Calculate customer tenure in days
    df_features["tenure"] = (max_db_date - df_features["signup_date"]).dt.days
    df_features["tenure"] = df_features["tenure"].apply(lambda x: max(1, x))
    
    # Select feature columns and label
    feature_cols = [
        "age", "satisfaction_score", "total_spend", "order_count", "recency", "tenure",
        "segment", "region", "gender", "churn_status"
    ]
    
    return df_features[feature_cols]

@st.cache_resource(show_spinner="Training predictive churn models...")
def train_churn_classifier(df_features, model_type="Random Forest", n_estimators=100, max_depth=6, learning_rate=0.1):
    """
    Performs data hot-encoding, splits train-test pools, fits chosen classifier,
    and returns metrics, confusion matrix, ROC data, and feature importances.
    """
    df = df_features.copy()
    
    # Hot-Encode categorical fields
    cat_cols = ["segment", "region", "gender"]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Isolate targets and features
    X = df_encoded.drop(columns=["churn_status"])
    y = df_encoded["churn_status"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Fit selected model
    if model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob_all = model.predict_proba(X_test)
    if y_prob_all.shape[1] > 1:
        y_prob = y_prob_all[:, 1]
    else:
        y_prob = y_prob_all[:, 0]
    
    # Basic Accuracy metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0)
    }
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC Curve points
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Feature Importances
    importances = model.feature_importances_
    feat_names = X.columns
    df_feats = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
    
    # Save training metadata in model to map columns later for custom prediction
    model.trained_cols = list(X.columns)
    model.cat_mappings = {
        "segment": [c for c in X.columns if "segment_" in c],
        "region": [c for c in X.columns if "region_" in c],
        "gender": [c for c in X.columns if "gender_" in c]
    }
    
    return model, X_test, y_test, y_pred, y_prob, metrics, cm, fpr, tpr, roc_auc, df_feats

def plot_confusion_matrix_plotly(cm):
    """
    Renders an elegant glassmorphism confusion matrix heatmap.
    """
    z = cm.tolist()
    x = ["Predicted Active", "Predicted Churned"]
    y = ["Actual Active", "Actual Churned"]
    
    fig = px.imshow(
        z, x=x, y=y, 
        color_continuous_scale="Blues",
        aspect="auto",
        text_auto=True
    )
    
    fig.update_layout(
        title=dict(text="Model Confusion Matrix Heatmap", font=dict(color="#1e1b4b", size=16, family="Outfit")),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,255,0.6)",
        font=dict(color="#1e1b4b", family="Inter", size=13),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig
 
def plot_roc_curve_plotly(fpr, tpr, roc_auc):
    """
    Renders an interactive ROC Curve visual.
    """
    fig = go.Figure()
    
    # Diagonal baseline (random guess)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Guess",
        line=dict(color="rgba(15, 23, 42, 0.2)", width=2, dash="dash")
    ))
    
    # Model ROC
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"Model ROC (AUC = {roc_auc:.3f})",
        line=dict(color="#352e8c", width=3)
    ))
    
    fig.update_layout(
        title=dict(text="Receiver Operating Characteristic (ROC)", font=dict(color="#1e1b4b", size=16, family="Outfit")),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
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
        legend=dict(
            yanchor="bottom", y=0.02, xanchor="right", x=0.98,
            font=dict(color="#1e1b4b", size=12),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(30,27,75,0.15)",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def predict_custom_churn(model, customer_data: dict):
    """
    Accepts user dictionary inputs, hot-encodes categoricals to match the training schema,
    and runs the classifier to yield churn probability and threat categorization.
    """
    # Create single-row pandas DataFrame
    df_raw = pd.DataFrame([customer_data])
    
    # Initialize encoded vector matching exactly the trained model features
    features_vector = {}
    for col in model.trained_cols:
        features_vector[col] = 0.0
        
    # Set numeric columns directly
    numerics = ["age", "satisfaction_score", "total_spend", "order_count", "recency", "tenure"]
    for col in numerics:
        if col in df_raw:
            features_vector[col] = float(df_raw[col].iloc[0])
            
    # Set dummy codes based on input categoricals
    segment_val = df_raw["segment"].iloc[0]
    region_val = df_raw["region"].iloc[0]
    gender_val = df_raw["gender"].iloc[0]
    
    segment_col = f"segment_{segment_val}"
    region_col = f"region_{region_val}"
    gender_col = f"gender_{gender_val}"
    
    if segment_col in features_vector:
        features_vector[segment_col] = 1.0
    if region_col in features_vector:
        features_vector[region_col] = 1.0
    if gender_col in features_vector:
        features_vector[gender_col] = 1.0
        
    df_pred = pd.DataFrame([features_vector])
    
    # Reorder columns to match original trained columns exactly
    df_pred = df_pred[model.trained_cols]
    
    churn_prob_all = model.predict_proba(df_pred)[0]
    if len(churn_prob_all) > 1:
        churn_prob = churn_prob_all[1]
    else:
        churn_prob = churn_prob_all[0]
    
    # Determine risk category
    if churn_prob >= 0.75:
        risk_level = "Critical Churn Threat"
        risk_color = "red"
    elif churn_prob >= 0.40:
        risk_level = "Elevated Risk"
        risk_color = "orange"
    else:
        risk_level = "Healthy Profile"
        risk_color = "green"
        
    return {
        "churn_probability": round(churn_prob * 100, 2),
        "risk_level": risk_level,
        "color": risk_color
    }
