import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st

@st.cache_data(show_spinner=False)
def prepare_time_series_data(df_sales):
    """
    Groups sales transactions by month and returns a cleaned, continuous monthly sales series.
    """
    df = df_sales.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    
    # Resample to monthly sums
    df_monthly = df.set_index("sale_date").resample("MS")["total_amount"].sum().reset_index()
    df_monthly.columns = ["date", "sales"]
    return df_monthly

@st.cache_data(show_spinner="Fitting business time-series models...")
def train_holt_winters_forecast(df_monthly, forecast_months=6):
    """
    Fits a triple Exponential Smoothing (Holt-Winters) model with additive trend and seasonality.
    Projects future monthly sales with error metrics.
    """
    series = df_monthly.set_index("date")["sales"]
    
    # Need at least 2 full seasonal cycles (24 months) for Holt-Winters additive seasonal model
    if len(series) < 24:
        # Fall back to double/single smoothing if data is short
        model = ExponentialSmoothing(series, trend="add", seasonal=None)
    else:
        model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=12)
        
    fitted_model = model.fit()
    
    # In-sample fit
    in_sample_pred = fitted_model.fittedvalues
    
    # Calculate metrics based on in-sample fit
    mae = mean_absolute_error(series, in_sample_pred)
    rmse = np.sqrt(mean_squared_error(series, in_sample_pred))
    r2 = r2_score(series, in_sample_pred)
    
    # Future forecast
    future_dates = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=forecast_months, freq="MS")
    forecast_values = fitted_model.forecast(forecast_months)
    
    df_forecast = pd.DataFrame({
        "date": future_dates,
        "forecast": forecast_values.values
    })
    
    return in_sample_pred.values, df_forecast, {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 3)}

@st.cache_data(show_spinner="Training ML regressor forecasters...")
def train_ml_regressor_forecast(df_monthly, forecast_months=6, n_estimators=100, max_depth=6):
    """
    Transforms time series into a supervised ML problem using lag features, rolling windows,
    and cyclical time variables. Trains a Random Forest regressor, displays feature importances,
    and recursively forecasts the future horizon.
    """
    df = df_monthly.copy()
    
    # Create lag features (1, 2, 3 months back)
    df["lag_1"] = df["sales"].shift(1)
    df["lag_2"] = df["sales"].shift(2)
    df["lag_3"] = df["sales"].shift(3)
    df["roll_mean_3"] = df["sales"].shift(1).rolling(3).mean()
    
    # Cyclical date features
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    
    # Drop rows with NaN due to lags
    df_ml = df.dropna().reset_index(drop=True)
    
    X = df_ml[["lag_1", "lag_2", "lag_3", "roll_mean_3", "month_sin", "month_cos"]]
    y = df_ml["sales"]
    
    # Fit RandomForest
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X, y)
    
    # In-sample predictions (backcast)
    in_sample_pred = model.predict(X)
    
    # Pad first 3 values where lags were unavailable for a fair visual overlay
    in_sample_padded = np.concatenate([np.full(3, np.nan), in_sample_pred])
    
    # Calculate metrics on valid regression rows
    mae = mean_absolute_error(y, in_sample_pred)
    rmse = np.sqrt(mean_squared_error(y, in_sample_pred))
    r2 = r2_score(y, in_sample_pred)
    
    # Feature Importances
    importances = model.feature_importances_
    feat_names = X.columns
    df_feats = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
    
    # Recursive Multi-Step Out-of-Sample Forecasting
    last_known_sales = list(df_monthly["sales"].values[-3:]) # Last 3 months
    forecast_values = []
    
    current_date = df_monthly["date"].iloc[-1]
    future_dates = []
    
    for i in range(forecast_months):
        current_date = current_date + pd.DateOffset(months=1)
        future_dates.append(current_date)
        
        # Assemble feature vector
        lag_1 = last_known_sales[-1]
        lag_2 = last_known_sales[-2]
        lag_3 = last_known_sales[-3]
        roll_mean_3 = np.mean(last_known_sales[-3:])
        month_sin = np.sin(2 * np.pi * current_date.month / 12)
        month_cos = np.cos(2 * np.pi * current_date.month / 12)
        
        feat_vector = np.array([[lag_1, lag_2, lag_3, roll_mean_3, month_sin, month_cos]])
        pred_sales = model.predict(feat_vector)[0]
        
        # Append predictions
        forecast_values.append(pred_sales)
        last_known_sales.append(pred_sales)
        
    df_forecast = pd.DataFrame({
        "date": future_dates,
        "forecast": forecast_values
    })
    
    return in_sample_padded, df_forecast, df_feats, {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 3)}

def plot_forecast_plotly(df_monthly, in_sample_pred, df_forecast, model_name="BusinessPulse Forecasting Model"):
    """
    Generates a gorgeous Plotly chart mapping historical monthly sales,
    in-sample fitted predictions, and forecasted future values.
    """
    fig = go.Figure()
    
    # 1. Historical Actual Sales
    fig.add_trace(go.Scatter(
        x=df_monthly["date"],
        y=df_monthly["sales"],
        mode="lines+markers",
        name="Historical Revenue",
        line=dict(color="#352e8c", width=3),
        marker=dict(size=6)
    ))
    
    # 2. In-sample model fitting
    # Overlay dates for fitted values (match length of in-sample array)
    fit_dates = df_monthly["date"].values
    fig.add_trace(go.Scatter(
        x=fit_dates,
        y=in_sample_pred,
        mode="lines",
        name="Model In-Sample Fit",
        line=dict(color="#ff5e7e", width=2, dash="dash")
    ))
    
    # 3. Future Projected Forecast
    # Append the last actual sales data point to the forecast series to make the line continuous on chart
    last_actual_date = df_monthly["date"].iloc[-1]
    last_actual_sales = df_monthly["sales"].iloc[-1]
    
    fc_dates = [last_actual_date] + list(df_forecast["date"].values)
    fc_vals = [last_actual_sales] + list(df_forecast["forecast"].values)
    
    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=fc_vals,
        mode="lines+markers",
        name="6-Month AI Projection",
        line=dict(color="#10b981", width=3),
        marker=dict(size=6, symbol="diamond")
    ))
    
    # Layout and glassmorphic aesthetics styling
    fig.update_layout(
        title=dict(text=f"{model_name} Sales Projections", font=dict(color="#1e1b4b", size=16, family="Outfit")),
        xaxis_title="Timeline",
        yaxis_title="Monthly Revenue ($)",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,255,0.6)",
        font=dict(color="#1e1b4b", family="Inter", size=13),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#1e1b4b", size=12),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(30,27,75,0.15)",
            borderwidth=1
        ),
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
        margin=dict(l=40, r=40, t=80, b=40),
        hovermode="x unified"
    )
    
    return fig
