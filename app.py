# --- IMPORTS ---
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# --- APP CONFIG ---
st.set_page_config(page_title="Sales Forecasting Dashboard", page_icon="📈", layout="wide")

# --- CACHE VERSION (forces fresh cache when you change code/data shape) ---
CACHE_VERSION = "gb_v2_2025-09-02"

# --- CACHING HELPERS ---
@st.cache_data
def load_sales_data(cache_buster: str):
    """
    Load CSV and normalize column names to snake_case.
    Ensures consistent references: order_date, total_price, product_id, product_category
    """
    try:
        df = pd.read_csv('larger_sales_dataset.csv', encoding='ISO-8859-1')

        # Normalize column names: strip, collapse spaces, lower, underscores
        df.columns = (
            df.columns
              .str.strip()
              .str.replace(r"\s+", "_", regex=True)
              .str.lower()
        )

        required = ['order_date', 'total_price', 'product_id', 'product_category']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing required columns after normalization: {missing}")
            return None

        return df
    except FileNotFoundError:
        st.error("Sales dataset not found. Please ensure 'larger_sales_dataset.csv' is in the app directory.")
        return None
    except Exception as e:
        st.error(f"Error loading sales data: {e}")
        return None

@st.cache_data
def preprocess_data(df: pd.DataFrame, cache_buster: str):
    """Clean and prep the normalized dataframe."""
    df = df.copy()
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df.dropna(subset=['order_date', 'total_price'], inplace=True)
    df = df[df['total_price'] > 0]

    # Optional calendar features (kept for potential downstream analysis)
    df['month'] = df['order_date'].dt.month
    df['day_of_week'] = df['order_date'].dt.dayofweek
    df['week_of_year'] = df['order_date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['order_date'].dt.quarter

    df.sort_values('order_date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_data
def prepare_daily_sales(df: pd.DataFrame, cache_buster: str):
    """Aggregate to daily totals and fill missing dates with zeros."""
    daily = df.groupby(df['order_date'].dt.date).agg({
        'total_price': 'sum',
        'product_id': 'count'
    }).reset_index()

    daily.columns = ['order_date', 'total_price', 'order_count']
    daily['order_date'] = pd.to_datetime(daily['order_date'])

    # Fill date gaps
    idx = pd.date_range(daily['order_date'].min(), daily['order_date'].max(), freq='D')
    full = pd.DataFrame({'order_date': idx})
    daily = full.merge(daily, on='order_date', how='left').fillna(0)

    return daily

@st.cache_data
def train_improved_model(daily_sales: pd.DataFrame, cache_buster: str):
    """
    Gradient Boosting with time features + rolling means.
    Returns fitted model, the feature DF (for future gen), and metrics on a temporal holdout.
    """
    feats = daily_sales.copy()
    feats['day_index']   = np.arange(len(feats))
    feats['day_of_week'] = feats['order_date'].dt.dayofweek
    feats['month']       = feats['order_date'].dt.month
    feats['quarter']     = feats['order_date'].dt.quarter
    feats['rolling_7']   = feats['total_price'].rolling(window=7,  min_periods=1).mean()
    feats['rolling_30']  = feats['total_price'].rolling(window=30, min_periods=1).mean()

    feature_cols = ['day_index', 'day_of_week', 'month', 'quarter', 'rolling_7', 'rolling_30']
    X = feats[feature_cols].values
    y = feats['total_price'].values

    # Time-aware split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=False)

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'r2': r2_score(y_test, y_pred)
    }
    return model, feats, metrics

def generate_predictions(model, feats: pd.DataFrame, days: int = 30):
    """
    Roll forward features deterministically (keeps last rolling stats frozen).
    """
    last_date = feats['order_date'].iloc[-1]
    last_r7   = feats['total_price'].tail(7).mean()
    last_r30  = feats['total_price'].tail(30).mean()

    future_features = []
    for i in range(1, days + 1):
        d = last_date + timedelta(days=i)
        row = [
            len(feats) + i - 1,  # day_index forward
            d.dayofweek,
            d.month,
            d.quarter,
            last_r7,
            last_r30
        ]
        future_features.append(row)

    future_X   = np.array(future_features)
    preds      = model.predict(future_X)
    future_idx = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    return future_idx, preds

# --- MAIN ---
def main():
    st.title("📈 AI-Powered Sales Forecasting Dashboard")
    st.markdown("**Advanced predictive analytics for strategic business decisions**")

    # Load + prep
    df = load_sales_data(CACHE_VERSION)
    if df is None:
        st.stop()

    df_processed = preprocess_data(df, CACHE_VERSION)
    daily_sales  = prepare_daily_sales(df_processed, CACHE_VERSION)

    with st.spinner("Training AI model..."):
        model, features_df, metrics = train_improved_model(daily_sales, CACHE_VERSION)

    # Top KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Model R² Score", f"{metrics['r2']:.3f}")
    with c2: st.metric("Mean Absolute Error", f"${metrics['mae']:,.2f}")
    with c3: st.metric("RMSE", f"${metrics['rmse']:,.2f}")
    with c4: st.metric("Total Revenue", f"${df_processed['total_price'].sum():,.2f}")

    # Forecast horizon
    horizon = st.slider("Forecast Period (Days)", 7, 60, 30)
    future_dates, predictions = generate_predictions(model, features_df, horizon)

    # Forecast chart
    st.subheader("📊 Sales Forecast Analysis")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_sales['order_date'],
        y=daily_sales['total_price'],
        mode='lines',
        name='Historical Sales',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines',
        name='Predicted Sales',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    fig.update_layout(
        title="Sales Forecast: Historical vs Predicted",
        xaxis_title="Date",
        yaxis_title="Daily Sales ($)",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Insights
    st.subheader("💡 Key Insights")
    k1, k2 = st.columns(2)
    with k1:
        avg_hist = daily_sales['total_price'].mean()
        avg_pred = float(np.mean(predictions))
        growth   = ((avg_pred - avg_hist) / avg_hist) * 100 if avg_hist != 0 else 0.0
        st.metric("Average Daily Sales (Next 30 Days)", f"${avg_pred:,.2f}", f"{growth:+.1f}% vs Historical")
        st.metric(f"Total Forecasted Revenue ({horizon} days)", f"${float(np.sum(predictions)):,.2f}")

    with k2:
        max_day = future_dates[int(np.argmax(predictions))]
        min_day = future_dates[int(np.argmin(predictions))]
        st.write(f"**Highest Predicted Sales:** {max_day.strftime('%B %d, %Y')} - ${float(np.max(predictions)):,.2f}")
        st.write(f"**Lowest Predicted Sales:** {min_day.strftime('%B %d, %Y')} - ${float(np.min(predictions)):,.2f}")
        st.write(f"**Prediction Volatility (σ):** ${float(np.std(predictions)):.2f}")

    # Top Categories (now robust to naming & cache)
    st.subheader("🏆 Top Performing Categories")
    if 'product_category' in df_processed.columns:
        top_cat = (
            df_processed.groupby('product_category')['total_price']
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        chart_df = pd.DataFrame({'Category': top_cat.index, 'Revenue': top_cat.values})
        fig_cat = px.bar(
            chart_df,
            x='Category',
            y='Revenue',
            title='Top 10 Categories by Revenue',
            color='Revenue',
            color_continuous_scale='viridis'
        )
        fig_cat.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.error("`product_category` column not found after normalization.")

if __name__ == "__main__":
    main()
