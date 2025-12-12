"""
Sales Forecasting Decision Support System
Prediksi penjualan, analisis produk, dan wawasan musiman tanpa modul stok
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')
import pickle
import joblib

# Time series models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Page configuration
st.set_page_config(
    page_title="DSS - Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1f77b4 0%, #2ecc71 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'forecasts_generated' not in st.session_state:
    st.session_state.forecasts_generated = False
if 'trained_models_loaded' not in st.session_state:
    st.session_state.trained_models_loaded = False

def _models_file_signature(models_dir: Path) -> Optional[tuple]:
    """Return tuple of mtimes to bust cache when models are retrained."""
    model_files = [
        models_dir / 'arima_model.pkl',
        models_dir / 'es_model.pkl',
        models_dir / 'ma_stats.pkl',
        models_dir / 'model_metadata.pkl',
    ]
    if not all(p.exists() for p in model_files):
        return None
    try:
        return tuple(p.stat().st_mtime for p in model_files)
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def load_trained_models(file_signature: Optional[tuple]):
    """Load pre-trained models from notebook; cache invalidates when retrained."""
    try:
        models_dir = Path(__file__).resolve().parent / 'models'

        if file_signature is None or not models_dir.exists():
            return None, None, None, None, False

        arima_model_path = models_dir / 'arima_model.pkl'
        es_model_path = models_dir / 'es_model.pkl'
        ma_stats_path = models_dir / 'ma_stats.pkl'
        metadata_path = models_dir / 'model_metadata.pkl'

        arima_model = joblib.load(arima_model_path)
        es_model = joblib.load(es_model_path)

        with open(ma_stats_path, 'rb') as f:
            ma_stats = pickle.load(f)

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        return arima_model, es_model, ma_stats, metadata, True

    except Exception as e:
        st.warning(f"⚠️ Error loading trained models: {str(e)}")
        return None, None, None, None, False


# Load models on app start
models_dir = Path(__file__).resolve().parent / 'models'
file_signature = _models_file_signature(models_dir)
arima_model, es_model, ma_stats, model_metadata, models_loaded = load_trained_models(file_signature)

@st.cache_data
def load_data(file):
    """Load and preprocess sales data (mirrors mining.ipynb cleaning steps)."""
    df = pd.read_csv(file)

    initial_rows = len(df)
    cleaning_log = {}

    df['IsCancelled'] = df['TransactionNo'].astype(str).str.startswith('C')
    cancelled_count = int(df['IsCancelled'].sum())
    df = df[~df['IsCancelled']].copy()
    df = df.drop('IsCancelled', axis=1)
    cleaning_log['Cancelled transactions removed'] = cancelled_count

    negative_qty = int((df['Quantity'] < 0).sum())
    negative_price = int((df['Price'] < 0).sum())
    if negative_qty > 0 or negative_price > 0:
        df = df[(df['Quantity'] > 0) & (df['Price'] > 0)].copy()
    cleaning_log['Negative qty/price removed'] = negative_qty + negative_price

    key_columns = ['Date', 'ProductNo', 'Price', 'Quantity']
    missing_before = int(df[key_columns].isnull().sum().sum())
    if missing_before > 0:
        df = df.dropna(subset=key_columns)
    cleaning_log['Rows with missing key fields removed'] = missing_before

    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    except Exception:
        df['Date'] = pd.to_datetime(df['Date'])

    df['Revenue'] = df['Price'] * df['Quantity']
    df = df.sort_values('Date').reset_index(drop=True)

    final_rows = len(df)
    cleaning_log['Final clean rows'] = final_rows
    cleaning_log['Total rows removed'] = initial_rows - final_rows
    cleaning_log['Data retention rate (%)'] = round((final_rows / initial_rows) * 100, 2) if initial_rows > 0 else 0

    # Persist cleaning details for UI
    st.session_state.cleaning_log = cleaning_log
    return df

@st.cache_data
def prepare_time_series(df):
    """Prepare daily aggregated time series"""
    daily_sales = df.groupby('Date').agg({
        'TransactionNo': 'count',
        'Quantity': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    daily_sales.columns = ['Date', 'NumTransactions', 'TotalQuantity', 'TotalRevenue']
    daily_sales = daily_sales.sort_values('Date').reset_index(drop=True)
    
    ts_data = daily_sales.set_index('Date')
    ts_data = ts_data.asfreq('D', method='ffill')
    return ts_data

@st.cache_data
def calculate_product_stats(df):
    """Calculate product-level statistics and ABC classification"""
    product_sales = df.groupby(['ProductNo', 'ProductName']).agg({
        'TransactionNo': 'count',
        'Quantity': 'sum',
        'Revenue': 'sum',
        'Price': 'mean'
    }).reset_index()
    product_sales.columns = ['ProductNo', 'ProductName', 'NumTransactions', 
                            'TotalQuantity', 'TotalRevenue', 'AvgPrice']
    product_sales = product_sales.sort_values('TotalRevenue', ascending=False).reset_index(drop=True)
    
    product_sales['RevenuePct'] = (product_sales['TotalRevenue'] / 
                                   product_sales['TotalRevenue'].sum() * 100)
    product_sales['CumulativePct'] = product_sales['RevenuePct'].cumsum()
    
    def classify_abc(cum_pct):
        if cum_pct <= 80:
            return 'A - Best Sellers (Top 80%)'
        elif cum_pct <= 95:
            return 'B - Moderate Sellers'
        return 'C - Slow Movers'
    
    product_sales['Category'] = product_sales['CumulativePct'].apply(classify_abc)
    return product_sales

def train_arima_model(train_data):
    """Train ARIMA model with dynamic order selection"""
    arima_configs = [
        (1, 1, 1), (2, 1, 2), (1, 1, 2), (2, 1, 1),
        (3, 1, 2), (1, 0, 1), (2, 0, 2)
    ]
    
    best_aic = np.inf
    best_model = None
    best_order = None
    
    for order in arima_configs:
        try:
            model = ARIMA(train_data, order=order).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_model = model
                best_order = order
        except:
            continue
    return best_model, best_order

def train_exponential_smoothing(train_data):
    """Train Exponential Smoothing model"""
    try:
        model = ExponentialSmoothing(
            train_data,
            seasonal_periods=7,
            trend='add',
            seasonal='add',
            initialization_method='estimated'
        )
        model_fit = model.fit(optimized=True)
    except:
        model = ExponentialSmoothing(train_data, trend='add', seasonal=None)
        model_fit = model.fit(optimized=True)
    return model_fit

def train_moving_average(train_data, window_size=7):
    """Train Moving Average model"""
    ma_values = train_data.rolling(window=window_size).mean()
    return ma_values.iloc[-window_size:].mean() if len(ma_values) >= window_size else train_data.mean()

# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown('<div class="main-header">📈 Decision Support System - Sales Forecasting</div>', 
            unsafe_allow_html=True)

st.markdown("""
### 🎯 Tujuan Sistem
Sistem ini membantu manajer dalam:
- **Prediksi Penjualan**: Forecast revenue harian/30 hari ke depan
- **Analisis Produk**: Identifikasi best sellers dan slow movers (ABC)
- **Analisis Musiman**: Pola mingguan dan weekday vs weekend
- **Strategi Promosi**: Rekomendasi hari dan kategori fokus
""")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    uploaded_file = st.file_uploader("Upload Data Penjualan (CSV)", type=['csv'])
    if uploaded_file is not None:
        st.session_state.data_loaded = True
    
    st.markdown("---")
    st.subheader("📊 Parameter Forecasting")
    forecast_days = st.slider("Periode Prediksi (hari)", 7, 90, 30)
    
    st.markdown("---")
    st.subheader("🤖 Pilih Model")
    model_choice = st.selectbox("Model Forecasting", 
                                ['Moving Average', 'ARIMA', 'Exponential Smoothing', 'Auto (Best Model)'])

# Main content
if not st.session_state.data_loaded:
    st.info("👈 Silakan upload file CSV data penjualan untuk memulai analisis")
    
    # Show sample data format
    st.subheader("📋 Format Data yang Dibutuhkan")
    sample_data = pd.DataFrame({
        'TransactionNo': ['536365', '536366', 'C536367'],
        'Date': ['12/1/2010', '12/1/2010', '12/1/2010'],
        'ProductNo': ['85123A', '71053', '84406B'],
        'ProductName': ['WHITE HANGING HEART', 'WHITE METAL LANTERN', 'CREAM CUPID HEARTS'],
        'Price': [2.55, 3.39, 2.75],
        'Quantity': [6, 6, 8],
        'CustomerNo': ['17850', '17850', '17850'],
        'Country': ['United Kingdom', 'United Kingdom', 'United Kingdom']
    })
    st.dataframe(sample_data)
    
else:
    # Load data
    df = load_data(uploaded_file)
    ts_data = prepare_time_series(df)
    product_stats = calculate_product_stats(df)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Dashboard", 
        "🔮 Forecasting", 
        "📦 Analisis Produk",
        "💡 Rekomendasi Bisnis",
        "📊 Analisis Detail"
    ])
    
    # ========================================================================
    # TAB 1: DASHBOARD
    # ========================================================================
    with tab1:
        st.header("📈 Overview Bisnis")
        
        with st.expander("📋 Data Quality Report", expanded=True):
            if hasattr(st.session_state, 'cleaning_log') and st.session_state.cleaning_log:
                log = st.session_state.cleaning_log
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Final Clean Rows", f"{log.get('Final clean rows', 0):,}")
                with col2:
                    st.metric("Data Retention", f"{log.get('Data retention rate (%)', 0):.1f}%")
                with col3:
                    st.metric("Total Removed", f"{log.get('Total rows removed', 0):,}")
                with col4:
                    st.metric("Date Range", f"{df['Date'].min().date()} - {df['Date'].max().date()}")
                
                st.markdown("**Cleaning Details:**")
                cleaning_details = {k: v for k, v in log.items() 
                        if k not in ['Final clean rows', 'Data retention rate (%)', 'Total rows removed']}
                for action, count in cleaning_details.items():
                    if count > 0:
                        st.warning(f"🗑️ {action}: {count:,}")
                
                st.markdown("**Data Validation:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    missing = df.isnull().sum().sum()
                    if missing == 0:
                        st.success("✅ No missing values")
                    else:
                        st.error(f"❌ {missing} missing values found")
                
                with col2:
                    dupes = df.duplicated().sum()
                    if dupes == 0:
                        st.success("✅ No duplicates")
                    else:
                        st.error(f"❌ {dupes} duplicates found")
                
                st.markdown("**Revenue Statistics:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min Revenue", f"£{df['Revenue'].min():,.2f}")
                with col2:
                    st.metric("Avg Revenue", f"£{df['Revenue'].mean():,.2f}")
                with col3:
                    st.metric("Max Revenue", f"£{df['Revenue'].max():,.2f}")
        
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        total_revenue = df['Revenue'].sum()
        total_transactions = df['TransactionNo'].nunique()
        total_products = df['ProductNo'].nunique()
        avg_daily_revenue = ts_data['TotalRevenue'].mean()
        
        with col1:
            st.metric("Total Revenue", f"£{total_revenue:,.0f}")
        with col2:
            st.metric("Total Transaksi", f"{total_transactions:,}")
        with col3:
            st.metric("Jumlah Produk", f"{total_products:,}")
        with col4:
            st.metric("Avg Daily Revenue", f"£{avg_daily_revenue:,.0f}")
        
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Trend Penjualan")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts_data.index,
                y=ts_data['TotalRevenue'],
                mode='lines',
                name='Daily Revenue',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ))
            fig.update_layout(
                height=400,
                hovermode='x unified',
                xaxis_title='Tanggal',
                yaxis_title='Revenue (£)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🏆 Top 5 Produk")
            top5 = product_stats.head(5)[['ProductName', 'TotalRevenue']]
            top5['Revenue'] = top5['TotalRevenue'].apply(lambda x: f"£{x:,.0f}")
            st.dataframe(
                top5[['ProductName', 'Revenue']].reset_index(drop=True),
                hide_index=True,
                use_container_width=True
            )
        
        st.markdown("---")
        st.subheader("📊 ABC Classification")
        abc_summary = product_stats.groupby('Category').agg({
            'ProductNo': 'count',
            'TotalRevenue': 'sum'
        }).reset_index()
        abc_summary.columns = ['Category', 'Jumlah Produk', 'Total Revenue']
        abc_summary['Revenue %'] = (abc_summary['Total Revenue'] / 
                                    abc_summary['Total Revenue'].sum() * 100).round(1)
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                abc_summary,
                values='Jumlah Produk',
                names='Category',
                title='Distribusi Produk per Kategori',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                abc_summary,
                x='Category',
                y='Revenue %',
                title='Kontribusi Revenue per Kategori',
                color='Category',
                color_discrete_sequence=px.colors.qualitative.Set2,
                text='Revenue %'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🔮 Forecasting Revenue")
        
        if models_loaded:
            st.success("✅ **Trained models loaded from notebook** (mining.ipynb)")
            with st.expander("📊 Model Information"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best ARIMA Order", str(model_metadata.get('best_order')))
                with col2:
                    st.metric("Best Model", model_metadata.get('best_model_name'))
                with col3:
                    st.metric("Training Date", str(model_metadata.get('forecast_date'))[:10])
        else:
            st.warning("⚠️ **No trained models found!** Please run mining.ipynb to train and save models first.")
        
        if st.button("🚀 Generate Forecast", type="primary"):
            with st.spinner("Memproses forecasting..."):
                if not models_loaded:
                    st.error("❌ Trained models not found! Please run the notebook (mining.ipynb) first to train and save models.")
                else:
                    train_size = int(len(ts_data) * 0.8)
                    train_data = ts_data['TotalRevenue'][:train_size]
                    test_data = ts_data['TotalRevenue'][train_size:]
                    
                    best_arima_order = model_metadata.get('best_order') if model_metadata else (2, 1, 2)
                    ma_avg = ma_stats['ma_value'] if ma_stats else train_data.mean()

                    arima_fit = ARIMA(train_data, order=best_arima_order).fit()
                    arima_test_pred = arima_fit.forecast(steps=len(test_data))

                    try:
                        es_fit = ExponentialSmoothing(
                            train_data, seasonal_periods=7, trend='add', seasonal='add',
                            initialization_method='estimated'
                        ).fit(optimized=True)
                    except:
                        es_fit = ExponentialSmoothing(train_data, trend='add', seasonal=None).fit(optimized=True)
                    es_test_pred = es_fit.forecast(steps=len(test_data))

                    ma_test_pred = pd.Series([ma_avg] * len(test_data), index=test_data.index)

                    mae_ma = mean_absolute_error(test_data, ma_test_pred)
                    mae_arima = mean_absolute_error(test_data, arima_test_pred)
                    mae_es = mean_absolute_error(test_data, es_test_pred)

                    if model_choice == 'Moving Average':
                        best_model_name = 'Moving Average'
                    elif model_choice == 'ARIMA':
                        best_model_name = 'ARIMA'
                    elif model_choice == 'Exponential Smoothing':
                        best_model_name = 'Exponential Smoothing'
                    else:
                        errors = {'Moving Average': mae_ma, 'ARIMA': mae_arima, 'Exponential Smoothing': mae_es}
                        best_model_name = min(errors, key=errors.get)

                    full_data = ts_data['TotalRevenue']
                    if best_model_name == 'Moving Average':
                        ma_avg_full = full_data.mean()
                        future_model = None
                    elif best_model_name == 'ARIMA':
                        future_model = ARIMA(full_data, order=best_arima_order).fit()
                        st.session_state.final_arima_order = best_arima_order
                    else:
                        try:
                            future_model = ExponentialSmoothing(
                                full_data, seasonal_periods=7, trend='add', seasonal='add',
                                initialization_method='estimated'
                            ).fit(optimized=True)
                        except:
                            future_model = ExponentialSmoothing(full_data, trend='add', seasonal=None).fit(optimized=True)

                    future_dates = pd.date_range(
                        start=ts_data.index[-1] + timedelta(days=1),
                        periods=forecast_days, freq='D'
                    )

                    if best_model_name == 'Moving Average':
                        future_forecast = pd.Series([ma_avg_full] * forecast_days, index=future_dates)
                    elif best_model_name == 'ARIMA':
                        future_forecast = future_model.forecast(steps=forecast_days)
                        future_forecast.index = future_dates
                    else:
                        future_forecast = future_model.forecast(steps=forecast_days)
                        future_forecast.index = future_dates

                    st.session_state.future_forecast = future_forecast
                    st.session_state.best_model_name = best_model_name
                    st.session_state.mae_ma = mae_ma
                    st.session_state.mae_arima = mae_arima
                    st.session_state.mae_es = mae_es
                    st.session_state.forecasts_generated = True

                    st.success(f"✅ Forecast berhasil! Model terbaik: {best_model_name}")
        
        if st.session_state.forecasts_generated:
            future_forecast = st.session_state.future_forecast
            best_model_name = st.session_state.best_model_name
            st.subheader("🎯 Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model Terpilih", best_model_name)
            with col2:
                st.metric("Moving Avg MAE", f"£{st.session_state.mae_ma:,.0f}")
            with col3:
                st.metric("ARIMA MAE", f"£{st.session_state.mae_arima:,.0f}")
            with col4:
                st.metric("Exp. Smoothing MAE", f"£{st.session_state.mae_es:,.0f}")
            
            if best_model_name == 'ARIMA' and hasattr(st.session_state, 'final_arima_order'):
                st.info(f"📊 **ARIMA Order Optimal:** {st.session_state.final_arima_order} (selected by lowest AIC)")
            
            st.subheader("📈 Prediksi Revenue")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts_data.index[-90:],
                y=ts_data['TotalRevenue'][-90:],
                mode='lines',
                name='Data Historis',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=future_forecast.index,
                y=future_forecast,
                mode='lines+markers',
                name='Prediksi',
                line=dict(color='#ff7f0e', width=3, dash='dash'),
                marker=dict(size=6)
            ))
            
            std_error = ts_data['TotalRevenue'].std() * 0.1
            upper_bound = future_forecast + 1.96 * std_error
            lower_bound = future_forecast - 1.96 * std_error
            
            fig.add_trace(go.Scatter(
                x=future_forecast.index, y=upper_bound,
                mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=future_forecast.index, y=lower_bound,
                mode='lines', line=dict(width=0),
                fillcolor='rgba(255, 127, 14, 0.2)',
                fill='tonexty',
                name='95% Confidence Interval',
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                height=500,
                hovermode='x unified',
                xaxis_title='Tanggal',
                yaxis_title='Revenue (£)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("💰 Ringkasan Forecast")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Forecast Revenue", f"£{future_forecast.sum():,.0f}")
            with col2:
                st.metric("Rata-rata Harian", f"£{future_forecast.mean():,.0f}")
            with col3:
                st.metric("Rentang", f"£{future_forecast.min():,.0f} - £{future_forecast.max():,.0f}")
    
    with tab3:
        st.header("📦 Analisis Produk & Forecast")
        
        category_filter = st.multiselect(
            "Filter Kategori",
            options=product_stats['Category'].unique(),
            default=product_stats['Category'].unique()
        )
        
        filtered_products = product_stats[product_stats['Category'].isin(category_filter)]
        st.subheader("📊 Daftar Produk")
        
        display_df = filtered_products[['ProductName', 'Category', 'TotalQuantity', 
                                       'TotalRevenue', 'AvgPrice']].copy()
        display_df['TotalRevenue'] = display_df['TotalRevenue'].apply(lambda x: f"£{x:,.0f}")
        display_df['AvgPrice'] = display_df['AvgPrice'].apply(lambda x: f"£{x:.2f}")
        display_df['TotalQuantity'] = display_df['TotalQuantity'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(
            display_df.reset_index(drop=True),
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        st.subheader("🔮 Forecast Per Produk")
        top_n = st.slider("Jumlah Produk Teratas", 5, 20, 10)
        
        if st.button("Generate Product Forecasts"):
            with st.spinner("Generating forecasts untuk produk..."):
                product_forecasts = {}
                for idx, row in product_stats.head(top_n).iterrows():
                    product_no = row['ProductNo']
                    product_name = row['ProductName']
                    
                    product_df = df[df['ProductNo'] == product_no].copy()
                    product_daily = product_df.groupby('Date')['Quantity'].sum()
                    product_daily = product_daily.reindex(
                        pd.date_range(product_daily.index.min(), 
                                     product_daily.index.max(), freq='D'),
                        fill_value=0
                    )
                    
                    window = min(7, len(product_daily) // 2)
                    if len(product_daily) > window:
                        forecast_qty = product_daily.rolling(window).mean().iloc[-1]
                        forecast_qty = product_daily.mean() if pd.isna(forecast_qty) else forecast_qty
                    else:
                        forecast_qty = product_daily.mean()
                    
                    product_forecasts[product_no] = {
                        'name': product_name,
                        'daily_forecast': forecast_qty,
                        '30day_forecast': forecast_qty * 30
                    }
                
                forecast_df = pd.DataFrame([
                    {
                        'Product': v['name'][:40],
                        'Daily Demand': f"{v['daily_forecast']:.1f}",
                        '30-Day Demand': f"{v['30day_forecast']:.0f}"
                    }
                    for k, v in product_forecasts.items()
                ])
                st.dataframe(forecast_df, use_container_width=True)
                
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name="product_forecasts.csv",
                    mime="text/csv"
                )
    
    with tab4:
        st.header("💡 Rekomendasi Bisnis")

        df['DayOfWeek'] = df['Date'].dt.day_name()
        dow_sales = df.groupby('DayOfWeek')['Revenue'].sum().sort_values(ascending=False)
        best_day = dow_sales.index[0]
        worst_day = dow_sales.index[-1]
        weekend_comparison = df.groupby(df['Date'].dt.dayofweek.isin([5, 6]))['Revenue'].sum()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Day", best_day)
        with col2:
            st.metric("Pendapatan Tertinggi", f"£{dow_sales.max():,.0f}")
        with col3:
            st.metric("Pendapatan Terendah", f"£{dow_sales.min():,.0f} ({worst_day})")

        st.markdown("---")
        st.subheader("📌 Rekomendasi Utama")
        a_revenue = product_stats[product_stats['Category'] == 'A - Best Sellers (Top 80%)']['TotalRevenue'].sum()
        c_revenue = product_stats[product_stats['Category'] == 'C - Slow Movers']['TotalRevenue'].sum()
        a_pct = (a_revenue / product_stats['TotalRevenue'].sum()) * 100
        
        best_day_revenue = dow_sales.max()
        worst_day_revenue = dow_sales.min()
        revenue_diff = ((best_day_revenue - worst_day_revenue) / worst_day_revenue) * 100
        
        top_product = product_stats.iloc[0]
        top_product_rev = top_product['TotalRevenue']
        top_product_pct = (top_product_rev / product_stats['TotalRevenue'].sum()) * 100
        
        avg_daily = ts_data['TotalRevenue'].mean()
        std_daily = ts_data['TotalRevenue'].std()
        
        recommendations = [
            f"🎯 Fokus promosi di hari {best_day}: revenue £{best_day_revenue:,.0f} ({revenue_diff:.0f}% lebih tinggi dari {worst_day})",
            f"⭐ Prioritaskan kategori A (£{a_revenue:,.0f}, {a_pct:.1f}% total revenue) untuk kampanye utama",
            f"📦 Monitor {len(product_stats[product_stats['Category'] == 'C - Slow Movers'])} produk kategori C (£{c_revenue:,.0f}) - pertimbangkan diskon/bundling",
            f"🏆 Produk '{top_product['ProductName'][:30]}' adalah top seller (£{top_product_rev:,.0f}, {top_product_pct:.1f}% revenue) - pastikan stok aman",
            f"📈 Target daily revenue: £{avg_daily:,.0f} (±£{std_daily:,.0f}) - monitor deviasi signifikan"
        ]
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")

        st.markdown("---")
        st.subheader("📅 Weekday vs Weekend")
        if len(weekend_comparison) == 2:
            weekday_pct = (weekend_comparison.iloc[0] / weekend_comparison.sum()) * 100
            weekend_pct = (weekend_comparison.iloc[1] / weekend_comparison.sum()) * 100
            st.write(f"Weekday: {weekday_pct:.1f}% | Weekend: {weekend_pct:.1f}% of revenue")
        st.write("Gunakan pola ini untuk menjadwalkan kampanye dan tenaga kerja.")
    
    with tab5:
        st.header("📊 Analisis Detail")
        
        st.subheader("📅 Analisis Musiman")
        df['DayOfWeek'] = df['Date'].dt.day_name()
        df['Month'] = df['Date'].dt.month_name()
        df['IsWeekend'] = df['Date'].dt.dayofweek.isin([5, 6])
        
        col1, col2 = st.columns(2)
        with col1:
            dow_data = df.groupby('DayOfWeek')['Revenue'].sum().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                'Friday', 'Saturday', 'Sunday'
            ])
            
            fig = go.Figure(data=[
                go.Bar(x=dow_data.index, y=dow_data.values,
                      marker_color='lightblue',
                      text=[f'£{v:,.0f}' for v in dow_data.values],
                      textposition='outside')
            ])
            fig.update_layout(
                title='Revenue per Hari',
                xaxis_title='Hari',
                yaxis_title='Revenue (£)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            weekend_data = df.groupby('IsWeekend')['Revenue'].sum()
            weekend_labels = ['Weekday', 'Weekend']
            
            fig = go.Figure(data=[
                go.Pie(labels=weekend_labels, values=weekend_data.values,
                      hole=0.4)
            ])
            fig.update_layout(
                title='Weekday vs Weekend Revenue',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Distribusi Permintaan")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Tanggal Mulai",
                value=ts_data.index.min(),
                min_value=ts_data.index.min(),
                max_value=ts_data.index.max()
            )
        with col2:
            end_date = st.date_input(
                "Tanggal Akhir",
                value=ts_data.index.max(),
                min_value=ts_data.index.min(),
                max_value=ts_data.index.max()
            )
        
        filtered_ts_data = ts_data[(ts_data.index >= pd.Timestamp(start_date)) & 
                                   (ts_data.index <= pd.Timestamp(end_date))]
        
        if len(filtered_ts_data) > 0:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=filtered_ts_data['TotalRevenue'],
                nbinsx=30,
                name='Daily Revenue Distribution',
                marker_color='lightgreen',
                hovertemplate='<b>Revenue Range</b><br>%{x}<br><b>Frequency (Hari)</b>: %{y}<extra></extra>'
            ))
            
            mean_rev = filtered_ts_data['TotalRevenue'].mean()
            median_rev = filtered_ts_data['TotalRevenue'].median()
            fig.add_vline(x=mean_rev, line_dash="dash", line_color="red",
                         annotation_text=f"Mean: £{mean_rev:,.0f}")
            fig.add_vline(x=median_rev, line_dash="dash", line_color="blue",
                         annotation_text=f"Median: £{median_rev:,.0f}")
            
            fig.update_layout(
                title=f'Distribusi Revenue Harian ({start_date} - {end_date})',
                xaxis_title='Revenue (£)',
                yaxis_title='Frequency (Jumlah Hari)',
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Ringkasan Statistik:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jumlah Hari", f"{len(filtered_ts_data)}")
            with col2:
                st.metric("Mean", f"£{mean_rev:,.0f}")
            with col3:
                st.metric("Median", f"£{median_rev:,.0f}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Std Dev", f"£{filtered_ts_data['TotalRevenue'].std():,.0f}")
            with col2:
                st.metric("Min", f"£{filtered_ts_data['TotalRevenue'].min():,.0f}")
            with col3:
                st.metric("Max", f"£{filtered_ts_data['TotalRevenue'].max():,.0f}")
        else:
            st.error("❌ Tidak ada data dalam range tanggal yang dipilih")
        
        st.markdown("---")
        st.subheader("🔗 Korelasi Metrics")
        corr_data = ts_data[['TotalRevenue', 'TotalQuantity', 'NumTransactions']].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_data.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title='Correlation Matrix',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📈 Decision Support System - Sales Forecasting v1.0</p>
    <p>Developed for UAS DSS - Semester 5</p>
</div>
""", unsafe_allow_html=True)
