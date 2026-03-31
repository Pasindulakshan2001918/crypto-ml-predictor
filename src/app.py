import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Crypto Predictor Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Crypto Price Predictor (Pro Edition)")
st.caption("⚠️ Model trained on historical data (2013–2018). Predictions may not reflect current market conditions.")

# ======================
# LOAD MODEL + FEATURES
# ======================
@st.cache_resource
def load_model():
    rf_model = joblib.load("models/rf_model.pkl")
    features = joblib.load("models/features.pkl")
    return rf_model, features

rf, FEATURES = load_model()

# ======================
# LOAD DEFAULT DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/all_crypto_currencies.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df_default = load_data()

# ======================
# SIDEBAR
# ======================
st.sidebar.header("⚙️ Control Panel")

coin = st.sidebar.selectbox(
    "Select Coin",
    sorted(df_default['slug'].unique())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Custom CSV",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.info("Model: Random Forest Regressor")

# ======================
# LOAD USER / DEFAULT DATA
# ======================
if uploaded_file:
    df_user = pd.read_csv(uploaded_file)
    df_user['date'] = pd.to_datetime(df_user['date'])

    required_cols = ['date','open','high','low','close','volume','market','ranknow','slug']
    missing = set(required_cols) - set(df_user.columns)

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    coin_df = df_user[df_user['slug'] == coin].sort_values('date')

    if coin_df.empty:
        st.warning("Uploaded file doesn't contain selected coin. Using default data.")
        coin_df = df_default[df_default['slug'] == coin].sort_values('date')
    else:
        st.success("Using uploaded dataset")

else:
    coin_df = df_default[df_default['slug'] == coin].sort_values('date')

# ======================
# DATA LIMIT CHECK
# ======================
MIN_DAYS = 120
MAX_DAYS = 730

if len(coin_df) < MIN_DAYS:
    st.error(f"Need at least {MIN_DAYS} days of data")
    st.stop()

if len(coin_df) > MAX_DAYS:
    coin_df = coin_df.tail(MAX_DAYS)
    st.warning(f"Using last {MAX_DAYS} days")

# ======================
# FEATURE ENGINEERING
# ======================
def compute_features(df):
    df = df.copy()
    g = df.groupby('slug', group_keys=False)

    df['daily_return'] = g['close'].pct_change()
    df['ma_30'] = g['close'].rolling(30, 1).mean().reset_index(level=0, drop=True)

    df['vol_7'] = g['daily_return'].rolling(7,1).std().reset_index(level=0, drop=True)
    df['vol_30'] = g['daily_return'].rolling(30,1).std().reset_index(level=0, drop=True)

    df['rolling_max_7'] = g['high'].rolling(7,1).max().reset_index(level=0, drop=True)
    df['rolling_min_7'] = g['low'].rolling(7,1).min().reset_index(level=0, drop=True)

    df['lag_1'] = g['close'].shift(1)
    df['lag_7'] = g['close'].shift(7)

    df['momentum_7'] = df['close'] - df['lag_7']
    df['momentum_14'] = df['close'] - g['close'].shift(14)

    vol_ma = g['volume'].rolling(7,1).mean().reset_index(level=0, drop=True)
    df['vol_ratio'] = df['volume'] / (vol_ma + 1e-9)

    df['high_low_spread'] = df['high'] - df['low']
    df['close_open_spread'] = df['close'] - df['open']
    df['high_close_ratio'] = df['high'] / (df['close'] + 1e-9)

    df['market_cap_ratio'] = df['market'] / df.groupby('date')['market'].transform('sum')
    df['rank_normalized'] = df['ranknow'] / df['ranknow'].max()

    df['ema_14'] = g['close'].transform(lambda x: x.ewm(span=14, adjust=False).mean())

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14,1).mean() / (loss.rolling(14,1).mean() + 1e-9)
    df['rsi_14'] = 100 - (100/(1+rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    df['vol_cluster_14'] = df['daily_return'].rolling(14,1).std()

    df.fillna(0, inplace=True)
    return df

# ======================
# DASHBOARD HEADER
# ======================
st.subheader(f"📊 {coin.upper()} Dashboard")

c1, c2, c3 = st.columns(3)

c1.metric("Data Points", len(coin_df))
c2.metric("Last Price", f"${coin_df['close'].iloc[-1]:,.2f}")
c3.metric("7D Volatility", f"{coin_df['close'].pct_change().std():.4f}")

# ======================
# PREDICTION
# ======================
if st.button("🔮 Predict Next Day Price"):

    df_feat = compute_features(coin_df)
    last = df_feat.iloc[-1]

    X = last[FEATURES].values.reshape(1, -1)

    # ---- Prediction ----
    pred = rf.predict(X)[0]

    # ---- Confidence (tree variance) ----
    tree_preds = np.array([t.predict(X)[0] for t in rf.estimators_])
    std = tree_preds.std()

    current = last['close']
    change_pct = (pred - current) / current * 100

    next_date = last['date'] + pd.Timedelta(days=1)

    st.markdown("## 🔮 Prediction Result")

    p1, p2, p3 = st.columns(3)

    p1.metric("Predicted Price", f"${pred:,.2f}")
    p2.metric("Change %", f"{change_pct:.2f}%", delta=f"{change_pct:.2f}%")
    p3.metric("Confidence (±)", f"{std:.2f}")

    # ======================
    # CHART
    # ======================
    plot_df = coin_df[['date','close']].copy()
    plot_df = pd.concat([
        plot_df,
        pd.DataFrame([{'date': next_date, 'close': pred}])
    ])

    st.subheader("📈 Price Trend + Prediction")
    st.line_chart(plot_df.set_index('date')['close'], height=400)

    # ======================
    # FEATURE IMPORTANCE
    # ======================
    st.subheader("🧠 Feature Importance")

    fi = pd.DataFrame({
        "feature": FEATURES,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False).head(10)

    st.bar_chart(fi.set_index("feature"))

# ======================
# DATA VIEW
# ======================
with st.expander("🔍 View Raw Data"):
    st.dataframe(coin_df.tail(50))