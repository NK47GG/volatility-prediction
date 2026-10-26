import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from arch import arch_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

PICKLE_PATH = "garch_lstm_lq45.pkl"

st.set_page_config(
    page_title="LQ45 Volatility Forecasting Dashboard (GARCH-LSTM + Optuna)",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------------------- Loader & Model Builder
@st.cache_resource
def load_art():
    with open(PICKLE_PATH, "rb") as fh:
        return pickle.load(fh)

def build_lstm(params, input_shape):
    model = Sequential([Input(shape=input_shape)])
    for i in range(params["n_layers"]):
        model.add(LSTM(params["n_units"], return_sequences=i < params["n_layers"] - 1))
        model.add(Dropout(params["dropout"]))
    model.add(Dense(1))
    return model

@st.cache_resource
def get_models(name):
    """Ensemble across all trained runs (consistent with N_RUNS ensemble evaluation)."""
    art = load_art()
    e = art["models"][name]
    shape = (art["config"]["timestep"], len(e["features"]))
    models = []
    for w in e["weights"]:
        m = build_lstm(e["params"], shape)
        m.set_weights(w)
        models.append(m)
    return models

# ---------------------------------------------------------------- Inference Engine
def to_orig(art, pred_scaled):
    c = art["config"]
    dummy = np.zeros((len(pred_scaled), len(c["features"]) + 1))
    dummy[:, c["target_col_idx"]] = pred_scaled
    return art["scaler"].inverse_transform(dummy)[:, c["target_col_idx"]]

def forecast_next(art, log_returns, model_name, z):
    c, g = art["config"], art["garch"]
    e = art["models"][model_name]
    r = np.asarray(log_returns, dtype=float).ravel()
    if len(r) < c["timestep"]:
        raise ValueError(f"Requires at least {c['timestep']} trading days of historical return data.")
    gm = arch_model(pd.Series(r * g["returns_scale"]), vol="GARCH", p=1, q=1, dist="normal").fix(g["params"])
    cond = np.asarray(gm.conditional_volatility) / g["returns_scale"]
    feats = np.column_stack([r, cond, np.zeros(len(r))])  # 3rd column placeholder for target
    scaled = art["scaler"].transform(feats)[:, : len(c["features"])]
    X_new = scaled[-c["timestep"]:][None, :, :][:, :, e["features"]]
    preds = np.array([m.predict(X_new, verbose=0).ravel()[0] for m in get_models(model_name)])
    vol_runs = to_orig(art, preds)
    vol = float(vol_runs.mean())
    return vol, float(z * vol), cond, float(vol_runs.std())

def fetch_new_returns(art):
    import yfinance as yf

    base_close = art["data"]["Close"]
    last_date = base_close.index[-1]
    start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(art["meta"]["ticker"], start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    new_close = raw["Close"].dropna()
    new_close = new_close[new_close.index > last_date]
    if new_close.empty:
        return pd.Series(dtype=float)
    chain = pd.concat([base_close.iloc[-1:], new_close])
    return np.log(chain / chain.shift(1)).dropna()

def convergence(trials):
    tr = trials.sort_values("number")
    best = tr["value"].cummin().values
    col = "user_attrs_train_sec" if "user_attrs_train_sec" in tr.columns else None
    secs = tr[col].values if col else tr["duration"].dt.total_seconds().values
    return best, np.cumsum(secs) / 60

# ---------------------------------------------------------------- UI Presentation
art = load_art()
c, meta = art["config"], art["meta"]
names = list(art["models"].keys())
default_i = names.index(art["default_model"])
conf_pct = int(round(c["confidence"] * 100))
w = c["realized_vol_window"]

st.title("LQ45 Stock Index Volatility Forecasting Dashboard")
st.caption(
    f"Index Benchmark: {meta['ticker']} ({meta['start']} to {meta['end']}) | "
    f"Training Window up to: {c['train_end_date']} | "
    f"Target: {w}-day Realized Volatility ({meta.get('target_mode', 'backward')}) | "
    f"Model Export: {meta['created_at'][:10]}"
)

tab_pred, tab_test, tab_tbl, tab_search = st.tabs(
    ["Next-Day Forecast", "Out-of-Sample Evaluation", "Performance Benchmark", "Hyperparameter Search"]
)

# ---- Tab 1: Next-Day Prediction
with tab_pred:
    col1, col2 = st.columns([2, 1])
    model_name = col1.selectbox("Model Architecture", names, index=default_i)
    dist = col2.radio("VaR Quantile Assumption", ["Normal", "Empirical"], horizontal=True)
    z = c["var_z"][dist]
    use_live = st.checkbox("Fetch latest market data via Yahoo Finance", value=False)

    returns = art["data"]["log_return"].copy()
    if use_live:
        try:
            new_r = fetch_new_returns(art)
            if len(new_r) > 0:
                returns = pd.concat([returns, new_r])
                st.info(f"Appended {len(new_r)} days of live market returns. Pre-estimated GARCH parameters "
                        f"and scalers remain fixed to training horizon ({c['train_end_date']}) to prevent look-ahead bias.")
            else:
                st.info("No new trading sessions detected beyond the current dataset snapshot.")
        except Exception as ex:
            st.warning(f"Failed to fetch live data from yfinance; reverting to offline snapshot. ({ex})")

    vol, var, cond, run_std = forecast_next(art, returns.values, model_name, z)

    m1, m2, m3 = st.columns(3)
    m1.metric("Latest Trading Date", str(returns.index[-1].date()))
    m2.metric("Forecasted Next-Day Volatility", f"{vol:.2%}",
              help=f"Ensemble mean across {meta['n_runs']} independent runs; Run std: {run_std:.2%}")
    m3.metric(f"Value-at-Risk ({conf_pct}%, z={z:.2f})", f"{var:.2%}")

    hist = pd.DataFrame({"GARCH Conditional Volatility": cond}, index=returns.index)
    hist[f"{w}-Day Realized Volatility"] = returns.rolling(w).std()
    st.line_chart(hist.tail(250))
    st.caption("Parametric Value-at-Risk assumes zero-mean daily returns. Normal z represents Gaussian quantiles; "
               "Empirical z is derived from standardized training residuals (capturing fat tails).")

# ---- Tab 2: Out-of-Sample Test Evaluation
with tab_test:
    t = art["test"]
    sel = st.multiselect("Selected Models for Comparison", names, default=[art["default_model"]])
    plot_df = pd.DataFrame({"Actual Realized Volatility": t["y_true"]}, index=t["dates"])
    for n in sel:
        plot_df[n] = t["pred_mean"][n]
    st.line_chart(plot_df)
    st.caption(f"Forecasts reflect original-scale ensemble averages across {meta['n_runs']} runs.")

    st.subheader("Value-at-Risk (VaR) vs. Actual Log Returns")
    cva, cvb = st.columns([2, 1])
    vm = cva.selectbox("Model for VaR Backtest Plot", names, index=default_i, key="var_model")
    vd = cvb.radio("Quantile Distribution", ["Normal", "Empirical"], horizontal=True, key="var_dist")
    var_s = np.asarray(t["var"][vm][vd])
    ret = np.asarray(t["actual_returns"])
    dates = pd.DatetimeIndex(t["dates"])
    exc = ret < var_s
    fig, ax = plt.subplots(figsize=(11, 4), dpi=150)
    ax.plot(dates, ret, lw=0.8, color="#64748B", label="Actual Daily Log Returns")
    ax.plot(dates, var_s, lw=1.2, color="#0284C7", label=f"95% VaR Threshold ({vd})")
    ax.scatter(dates[exc], ret[exc], color="#DC2626", s=14, label=f"Exceedances ({int(exc.sum())})")
    ax.set_ylabel("Daily Return / Risk Level", fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(art["tables"]["backtest"].reset_index(), use_container_width=True)
    st.caption("Decision Rule: Kupiec POF and Christoffersen p-values > 0.05 indicate a statistically valid risk model.")

# ---- Tab 3: Performance Tables
with tab_tbl:
    st.subheader("Table A. Out-of-Sample Model Accuracy Summary (Original Scale)")
    st.dataframe(art["tables"]["results"], use_container_width=True)
    
    st.subheader("Table B. Pairwise Statistical Significance (Diebold-Mariano & Welch's t-test)")
    st.dataframe(art["tables"]["significance"], use_container_width=True)
    
    st.subheader("Hyperparameter Search Computational Efficiency")
    st.dataframe(art["tables"]["efficiency"], use_container_width=True)
    
    if "search_dist" in art["tables"]:
        st.subheader("Multi-Seed Convergence Distribution (Optuna vs. Random Grid)")
        st.dataframe(art["tables"]["search_dist"], use_container_width=True)
        
    st.subheader("Optimal Hyperparameter Sets per Architecture")
    st.dataframe(pd.DataFrame({n: e["params"] for n, e in art["models"].items()}).T.astype(str), use_container_width=True)

# ---- Tab 4: Search Convergence Curves
with tab_search:
    g_best, g_min = convergence(art["tables"]["trials_grid"])
    o_best, o_min = convergence(art["tables"]["trials_optuna"])
    fig2, axes = plt.subplots(1, 2, figsize=(13, 4), dpi=150)
    axes[0].plot(np.arange(1, len(g_best) + 1), g_best, marker="o", markersize=3, color="tab:blue", label="Grid Search")
    axes[0].plot(np.arange(1, len(o_best) + 1), o_best, marker="s", markersize=3, color="tab:orange", label="Optuna (TPE)")
    axes[0].set_xlabel("Trial Number", fontsize=9)
    axes[0].set_ylabel("Best Validation Score (MSE)", fontsize=9)
    axes[0].set_title("Convergence vs. Trials", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    axes[1].plot(g_min, g_best, marker="o", markersize=3, color="tab:blue", label="Grid Search")
    axes[1].plot(o_min, o_best, marker="o", markersize=3, color="tab:orange", label="Optuna (TPE)")
    axes[1].set_xlabel("Cumulative Training Time (Minutes)", fontsize=9)
    axes[1].set_ylabel("Best Validation Score (MSE)", fontsize=9)
    axes[1].set_title("Convergence vs. Training Time", fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    st.subheader("Hyperparameter Search Space Definition")
    st.json(c["search_space"])

st.divider()
st.caption("Academic Research Project | Universitas Dian Nuswantoro | Benchmarked on IDX LQ45 Index.")
