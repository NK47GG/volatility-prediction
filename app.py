import pickle

import numpy as np
import pandas as pd
import streamlit as st
from arch import arch_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

PICKLE_PATH = "garch_lstm_lq45.pkl"
CSV_PATH = "harga_terbaru.csv"
MAX_HORIZON = 30
EXTREME_RET = 0.08

st.set_page_config(page_title="LQ45 Volatility Forecasting (GARCH-LSTM)", layout="wide")


# ================================================================ loader & model
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
    art = load_art()
    e = art["models"][name]
    shape = (art["config"]["timestep"], len(e["features"]))
    models = []
    for w in e["weights"]:
        m = build_lstm(e["params"], shape)
        m.set_weights(w)
        models.append(m)
    return models


# ================================================================ data
@st.cache_data(ttl=3600)
def load_csv_close():
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    return pd.Series(df["Close"].values, index=df["Date"]).sort_index().dropna()


@st.cache_data(ttl=3600)
def build_close_series(_art):
    close = _art["data"]["Close"].copy()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index))
    try:
        s = load_csv_close()
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
        close = pd.concat([close, s[s.index > close.index[-1]]])
    except Exception as ex:
        st.warning(f"Could not read {CSV_PATH}, using snapshot only. ({ex})")
    return close[~close.index.duplicated(keep="last")].sort_index()


# ================================================================ inference
def to_orig(art, pred_scaled):
    c = art["config"]
    dummy = np.zeros((len(pred_scaled), len(c["features"]) + 1))
    dummy[:, c["target_col_idx"]] = pred_scaled
    return art["scaler"].inverse_transform(dummy)[:, c["target_col_idx"]]


def forecast_horizon(art, log_returns, model_name, z, horizon):
    c, g = art["config"], art["garch"]
    e = art["models"][model_name]
    ts = c["timestep"]
    r = np.asarray(log_returns, dtype=float).ravel()

    gm = arch_model(pd.Series(r * g["returns_scale"]), vol="GARCH", p=1, q=1,
                    dist="normal").fix(g["params"])
    cond_hist = np.asarray(gm.conditional_volatility) / g["returns_scale"]
    fc = gm.forecast(horizon=horizon, reindex=False)
    cond_fut = np.sqrt(fc.variance.values[-1]) / g["returns_scale"]

    r_ext = np.concatenate([r, np.zeros(horizon)])
    cond_ext = np.concatenate([cond_hist, cond_fut])
    feats = np.column_stack([r_ext, cond_ext, np.zeros(len(r_ext))])
    scaled = art["scaler"].transform(feats)[:, : len(c["features"])]

    models = get_models(model_name)
    rows = []
    for h in range(1, horizon + 1):
        end = len(r) + h - 1
        X = scaled[end - ts:end][None, :, :][:, :, e["features"]]
        p = np.array([m.predict(X, verbose=0).ravel()[0] for m in models])
        vols = to_orig(art, p)
        rows.append({"GARCH-LSTM Volatility": vols.mean(),
                     "Inter-run Std": vols.std(),
                     "VaR": z * vols.mean(),
                     "GARCH-only Volatility": cond_fut[h - 1]})
    return pd.DataFrame(rows), cond_hist


def convergence(trials):
    tr = trials.sort_values("number")
    best = tr["value"].cummin().values
    if "user_attrs_train_sec" in tr.columns:
        secs = tr["user_attrs_train_sec"].values
    else:
        secs = pd.to_timedelta(tr["duration"]).dt.total_seconds().values
    return best, np.cumsum(secs) / 60


# ================================================================ UI
art = load_art()
c, meta = art["config"], art["meta"]
names = list(art["models"].keys())
default_i = names.index(art["default_model"])
conf_pct = int(round(c["confidence"] * 100))
w = c["realized_vol_window"]

# Auto-detect VaR quantile keys ("Normal"/"Empirical" in v8 vs "normal"/"empiris" in older pkl)
var_keys = list(c["var_z"].keys())
dist_labels = ["Normal" if k.lower().startswith("norm") else "Empirical" for k in var_keys]
dist_map = dict(zip(dist_labels, var_keys))

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", names, index=default_i)
    dist_label = st.radio("VaR Distribution", dist_labels, horizontal=True)
    dist = dist_map[dist_label]
    z = c["var_z"][dist]
    st.divider()
    close = build_close_series(art)
    returns = np.log(close / close.shift(1)).dropna()
    last_date = returns.index[-1]
    st.caption(f"Data up to **{last_date.date()}**  \nLatest Close **{close.iloc[-1]:,.2f}**")
    st.caption(f"Timestep {c['timestep']} days · {meta['n_runs']}-run ensemble")

st.title("LQ45 Index Volatility Forecasting (GARCH-LSTM + VU + Optuna)")
st.caption(
    f"Trained on {meta['ticker']} {meta['start']} to {meta['end']} "
    f"(training data up to {c['train_end_date']}) | target: {w}-day realized volatility "
    f"({meta.get('target_mode', 'backward')}) | exported {meta['created_at'][:10]}"
)

tab_pred, tab_test, tab_tbl, tab_search = st.tabs(
    ["Forecast", "Test Set Evaluation", "Result Tables", "Hyperparameter Search"]
)

# ---------------------------------------------------------------- Tab 1
with tab_pred:
    stale = (pd.Timestamp.today().normalize() - last_date).days
    if stale > 7:
        st.warning(f"Latest data ({last_date.date()}) is {stale} days old; "
                   "forecasts for current dates may be less informative.")
    recent = returns.tail(10)
    big = recent[recent.abs() > EXTREME_RET]
    if len(big):
        st.warning("Extreme daily returns detected: " +
                   ", ".join(f"{d.date()} ({v:+.1%})" for d, v in big.items()) +
                   ". Please verify data integrity.")

    min_d = (last_date + pd.offsets.BDay(1)).date()
    max_d = (last_date + pd.offsets.BDay(MAX_HORIZON)).date()
    left, right = st.columns([1, 2])
    with left:
        target_date = st.date_input("Forecast Target Date", value=min_d,
                                    min_value=min_d, max_value=max_d)
    fut_dates = pd.bdate_range(last_date, target_date)[1:]
    horizon = len(fut_dates)
    with right:
        if horizon <= c["timestep"]:
            st.success(f"Horizon: {horizon} trading days — input window still contains actual market observations.")
        else:
            st.info(f"Horizon: {horizon} trading days — exceeds {c['timestep']}-day timestep; "
                    "inputs rely entirely on GARCH projections, converging toward long-run volatility.")

    fc_df, cond = forecast_horizon(art, returns.values, model_name, z, horizon)
    fc_df.index = fut_dates
    last = fc_df.iloc[-1]

    st.markdown(f"### {target_date.strftime('%A, %d %B %Y')}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Daily Volatility (GARCH-LSTM)", f"{last['GARCH-LSTM Volatility']:.2%}",
              help=f"Mean across {meta['n_runs']} runs; inter-run std: {last['Inter-run Std']:.2%}")
    m2.metric("Daily Volatility (GARCH only)", f"{last['GARCH-only Volatility']:.2%}",
              delta=f"{last['GARCH-LSTM Volatility'] - last['GARCH-only Volatility']:+.2%}",
              delta_color="off")
    m3.metric(f"1-Day {conf_pct}% VaR", f"{last['VaR']:.2%}",
              help=f"z = {z:.3f} ({dist_label}); worst expected daily loss at {conf_pct}% confidence level")

    if horizon > 1:
        st.subheader("Forecast Trajectory")
        st.line_chart(fc_df[["GARCH-LSTM Volatility", "GARCH-only Volatility"]])
        with st.expander("Daily Forecast Breakdown"):
            st.dataframe(fc_df.style.format("{:.2%}"), use_container_width=True)

    st.subheader("Past Year Historical Trajectory")
    hist = pd.DataFrame({"GARCH Conditional Volatility": cond}, index=returns.index)
    hist[f"Realized Volatility ({w}-day)"] = returns.rolling(w).std()
    st.line_chart(hist.tail(250))

    with st.expander("Methodological Notes"):
        st.markdown(
            f"- Target: {w}-day realized volatility ({meta.get('target_mode', 'backward')}), "
            f"using the last {c['timestep']} trading days as input.\n"
            "- Multi-step horizons (> 1 day) are computed recursively: future log returns are assumed 0, "
            "while conditional volatility is driven by analytical GARCH projections.\n"
            "- Pre-estimated GARCH parameters and MinMaxScaler remain fixed from the training core (no refitting on new data).\n"
            "- Trading days follow Monday–Friday business calendar (IDX holidays not excluded)."
        )

# ---------------------------------------------------------------- Tab 2
with tab_test:
    t = art["test"]
    dates = pd.to_datetime(t["dates"])
    y_true = np.asarray(t["y_true"]).ravel()
    pm = t["pred_mean"]
    pred = np.asarray(pm[model_name] if isinstance(pm, dict) else pm).ravel()

    st.subheader(f"{model_name} — Out-of-Sample Test Set")
    st.line_chart(pd.DataFrame({"Actual": y_true, "Predicted": pred}, index=dates))

    err = pred - y_true
    e1, e2, e3 = st.columns(3)
    e1.metric("RMSE", f"{np.sqrt(np.mean(err ** 2)):.5f}")
    e2.metric("MAE", f"{np.mean(np.abs(err)):.5f}")
    e3.metric("Test Observations", f"{len(y_true)}")

    if "actual_returns" in t and "var" in t:
        v = t["var"]
        if isinstance(v, dict) and model_name in v:
            v = v[model_name]
        if isinstance(v, dict):
            v = v.get(dist, v.get(dist_label, next(iter(v.values()))))
        v = np.asarray(v).ravel()
        ar = np.asarray(t["actual_returns"]).ravel()
        if len(v) == len(ar):
            st.subheader(f"{conf_pct}% VaR vs. Actual Log Returns")
            st.line_chart(pd.DataFrame({"Actual Return": ar, f"{conf_pct}% VaR": v}, index=dates))
            st.caption(f"Violations: {(ar < v).sum()} out of {len(ar)} days "
                       f"(Expected: {(1 - c['confidence']) * len(ar):.1f})")

# ---------------------------------------------------------------- Tab 3
with tab_tbl:
    tbl = art["tables"]
    for key, title in [("results", "Model Evaluation Summary"),
                       ("significance", "Diebold-Mariano & Welch Significance Tests"),
                       ("efficiency", "Hyperparameter Search Efficiency"),
                       ("search_dist", "Multi-Seed Convergence Distribution"),
                       ("backtest", "Value-at-Risk (VaR) Backtesting")]:
        if key in tbl:
            st.subheader(title)
            st.dataframe(tbl[key], use_container_width=True)

# ---------------------------------------------------------------- Tab 4
with tab_search:
    tbl = art["tables"]
    if "trials_grid" in tbl and "trials_optuna" in tbl:
        bg, tg = convergence(tbl["trials_grid"])
        bo, to = convergence(tbl["trials_optuna"])
        st.subheader("Best Validation Score Convergence vs. Training Time")
        conv = pd.concat([
            pd.DataFrame({"Grid Search": bg}, index=np.round(tg, 2)),
            pd.DataFrame({"Optuna (TPE)": bo}, index=np.round(to, 2)),
        ]).sort_index().ffill()
        conv.index.name = "Minutes"
        st.line_chart(conv)
        st.subheader("Grid Search Trials")
        st.dataframe(tbl["trials_grid"], use_container_width=True)
        st.subheader("Optuna (TPE) Trials")
        st.dataframe(tbl["trials_optuna"], use_container_width=True)
    st.subheader("Hyperparameter Search Space")
    st.json(c["search_space"])
