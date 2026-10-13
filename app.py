import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm
from arch import arch_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

PICKLE_PATH = "garch_lstm_lq45.pkl"

st.set_page_config(page_title="Prediksi Volatilitas LQ45 (GARCH-LSTM)", layout="wide")


# ---------------------------------------------------------------- loader & model
@st.cache_resource
def load_art():
    with open(PICKLE_PATH, "rb") as fh:
        return pickle.load(fh)


def build_lstm(params, input_shape):
    # harus sama persis dengan build_lstm di notebook
    model = Sequential()
    n_layers = params["n_layers"]
    for i in range(n_layers):
        return_seq = i < n_layers - 1
        if i == 0:
            model.add(LSTM(params["n_units"], return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(LSTM(params["n_units"], return_sequences=return_seq))
        model.add(Dropout(params["dropout"]))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=params["learning_rate"]), loss="mse")
    return model


@st.cache_resource
def get_model(name):
    art = load_art()
    entry = art["models"][name]
    m = build_lstm(entry["params"], (art["config"]["timestep"], len(entry["features"])))
    m.set_weights(entry["weights"])
    return m


# ---------------------------------------------------------------- inferensi
def to_orig(art, pred_scaled):
    c = art["config"]
    dummy = np.zeros((len(pred_scaled), len(c["features"]) + 1))
    dummy[:, c["target_col_idx"]] = pred_scaled
    return art["scaler"].inverse_transform(dummy)[:, c["target_col_idx"]]


def forecast_next(art, log_returns, model_name):
    c, g = art["config"], art["garch"]
    entry = art["models"][model_name]
    r = np.asarray(log_returns, dtype=float).ravel()
    gm = arch_model(pd.Series(r * g["returns_scale"]), vol="GARCH", p=1, q=1, dist="normal").fix(g["params"])
    cond = np.asarray(gm.conditional_volatility) / g["returns_scale"]
    feats = np.column_stack([r, cond, np.zeros(len(r))])  # kolom ke-3 = placeholder target
    scaled = art["scaler"].transform(feats)[:, : len(c["features"])]
    X_new = scaled[-c["timestep"]:][None, :, :][:, :, entry["features"]]
    pred_scaled = get_model(model_name).predict(X_new, verbose=0).flatten()
    vol = float(to_orig(art, pred_scaled)[0])
    var = float(norm.ppf(1 - c["confidence"]) * vol)
    return vol, var, cond


def fetch_new_returns(art):
    import yfinance as yf

    base_close = art["data"]["Close"]
    last_date = base_close.index[-1]
    start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(art["meta"]["ticker"], start=start, progress=False)
    if raw is None or len(raw) == 0:
        return pd.Series(dtype=float)
    new_close = pd.Series(np.asarray(raw["Close"]).ravel(), index=pd.DatetimeIndex(raw.index)).dropna()
    new_close = new_close[new_close.index > last_date]
    if new_close.empty:
        return pd.Series(dtype=float)
    chain = pd.concat([base_close.iloc[-1:], new_close])
    return np.log(chain / chain.shift(1)).dropna()


def convergence(trials):
    tr = trials.sort_values("number")
    best = tr["value"].cummin().values
    cum_min = tr["duration"].dt.total_seconds().cumsum().values / 60
    return best, cum_min


# ---------------------------------------------------------------- UI
art = load_art()
c, meta = art["config"], art["meta"]
names = list(art["models"].keys())
default_i = names.index(art["default_model"])
conf_pct = int(round(c["confidence"] * 100))

st.title("Prediksi Volatilitas Indeks LQ45 (GARCH-LSTM + Optuna)")
st.caption(
    f"Data {meta['ticker']} {meta['start']} s.d. {meta['end']} | data latih sampai {c['train_end_date']} | "
    f"model diekspor {meta['created_at'][:10]}"
)

tab_pred, tab_test, tab_tbl, tab_search = st.tabs(
    ["Prediksi", "Evaluasi data test", "Tabel hasil", "Pencarian hyperparameter"]
)

# ---- Tab 1: prediksi hari berikutnya
with tab_pred:
    model_name = st.selectbox("Model", names, index=default_i)
    use_live = st.checkbox("Sertakan data terbaru dari yfinance", value=False)

    returns = art["data"]["log_return"].copy()
    if use_live:
        try:
            new_r = fetch_new_returns(art)
            if len(new_r) > 0:
                returns = pd.concat([returns, new_r])
                st.info(f"Menambahkan {len(new_r)} hari data baru dari yfinance.")
            else:
                st.info("Tidak ada data baru setelah snapshot.")
        except Exception as e:
            st.warning(f"Gagal mengambil data yfinance, memakai snapshot saja. ({e})")

    vol, var, cond = forecast_next(art, returns.values, model_name)

    m1, m2, m3 = st.columns(3)
    m1.metric("Tanggal data terakhir", str(returns.index[-1].date()))
    m2.metric("Prediksi volatilitas harian (hari berikutnya)", f"{vol:.2%}")
    m3.metric(f"VaR {conf_pct}% (1 hari)", f"{var:.2%}")

    w = c["realized_vol_window"]
    hist = pd.DataFrame({"Volatilitas kondisional GARCH": cond}, index=returns.index)
    hist[f"Realized volatility ({w} hari)"] = returns.rolling(w).std()
    st.line_chart(hist.tail(250))
    st.caption(
        "Target model adalah realized volatility (simpangan baku log return rolling). "
        "VaR parametrik: asumsi distribusi normal dan mean return nol."
    )

# ---- Tab 2: evaluasi data test
with tab_test:
    t = art["test"]
    sel = st.multiselect("Model", names, default=[art["default_model"]])
    plot_df = pd.DataFrame({"Realized volatility aktual": t["y_true"]}, index=t["dates"])
    for n in sel:
        plot_df[n] = t["pred_mean"][n]
    st.line_chart(plot_df)
    st.caption(f"Prediksi = rata-rata {meta['n_runs']} run (seperti di notebook), skala asli.")

    st.subheader("VaR vs return aktual")
    vm = st.selectbox("Model untuk grafik VaR", names, index=default_i, key="var_model")
    var_s = np.asarray(t["var"][vm])
    ret = np.asarray(t["actual_returns"])
    exc = ret < var_s
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t["dates"], ret, lw=0.8, label="Log return aktual")
    ax.plot(t["dates"], var_s, lw=1.2, label=f"VaR {conf_pct}%")
    ax.scatter(t["dates"][exc], ret[exc], color="red", s=14, label=f"Exceedance ({int(exc.sum())})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    st.dataframe(art["tables"]["backtest"])
    st.caption("p-value > 0.05 = model tidak ditolak (VaR dianggap valid).")

# ---- Tab 3: tabel hasil
with tab_tbl:
    st.subheader("Tabel A. Ringkasan tiap model")
    st.dataframe(art["tables"]["results"])
    st.subheader("Tabel B. Perbandingan berpasangan (error dan efisiensi)")
    st.dataframe(art["tables"]["significance"])
    st.subheader("Efisiensi pencarian")
    st.dataframe(art["tables"]["efficiency"])
    st.subheader("Setelan terbaik tiap model")
    st.dataframe(pd.DataFrame({n: e["params"] for n, e in art["models"].items()}).T.astype(str))

# ---- Tab 4: konvergensi pencarian
with tab_search:
    g_best, g_min = convergence(art["tables"]["trials_grid"])
    o_best, o_min = convergence(art["tables"]["trials_optuna"])
    fig2, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(np.arange(1, len(g_best) + 1), g_best, marker="o", markersize=3, label="Grid Search")
    axes[0].plot(np.arange(1, len(o_best) + 1), o_best, marker="o", markersize=3, label="Optuna (TPE)")
    axes[0].set_xlabel("Nomor percobaan")
    axes[0].set_ylabel("Skor validasi terbaik (MSE)")
    axes[0].set_title("Konvergensi vs percobaan")
    axes[0].legend(fontsize=8)
    axes[1].plot(g_min, g_best, marker="o", markersize=3, label="Grid Search")
    axes[1].plot(o_min, o_best, marker="o", markersize=3, label="Optuna (TPE)")
    axes[1].set_xlabel("Waktu komputasi kumulatif (menit)")
    axes[1].set_ylabel("Skor validasi terbaik (MSE)")
    axes[1].set_title("Konvergensi vs waktu")
    axes[1].legend(fontsize=8)
    fig2.tight_layout()
    st.pyplot(fig2)
    st.subheader("Ruang pencarian")
    st.json(c["search_space"])

st.divider()
st.caption("Proyek riset akademik. Bukan saran investasi.")
