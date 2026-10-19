import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from arch import arch_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

PICKLE_PATH = "garch_lstm_lq45.pkl"
MAX_HORIZON = 30  # hari bursa

st.set_page_config(page_title="Prediksi Volatilitas LQ45 (GARCH-LSTM)", layout="wide")


# ---------------------------------------------------------------- loader & model
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


# ---------------------------------------------------------------- inferensi
def to_orig(art, pred_scaled):
    c = art["config"]
    dummy = np.zeros((len(pred_scaled), len(c["features"]) + 1))
    dummy[:, c["target_col_idx"]] = pred_scaled
    return art["scaler"].inverse_transform(dummy)[:, c["target_col_idx"]]


def forecast_horizon(art, log_returns, model_name, z, horizon):
    """Ramalan rekursif h hari bursa ke depan.
    Return masa depan diasumsikan 0; volatilitas kondisional masa depan dari peramalan GARCH."""
    c, g = art["config"], art["garch"]
    e = art["models"][model_name]
    ts = c["timestep"]
    r = np.asarray(log_returns, dtype=float).ravel()
    if len(r) < ts:
        raise ValueError(f"Butuh minimal {ts} hari data.")

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
        rows.append({"Volatilitas GARCH-LSTM": vols.mean(), "Simpangan antar-run": vols.std(),
                     "VaR": z * vols.mean(), "Volatilitas GARCH saja": cond_fut[h - 1]})
    return pd.DataFrame(rows), cond_hist


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


# ---------------------------------------------------------------- UI
art = load_art()
c, meta = art["config"], art["meta"]
names = list(art["models"].keys())
default_i = names.index(art["default_model"])
conf_pct = int(round(c["confidence"] * 100))
w = c["realized_vol_window"]

st.title("Prediksi Volatilitas Indeks LQ45 (GARCH-LSTM + VU + Optuna)")
st.caption(
    f"Data {meta['ticker']} {meta['start']} s.d. {meta['end']} | data latih sampai {c['train_end_date']} | "
    f"target: realized volatility {w} hari ({meta.get('target_mode', 'backward')}) | "
    f"model diekspor {meta['created_at'][:10]}"
)

tab_pred, tab_test, tab_tbl, tab_search = st.tabs(
    ["Prediksi", "Evaluasi data test", "Tabel hasil", "Pencarian hyperparameter"]
)

# ---- Tab 1: prediksi sampai tanggal tertentu
with tab_pred:
    col1, col2 = st.columns([2, 1])
    model_name = col1.selectbox("Model", names, index=default_i)
    dist = col2.radio("Distribusi VaR", ["normal", "empiris"], horizontal=True)
    z = c["var_z"][dist]
    use_live = st.checkbox("Sertakan data terbaru dari yfinance", value=True)

    returns = art["data"]["log_return"].copy()
    if use_live:
        try:
            new_r = fetch_new_returns(art)
            if len(new_r) > 0:
                returns = pd.concat([returns, new_r])
                st.info(f"Menambahkan {len(new_r)} hari data baru (sampai {returns.index[-1].date()}). "
                        f"Parameter GARCH dan scaler tetap dari data latih; data baru hanya difilter.")
            else:
                st.info("Tidak ada data baru setelah snapshot.")
        except Exception as ex:
            st.warning(f"Gagal mengambil data yfinance, memakai snapshot saja. ({ex})")

    last_date = returns.index[-1]
    min_d = (last_date + pd.offsets.BDay(1)).date()
    max_d = (last_date + pd.offsets.BDay(MAX_HORIZON)).date()
    target_date = st.date_input("Prediksi volatilitas untuk tanggal", value=min_d,
                                min_value=min_d, max_value=max_d)

    fut_dates = pd.bdate_range(last_date, target_date)[1:]
    if len(fut_dates) == 0:
        st.error("Pilih tanggal setelah data terakhir.")
        st.stop()
    horizon = len(fut_dates)

    fc_df, cond = forecast_horizon(art, returns.values, model_name, z, horizon)
    fc_df.index = fut_dates
    last = fc_df.iloc[-1]

    m1, m2, m3 = st.columns(3)
    m1.metric("Data terakhir", str(last_date.date()), help=f"{horizon} hari bursa ke tanggal target")
    m2.metric(f"Prediksi volatilitas harian {target_date}", f"{last['Volatilitas GARCH-LSTM']:.2%}",
              help=f"Rata-rata {meta['n_runs']} run; simpangan antar-run {last['Simpangan antar-run']:.2%}")
    m3.metric(f"VaR {conf_pct}% (1 hari, z={z:.2f})", f"{last['VaR']:.2%}")

    st.subheader("Lintasan ramalan sampai tanggal target")
    st.line_chart(fc_df[["Volatilitas GARCH-LSTM", "Volatilitas GARCH saja"]])
    st.dataframe(fc_df.style.format("{:.2%}"))

    st.subheader(f"Histori {w}-hari terakhir")
    hist = pd.DataFrame({"Volatilitas kondisional GARCH": cond}, index=returns.index)
    hist[f"Realized volatility ({w} hari)"] = returns.rolling(w).std()
    st.line_chart(hist.tail(250))

    st.caption(
        "Hari bursa dihitung Senin–Jumat; libur BEI belum dikecualikan. "
        "Untuk horizon > 1 hari, return masa depan diasumsikan 0 dan volatilitas kondisional memakai "
        "peramalan GARCH, jadi prediksi mendekati GARCH murni dan makin tidak pasti seiring horizon. "
        "VaR parametrik dengan mean return nol; z empiris = kuantil return terstandardisasi dari data latih."
    )

# ---- Tab 2, 3, 4: sama persis dengan app.py lama (salin dari file asli)
