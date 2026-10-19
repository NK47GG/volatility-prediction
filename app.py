import pickle

import numpy as np
import pandas as pd
import streamlit as st
from arch import arch_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

PICKLE_PATH = "garch_lstm_lq45.pkl"
CSV_PATH = "harga_terbaru.csv"   # atau URL raw GitHub bila app tidak di-deploy dari repo yang sama
MAX_HORIZON = 30                 # hari bursa
EXTREME_RET = 0.08

st.set_page_config(page_title="Prediksi Volatilitas LQ45 (GARCH-LSTM)", layout="wide")


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
def load_yf_close(ticker, start):
    import yfinance as yf
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    s = raw["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def build_close_series(art, use_csv, use_live):
    close = art["data"]["Close"].copy()
    close.index = pd.to_datetime(close.index)
    sources = []
    if use_csv:
        try:
            s = load_csv_close()
            s = s[s.index > close.index[-1]]
            if len(s):
                close = pd.concat([close, s])
                sources.append(f"harga_terbaru.csv s.d. {close.index[-1].date()}")
        except Exception as ex:
            st.warning(f"CSV tidak terbaca: {ex}")
    if use_live:
        try:
            s = load_yf_close(art["meta"]["ticker"],
                              (close.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
            s = s[s.index > close.index[-1]]
            if len(s):
                close = pd.concat([close, s])
                sources.append(f"yfinance s.d. {close.index[-1].date()}")
        except Exception as ex:
            st.warning(f"yfinance gagal: {ex}")
    return close[~close.index.duplicated(keep="last")].sort_index(), sources


# ================================================================ inferensi
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
        rows.append({"Volatilitas GARCH-LSTM": vols.mean(),
                     "Simpangan antar-run": vols.std(),
                     "VaR": z * vols.mean(),
                     "Volatilitas GARCH saja": cond_fut[h - 1]})
    return pd.DataFrame(rows), cond_hist


def convergence(trials):
    tr = trials.sort_values("number")
    best = tr["value"].cummin().values
    col = "user_attrs_train_sec" if "user_attrs_train_sec" in tr.columns else None
    if col:
        secs = tr[col].values
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

st.title("Prediksi Volatilitas Indeks LQ45 (GARCH-LSTM + VU + Optuna)")
st.caption(
    f"Model dilatih dengan {meta['ticker']} {meta['start']} s.d. {meta['end']} "
    f"(data latih sampai {c['train_end_date']}) | target: realized volatility {w} hari "
    f"({meta.get('target_mode', 'backward')}) | model diekspor {meta['created_at'][:10]}"
)

tab_pred, tab_test, tab_tbl, tab_search = st.tabs(
    ["Prediksi", "Evaluasi data test", "Tabel hasil", "Pencarian hyperparameter"]
)

# ---------------------------------------------------------------- Tab 1
with tab_pred:
    col1, col2 = st.columns([2, 1])
    model_name = col1.selectbox("Model", names, index=default_i)
    dist = col2.radio("Distribusi VaR", ["normal", "empiris"], horizontal=True)
    z = c["var_z"][dist]

    cc1, cc2 = st.columns(2)
    use_csv = cc1.checkbox("Pakai harga_terbaru.csv", value=True)
    use_live = cc2.checkbox("Tambah data yfinance setelah CSV", value=False)

    close, sources = build_close_series(art, use_csv, use_live)
    returns = np.log(close / close.shift(1)).dropna()
    last_date = returns.index[-1]

    st.info("Sumber data tambahan: " + (", ".join(sources) if sources else "tidak ada, memakai snapshot saja"))
    stale = (pd.Timestamp.today().normalize() - last_date).days
    if stale > 7:
        st.warning(f"Data terakhir {last_date.date()} sudah {stale} hari lalu. "
                   "Perbarui harga_terbaru.csv agar prediksi untuk tanggal sekarang bermakna.")
    recent = returns.tail(10)
    big = recent[recent.abs() > EXTREME_RET]
    if len(big):
        st.warning("Return harian ekstrem pada " +
                   ", ".join(f"{d.date()} ({v:+.1%})" for d, v in big.items()) +
                   ". Pastikan datanya benar; prediksi volatilitas akan sangat terpengaruh.")

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

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Harga terakhir", f"{close.iloc[-1]:,.2f}", help=str(last_date.date()))
    m2.metric("Horizon", f"{horizon} hari bursa")
    m3.metric(f"Volatilitas harian {target_date}", f"{last['Volatilitas GARCH-LSTM']:.2%}",
              help=f"Rata-rata {meta['n_runs']} run; simpangan antar-run {last['Simpangan antar-run']:.2%}")
    m4.metric(f"VaR {conf_pct}% 1 hari (z={z:.2f})", f"{last['VaR']:.2%}")

    st.subheader("Lintasan ramalan sampai tanggal target")
    st.line_chart(fc_df[["Volatilitas GARCH-LSTM", "Volatilitas GARCH saja"]])
    st.dataframe(fc_df.style.format("{:.2%}"), use_container_width=True)

    st.subheader("Histori terakhir")
    hist = pd.DataFrame({"Volatilitas kondisional GARCH": cond}, index=returns.index)
    hist[f"Realized volatility ({w} hari)"] = returns.rolling(w).std()
    st.line_chart(hist.tail(250))

    st.caption(
        "Hari bursa dihitung Senin sampai Jumat; libur BEI belum dikecualikan. "
        "Untuk horizon lebih dari 1 hari, return masa depan diasumsikan 0 dan volatilitas kondisional "
        "memakai peramalan GARCH, sehingga prediksi mendekati GARCH murni dan makin tidak pasti. "
        "Parameter GARCH dan scaler tetap dari data latih (tidak di-refit pada data baru)."
    )

# ---------------------------------------------------------------- Tab 2
with tab_test:
    t = art["test"]
    dates = pd.to_datetime(t["dates"])
    y_true = np.asarray(t["y_true"]).ravel()
    model_eval = st.selectbox("Model", names, index=default_i, key="eval_model")

    pm = t["pred_mean"]
    pred = np.asarray(pm[model_eval] if isinstance(pm, dict) else pm).ravel()
    df_eval = pd.DataFrame({"Aktual": y_true, "Prediksi": pred}, index=dates)
    st.line_chart(df_eval)

    err = pred - y_true
    e1, e2, e3 = st.columns(3)
    e1.metric("RMSE", f"{np.sqrt(np.mean(err ** 2)):.5f}")
    e2.metric("MAE", f"{np.mean(np.abs(err)):.5f}")
    e3.metric("Jumlah observasi test", f"{len(y_true)}")

    if "actual_returns" in t and "var" in t:
        v = t["var"]
        if isinstance(v, dict) and model_eval in v:
            v = v[model_eval]
        if isinstance(v, dict):
            v = v.get(dist, next(iter(v.values())))
        v = np.asarray(v).ravel()
        ar = np.asarray(t["actual_returns"]).ravel()
        if len(v) == len(ar):
            df_var = pd.DataFrame({"Return aktual": ar, f"VaR {conf_pct}%": v}, index=dates)
            st.subheader(f"VaR {conf_pct}% vs return aktual")
            st.line_chart(df_var)
            st.caption(f"Pelanggaran: {(ar < v).sum()} dari {len(ar)} "
                       f"(diharapkan {(1 - c['confidence']) * len(ar):.1f})")

# ---------------------------------------------------------------- Tab 3
with tab_tbl:
    tbl = art["tables"]
    for key, title in [("results", "Hasil evaluasi"),
                       ("significance", "Uji Diebold-Mariano"),
                       ("efficiency", "Efisiensi pencarian"),
                       ("search_dist", "Distribusi multi-seed"),
                       ("backtest", "Backtest VaR")]:
        if key in tbl:
            st.subheader(title)
            st.dataframe(tbl[key], use_container_width=True)

# ---------------------------------------------------------------- Tab 4
with tab_search:
    tbl = art["tables"]
    if "trials_grid" in tbl and "trials_optuna" in tbl:
        bg, tg = convergence(tbl["trials_grid"])
        bo, to = convergence(tbl["trials_optuna"])
        st.subheader("Konvergensi skor terbaik terhadap waktu")
        conv = pd.concat([
            pd.DataFrame({"Grid": bg}, index=np.round(tg, 2)),
            pd.DataFrame({"Optuna": bo}, index=np.round(to, 2)),
        ]).sort_index().ffill()
        conv.index.name = "menit"
        st.line_chart(conv)
        st.subheader("Trial Grid Search")
        st.dataframe(tbl["trials_grid"], use_container_width=True)
        st.subheader("Trial Optuna")
        st.dataframe(tbl["trials_optuna"], use_container_width=True)
    st.subheader("Ruang pencarian")
    st.json(c["search_space"])
