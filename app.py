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
def build_close_series(_art):
    close = _art["data"]["Close"].copy()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index))
    try:
        s = load_csv_close()
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
        close = pd.concat([close, s[s.index > close.index[-1]]])
    except Exception as ex:
        st.warning(f"harga_terbaru.csv tidak terbaca, memakai snapshot saja. ({ex})")
    return close[~close.index.duplicated(keep="last")].sort_index()


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

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Pengaturan")
    model_name = st.selectbox("Model", names, index=default_i)
    dist = st.radio("Distribusi VaR", ["normal", "empiris"], horizontal=True)
    z = c["var_z"][dist]
    st.divider()
    close = build_close_series(art)
    returns = np.log(close / close.shift(1)).dropna()
    last_date = returns.index[-1]
    st.caption(f"Data sampai **{last_date.date()}**  \nHarga terakhir **{close.iloc[-1]:,.2f}**")
    st.caption(f"Timestep {c['timestep']} hari · {meta['n_runs']} run ensemble")

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
    stale = (pd.Timestamp.today().normalize() - last_date).days
    if stale > 7:
        st.warning(f"Data terakhir {last_date.date()} sudah {stale} hari lalu; "
                   "prediksi untuk tanggal sekarang mungkin tidak bermakna.")
    recent = returns.tail(10)
    big = recent[recent.abs() > EXTREME_RET]
    if len(big):
        st.warning("Return harian ekstrem: " +
                   ", ".join(f"{d.date()} ({v:+.1%})" for d, v in big.items()) +
                   ". Pastikan datanya benar.")

    min_d = (last_date + pd.offsets.BDay(1)).date()
    max_d = (last_date + pd.offsets.BDay(MAX_HORIZON)).date()
    left, right = st.columns([1, 2])
    with left:
        target_date = st.date_input("Prediksi untuk tanggal", value=min_d,
                                    min_value=min_d, max_value=max_d)
    fut_dates = pd.bdate_range(last_date, target_date)[1:]
    horizon = len(fut_dates)
    with right:
        if horizon <= c["timestep"]:
            st.success(f"Horizon {horizon} hari bursa — jendela input masih memuat data pasar nyata.")
        else:
            st.info(f"Horizon {horizon} hari bursa — melebihi timestep {c['timestep']}; "
                    "seluruh input sudah berupa proyeksi GARCH, hasil mendekati volatilitas jangka panjang.")

    fc_df, cond = forecast_horizon(art, returns.values, model_name, z, horizon)
    fc_df.index = fut_dates
    last = fc_df.iloc[-1]

    st.markdown(f"### {target_date.strftime('%A, %d %B %Y')}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Volatilitas harian (GARCH-LSTM)", f"{last['Volatilitas GARCH-LSTM']:.2%}",
              help=f"Rata-rata {meta['n_runs']} run; simpangan antar-run {last['Simpangan antar-run']:.2%}")
    m2.metric("Volatilitas harian (GARCH saja)", f"{last['Volatilitas GARCH saja']:.2%}",
              delta=f"{last['Volatilitas GARCH-LSTM'] - last['Volatilitas GARCH saja']:+.2%}",
              delta_color="off")
    m3.metric(f"VaR {conf_pct}% 1 hari", f"{last['VaR']:.2%}",
              help=f"z = {z:.3f} ({dist}); potensi rugi harian terburuk pada tingkat keyakinan {conf_pct}%")

    if horizon > 1:
        st.subheader("Lintasan ramalan")
        st.line_chart(fc_df[["Volatilitas GARCH-LSTM", "Volatilitas GARCH saja"]])
        with st.expander("Tabel per hari"):
            st.dataframe(fc_df.style.format("{:.2%}"), use_container_width=True)

    st.subheader("Setahun terakhir")
    hist = pd.DataFrame({"Volatilitas kondisional GARCH": cond}, index=returns.index)
    hist[f"Realized volatility ({w} hari)"] = returns.rolling(w).std()
    st.line_chart(hist.tail(250))

    with st.expander("Catatan metodologi"):
        st.markdown(
            f"- Target: realized volatility {w} hari ({meta.get('target_mode', 'backward')}), "
            f"input {c['timestep']} hari terakhir.\n"
            "- Horizon > 1 hari dihitung rekursif: return masa depan diasumsikan 0, "
            "volatilitas kondisional dari peramalan GARCH.\n"
            "- Parameter GARCH dan scaler tetap dari data latih, tidak di-refit pada data baru.\n"
            "- Hari bursa = Senin–Jumat; libur BEI belum dikecualikan."
        )

# ---------------------------------------------------------------- Tab 2
with tab_test:
    t = art["test"]
    dates = pd.to_datetime(t["dates"])
    y_true = np.asarray(t["y_true"]).ravel()
    pm = t["pred_mean"]
    pred = np.asarray(pm[model_name] if isinstance(pm, dict) else pm).ravel()

    st.subheader(f"{model_name} — data test")
    st.line_chart(pd.DataFrame({"Aktual": y_true, "Prediksi": pred}, index=dates))

    err = pred - y_true
    e1, e2, e3 = st.columns(3)
    e1.metric("RMSE", f"{np.sqrt(np.mean(err ** 2)):.5f}")
    e2.metric("MAE", f"{np.mean(np.abs(err)):.5f}")
    e3.metric("Observasi test", f"{len(y_true)}")

    if "actual_returns" in t and "var" in t:
        v = t["var"]
        if isinstance(v, dict) and model_name in v:
            v = v[model_name]
        if isinstance(v, dict):
            v = v.get(dist, next(iter(v.values())))
        v = np.asarray(v).ravel()
        ar = np.asarray(t["actual_returns"]).ravel()
        if len(v) == len(ar):
            st.subheader(f"VaR {conf_pct}% vs return aktual")
            st.line_chart(pd.DataFrame({"Return aktual": ar, f"VaR {conf_pct}%": v}, index=dates))
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
