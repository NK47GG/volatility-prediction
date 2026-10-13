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


@st.cache_data(ttl=3600, show_spinner="Mengambil data terbaru dari yfinance...")
def fetch_new_close(ticker, start):
    import yfinance as yf

    raw = yf.download(ticker, start=start, progress=False)
    if raw is None or len(raw) == 0:
        return pd.Series(dtype=float)
    return pd.Series(np.asarray(raw["Close"]).ravel(), index=pd.DatetimeIndex(raw.index)).dropna()


def build_returns(art, use_live):
    # gabungkan snapshot + data baru yfinance. Return (log_return, pesan_error_atau_None)
    base = art["data"]["log_return"]
    base_close = art["data"]["Close"]
    if not use_live:
        return base, None
    try:
        last = base_close.index[-1]
        new_close = fetch_new_close(art["meta"]["ticker"], (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        new_close = new_close[new_close.index > last]
        # buang bar hari ini jika pasar (WIB) belum tutup, supaya tidak memakai harga intraday
        now = pd.Timestamp.now(tz="Asia/Jakarta")
        if len(new_close) and new_close.index[-1].date() == now.date() and now.hour * 60 + now.minute < 16 * 60 + 30:
            new_close = new_close.iloc[:-1]
        if new_close.empty:
            return base, None
        chain = pd.concat([base_close.iloc[-1:], new_close])
        new_ret = np.log(chain / chain.shift(1)).dropna()
        return pd.concat([base, new_ret]), None
    except Exception as e:
        return base, str(e)


def predict_for_date(art, returns, target_date, model_name):
    # Model satu langkah: memprediksi volatilitas satu hari perdagangan dari data SEBELUM hari itu.
    c = art["config"]
    td = pd.Timestamp(target_date)
    on_or_after = returns.index[returns.index >= td]
    if len(on_or_after) > 0:
        eff = on_or_after[0]                      # hari perdagangan yang sudah punya data aktual
        hist = returns[returns.index < eff]
    else:
        eff = None                                # tanggal setelah data terakhir: prakiraan murni
        hist = returns
        gap = (td - hist.index[-1]).days
        if gap > 5:
            raise ValueError(
                f"Data terakhir yang tersedia {hist.index[-1].date()}. Model hanya memprediksi 1 hari perdagangan "
                f"setelah data terakhir, jadi tanggal {td.date()} terlalu jauh."
            )
    if len(hist) < c["timestep"]:
        raise ValueError(f"Tanggal terlalu awal: butuh minimal {c['timestep']} hari data sebelumnya.")
    vol, var, cond = forecast_next(art, hist.values, model_name)
    realized = returns.rolling(c["realized_vol_window"]).std()
    out = {"eff": eff, "hist": hist, "cond": cond, "vol": vol, "var": var, "realized": realized}
    if eff is not None:
        out["actual_vol"] = float(realized.loc[eff])
        out["actual_ret"] = float(returns.loc[eff])
        out["in_sample"] = eff <= pd.Timestamp(c["train_end_date"])
    return out


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

# ---- Tab 1: prediksi untuk tanggal tertentu
with tab_pred:
    cA, cB = st.columns([1, 1])
    model_name = cA.selectbox("Model", names, index=default_i)
    use_live = cB.checkbox("Pakai data terbaru dari yfinance", value=True)

    returns, live_err = build_returns(art, use_live)
    if live_err:
        st.warning(f"Gagal mengambil data yfinance, memakai snapshot saja. ({live_err})")
    last_data = returns.index[-1].date()
    today = pd.Timestamp.now(tz="Asia/Jakarta").date()

    target = st.date_input(
        "Tanggal yang ingin diprediksi",
        value=today,
        min_value=returns.index[c["timestep"]].date(),
    )
    st.caption(f"Data tersedia sampai {last_data}.")

    try:
        res = predict_for_date(art, returns, target, model_name)
    except ValueError as e:
        st.error(str(e))
        res = None

    if res is not None:
        w = c["realized_vol_window"]
        if res["eff"] is None:
            next_day = (pd.Timestamp(last_data) + pd.offsets.BDay(1)).date()  # hari kerja berikutnya (libur bursa tidak dikenali)
            if target > next_day:
                st.warning(
                    f"Untuk memprediksi {target} dibutuhkan return hari sebelumnya yang belum tersedia "
                    f"(data terakhir {last_data}). Hasil di bawah adalah prediksi untuk {next_day}, bukan {target}."
                )
            else:
                st.info(f"Prakiraan murni untuk {next_day}, dihitung dari data sampai {last_data}.")
            m1, m2 = st.columns(2)
            m1.metric(f"Prediksi volatilitas {next_day}", f"{res['vol']:.2%}")
            m2.metric(f"VaR {conf_pct}% (1 hari)", f"{res['var']:.2%}")
        else:
            eff = res["eff"]
            if eff.date() != target:
                st.info(f"{target} bukan hari perdagangan. Dipakai hari perdagangan berikutnya: {eff.date()}.")
            if res["in_sample"]:
                st.warning("Tanggal ini masuk data latih (in-sample). Hasil tidak bisa dipakai sebagai bukti performa model.")
            else:
                st.success("Tanggal ini di luar data latih (out-of-sample). Prediksi hanya memakai data sebelum tanggal tersebut.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Prediksi volatilitas", f"{res['vol']:.2%}")
            m2.metric(f"Realized volatility aktual ({w} hari)", f"{res['actual_vol']:.2%}")
            m3.metric("Selisih (prediksi - aktual)", f"{res['vol'] - res['actual_vol']:+.2%}")
            m4, m5 = st.columns(2)
            m4.metric(f"VaR {conf_pct}% (1 hari)", f"{res['var']:.2%}")
            m5.metric(
                "Log return aktual",
                f"{res['actual_ret']:.2%}",
                delta="Melewati VaR (exceedance)" if res["actual_ret"] < res["var"] else "Di dalam VaR",
                delta_color="off",
            )

        hist = pd.DataFrame({"Volatilitas kondisional GARCH": res["cond"]}, index=res["hist"].index)
        hist[f"Realized volatility ({w} hari)"] = res["realized"].reindex(res["hist"].index)
        st.line_chart(hist.tail(250))
        st.caption(
            "Target model adalah realized volatility (simpangan baku log return rolling). "
            "VaR parametrik: asumsi distribusi normal dan mean return nol. "
            "Parameter GARCH dan bobot model tetap dari data latih (tidak dilatih ulang)."
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
