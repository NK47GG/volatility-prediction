import pickle
from pathlib import Path

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
MAX_H = 250                                   # horizon maksimal prakiraan (hari kerja)
OUTLOOK_VER = 2   # naikkan tiap kali kolom/format hasil make_outlook berubah, supaya session_state lama tidak dipakai
CSV_PATH = Path(__file__).parent / "harga_terbaru.csv"   # cadangan bila yfinance diblokir (dibuat dari notebook)

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


def _to_close(raw):
    if raw is None or len(raw) == 0:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    idx = pd.DatetimeIndex(raw.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return pd.Series(np.asarray(raw["Close"]).ravel(), index=idx.normalize()).dropna()


@st.cache_data(ttl=3600, show_spinner="Mengambil data terbaru dari yfinance...")
def fetch_new_close(ticker, start):
    import yfinance as yf

    close = _to_close(yf.download(ticker, start=start, progress=False, threads=False))
    if close.empty:
        try:  # jalur lain di yfinance, kadang berhasil saat download() kosong
            close = _to_close(yf.Ticker(ticker).history(start=start, auto_adjust=False))
        except Exception:
            pass
    today = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None).normalize()
    if close.empty and (today - pd.Timestamp(start)).days >= 4:
        # kosong padahal sudah >4 hari: hampir pasti diblokir/rate limit. Raise agar TIDAK di-cache.
        raise RuntimeError("yfinance tidak mengembalikan data (kemungkinan rate limit di server)")
    return close


def load_csv_close():
    if not CSV_PATH.exists():
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    d = pd.read_csv(CSV_PATH)
    d["Date"] = pd.to_datetime(d["Date"])
    return pd.Series(d["Close"].astype(float).values, index=pd.DatetimeIndex(d["Date"])).dropna().sort_index()


def build_returns(art, use_live):
    # snapshot + harga_terbaru.csv (jika ada) + yfinance (jika aktif). Return (log_return, error_atau_None, info_sumber)
    base = art["data"]["log_return"]
    base_close = art["data"]["Close"]
    last = base_close.index[-1]
    csv_close = load_csv_close()
    csv_close = csv_close[csv_close.index > last]
    new_close = csv_close
    live = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    err = None
    if use_live:
        try:
            known_last = new_close.index[-1] if len(new_close) else last
            fetched = fetch_new_close(art["meta"]["ticker"], (known_last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
            live = fetched[fetched.index > known_last]
            new_close = pd.concat([new_close, live])
        except Exception as e:
            err = str(e)
    # buang bar hari ini jika pasar (WIB) belum tutup, supaya tidak memakai harga intraday
    now = pd.Timestamp.now(tz="Asia/Jakarta")
    if len(new_close) and new_close.index[-1].date() == now.date() and now.hour * 60 + now.minute < 16 * 60 + 30:
        new_close = new_close.iloc[:-1]
    info = f"Sumber data: snapshot sampai {last.date()}"
    if len(csv_close):
        info += f" | harga_terbaru.csv sampai {csv_close.index[-1].date()}"
    elif not CSV_PATH.exists():
        info += " | harga_terbaru.csv tidak ada"
    if use_live and err:
        info += " | yfinance GAGAL (lihat peringatan)"
    elif use_live:
        info += f" | yfinance: +{len(live)} hari" + (f" sampai {live.index[-1].date()}" if len(live) else " (tidak ada data baru)")
    else:
        info += " | yfinance dimatikan"
    if new_close.empty:
        return base, err, info
    chain = pd.concat([base_close.iloc[-1:], new_close])
    new_ret = np.log(chain / chain.shift(1)).dropna()
    return pd.concat([base, new_ret]), err, info


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


# >>> FORECAST_UTILS
# (segmen ini juga dipakai notebook Tahap 20 untuk validasi)
from numpy.lib.stride_tricks import sliding_window_view


def garch_param_dict(art):
    g = art["garch"]
    return dict(zip(g["param_names"], np.asarray(g["params"], dtype=float)))


def simulate_volatility(art, hist_returns, horizon, n_sims, entry, predict_fn, seed=0):
    # Prakiraan multi-hari: simulasikan jalur return dari GARCH, hitung fitur tiap jalur
    # ([log_return, conditional_volatility]), lalu prediksi realized volatility tiap hari dengan LSTM.
    # Return: (vol_lstm, vol_garch_kondisional, realized_vol_simulasi), tiap [n_sims, horizon], skala asli (std harian).
    c, g = art["config"], art["garch"]
    sc, T = g["returns_scale"], c["timestep"]
    r = np.asarray(hist_returns, dtype=float).ravel()
    gm = arch_model(pd.Series(r * sc), vol="GARCH", p=1, q=1, dist="normal").fix(g["params"])
    cond = np.asarray(gm.conditional_volatility) / sc
    rs = np.random.RandomState(seed)                  # rng eksplisit agar hasil reproducible (random_state diabaikan arch)
    # Selalu menarik 250 kolom lalu dipotong: guncangan hari ke-k sama persis berapa pun horizonnya,
    # jadi tanggal yang sama menghasilkan angka yang sama di prakiraan 1 minggu maupun 1 bulan.
    fc = gm.forecast(
        horizon=horizon, start=len(r) - 1, method="simulation", simulations=n_sims,
        reindex=False, rng=lambda size: rs.standard_normal((size[0], max(size[1], 250)))[:, : size[1]],
    )
    sim_ret = fc.simulations.values[-1] / sc
    sim_vol = np.sqrt(fc.simulations.variances[-1]) / sc
    tail = np.column_stack([r[-T:], cond[-T:]])                                   # T hari terakhir yang nyata
    ext = np.concatenate(
        [np.broadcast_to(tail, (n_sims, T, 2)), np.stack([sim_ret, sim_vol], axis=-1)], axis=1
    )                                                                              # (S, T+H, 2)
    flat = ext.reshape(-1, 2)
    scaled = art["scaler"].transform(np.column_stack([flat, np.zeros(len(flat))]))[:, :2]
    scaled = scaled.reshape(n_sims, T + horizon, 2)
    wins = sliding_window_view(scaled, T, axis=1)[:, :horizon]                    # (S, H, 2, T)
    wins = wins.transpose(0, 1, 3, 2).reshape(n_sims * horizon, T, 2)[:, :, entry["features"]]
    pred_scaled = np.asarray(predict_fn(np.ascontiguousarray(wins))).reshape(-1)
    dummy = np.zeros((len(pred_scaled), len(c["features"]) + 1))
    dummy[:, c["target_col_idx"]] = pred_scaled
    vol = art["scaler"].inverse_transform(dummy)[:, c["target_col_idx"]].reshape(n_sims, horizon)
    # Realized volatility simulasi (std rolling return, definisi yang sama dengan target LSTM) sebagai pembanding setara
    w = c["realized_vol_window"]
    rets = np.concatenate([np.broadcast_to(r[-(w - 1):], (n_sims, w - 1)), sim_ret], axis=1)
    sim_real = sliding_window_view(rets, w, axis=1).std(axis=-1, ddof=1)[:, :horizon]
    return vol, sim_vol, sim_real
# <<< FORECAST_UTILS


def make_outlook(art, returns, horizon, n_sims, model_name, seed=0):
    entry = art["models"][model_name]
    model = get_model(model_name)
    vol, sim_vol, sim_real = simulate_volatility(
        art, returns.values, horizon, n_sims, entry,
        lambda w: model.predict(w, verbose=0, batch_size=4096), seed,
    )
    dates = pd.bdate_range(returns.index[-1] + pd.Timedelta(days=1), periods=horizon)
    p = np.percentile(vol, [5, 50, 95], axis=0)
    gp = garch_param_dict(art)
    persist = gp["alpha[1]"] + gp["beta[1]"]
    long_run = np.sqrt(gp["omega"] / (1 - persist)) / art["garch"]["returns_scale"] if persist < 1 else np.nan
    out = pd.DataFrame(
        {
            "LSTM P5": p[0], "LSTM median": p[1], "LSTM P95": p[2],
            "GARCH realized (median)": np.median(sim_real, axis=0),
            "GARCH std kondisional": np.sqrt((sim_vol ** 2).mean(axis=0)),
            "Vol jangka panjang GARCH": long_run,
        },
        index=dates,
    )
    half_life = np.log(0.5) / np.log(persist) if 0 < persist < 1 else np.nan
    return out, persist, half_life


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

tab_pred, tab_fc, tab_test, tab_tbl, tab_search = st.tabs(
    ["Prediksi", "Prakiraan ke depan", "Evaluasi data test", "Tabel hasil", "Pencarian hyperparameter"]
)

# ---- Tab 1: prediksi untuk tanggal tertentu
with tab_pred:
    cA, cB = st.columns([1, 1])
    model_name = cA.selectbox("Model", names, index=default_i)
    use_live = cB.checkbox("Pakai data terbaru dari yfinance", value=True)

    returns, live_err, src_info = build_returns(art, use_live)
    if live_err:
        st.warning(f"Gagal mengambil data terbaru dari yfinance ({live_err}). Memakai data tersimpan sampai {returns.index[-1].date()}.")
    last_data = returns.index[-1].date()
    today = pd.Timestamp.now(tz="Asia/Jakarta").date()

    target = st.date_input(
        "Tanggal yang ingin diprediksi",
        value=today,
        min_value=returns.index[c["timestep"]].date(),
    )
    st.caption(f"Data tersedia sampai {last_data}. {src_info}")

    try:
        res = predict_for_date(art, returns, target, model_name)
    except ValueError as e:
        st.error(str(e))
        res = None

    if res is not None:
        w = c["realized_vol_window"]
        if res["eff"] is None:
            next_day = (pd.Timestamp(last_data) + pd.offsets.BDay(1)).date()  # hari kerja berikutnya (libur bursa tidak dikenali)
            if target <= next_day:
                st.info(f"Prakiraan 1 langkah untuk {next_day}, dihitung dari data sampai {last_data}.")
                m1, m2 = st.columns(2)
                m1.metric(f"Prediksi volatilitas {next_day}", f"{res['vol']:.2%}")
                m2.metric(f"VaR {conf_pct}% (1 hari)", f"{res['var']:.2%}")
            else:
                # tanggal lebih jauh dari hari kerja berikutnya: simulasi GARCH + LSTM sampai tanggal itu
                start_d = returns.index[-1] + pd.offsets.Day(1)
                bd = pd.bdate_range(start_d, pd.Timestamp(target))
                n_days = len(bd) + (1 if bd[-1].date() < target else 0)   # target akhir pekan: pakai hari kerja berikutnya
                eff_f = pd.bdate_range(start_d, periods=n_days)[-1]
                if n_days > MAX_H:
                    st.error(f"Tanggal terlalu jauh. Horizon maksimal {MAX_H} hari kerja (sekitar 1 tahun).")
                else:
                    key1 = (OUTLOOK_VER, model_name, str(eff_f.date()), str(returns.index[-1]))
                    if st.session_state.get("single_outlook", (None, None))[0] != key1:
                        with st.spinner("Menyimulasikan jalur return dan memprediksi dengan LSTM..."):
                            st.session_state["single_outlook"] = (key1, make_outlook(art, returns, n_days, 500, model_name))
                    df_f, persist_f, half_f = st.session_state["single_outlook"][1]
                    row = df_f.iloc[-1]
                    if eff_f.date() != target:
                        st.info(f"{target} bukan hari kerja. Dipakai hari kerja berikutnya: {eff_f.date()}.")
                    st.info(
                        f"Prakiraan {n_days} hari kerja ke depan (dari data sampai {last_data}) lewat simulasi "
                        f"GARCH + LSTM, nilai untuk {eff_f.date()}."
                    )
                    if (today - last_data).days > 4:
                        st.warning(f"Data harga terakhir {last_data}. Hari setelahnya ikut disimulasikan, bukan data aktual.")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric(f"LSTM median ({eff_f.date()})", f"{row['LSTM median']:.2%}")
                    m2.metric("Rentang P5 sampai P95", f"{row['LSTM P5']:.2%} sampai {row['LSTM P95']:.2%}")
                    m3.metric("GARCH realized (median)", f"{row['GARCH realized (median)']:.2%}")
                    m4.metric(f"VaR {conf_pct}% dari median", f"{norm.ppf(1 - c['confidence']) * row['LSTM median']:.2%}")
                    st.caption(
                        f"Makin jauh tanggalnya, prakiraan makin mendekati volatilitas jangka panjang "
                        f"({row['Vol jangka panjang GARCH']:.2%}). Rentang P5 sampai P95 adalah sebaran antar jalur simulasi, "
                        "bukan interval kepercayaan error model. Grafik lengkap ada di tab Prakiraan ke depan."
                    )
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

# ---- Tab 2: prakiraan beberapa hari ke depan
with tab_fc:
    last_known = returns.index[-1]
    first_day = (last_known + pd.offsets.BDay(1)).date()
    f1, f2, f3 = st.columns(3)
    fc_model = f1.selectbox("Model", names, index=default_i, key="fc_model")
    end_date = f2.date_input(
        "Prakiraan sampai tanggal",
        value=max(first_day, today + pd.Timedelta(days=7)),
        min_value=first_day,
        key="fc_end",
    )
    n_sims = f3.slider("Jumlah simulasi", 100, 1000, 500, 100, key="fc_sims")

    horizon = len(pd.bdate_range(last_known + pd.Timedelta(days=1), pd.Timestamp(end_date)))
    st.caption(f"Data tersedia sampai {last_data}. Horizon: {horizon} hari kerja ({first_day} sampai {end_date}).")
    if (today - last_data).days > 4:
        st.warning(
            f"Data harga terakhir {last_data}. Hari-hari setelahnya sampai hari ini ikut disimulasikan, bukan data aktual."
        )
    if horizon > MAX_H:
        st.error(f"Horizon maksimal {MAX_H} hari kerja (sekitar 1 tahun).")
    else:
        key = (OUTLOOK_VER, fc_model, str(end_date), n_sims, str(last_known))
        if st.button("Hitung prakiraan", key="fc_btn"):
            with st.spinner("Menyimulasikan jalur return dan memprediksi dengan LSTM..."):
                st.session_state["outlook"] = (key, make_outlook(art, returns, horizon, n_sims, fc_model))
        stored = st.session_state.get("outlook")
        if stored is None or stored[0] != key:
            st.info("Atur pilihan di atas, lalu klik **Hitung prakiraan**.")
        else:
            df_fc, persist, half_life = stored[1]
            last_row = df_fc.iloc[-1]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric(f"LSTM median ({df_fc.index[-1].date()})", f"{last_row['LSTM median']:.2%}")
            k2.metric("Rentang P5 sampai P95", f"{last_row['LSTM P5']:.2%} sampai {last_row['LSTM P95']:.2%}")
            k3.metric("GARCH realized (median)", f"{last_row['GARCH realized (median)']:.2%}")
            k4.metric("Vol jangka panjang", f"{last_row['Vol jangka panjang GARCH']:.2%}")
            st.line_chart(df_fc)
            st.dataframe(df_fc.apply(lambda col: col.map(lambda x: f"{x:.2%}")))
            export = (df_fc * 100).round(4)
            export.columns = [f"{col} (%)" for col in export.columns]
            export.index = export.index.strftime("%Y-%m-%d")
            export.index.name = "Tanggal"
            st.download_button(
                "Unduh CSV", export.to_csv().encode("utf-8"), file_name="prakiraan_volatilitas.csv", mime="text/csv"
            )
            st.caption(
                "Cara membaca: LSTM memprediksi **realized volatility** (std 5 hari), yang secara rata-rata sekitar 8 sampai 10% "
                "lebih rendah dari std kondisional GARCH dan volatilitas jangka panjang (keduanya berskala sigma). "
                "Pembanding yang setara dengan LSTM adalah kolom **GARCH realized (median)**. Hari pertama tidak punya rentang "
                "karena semua jalur memakai data nyata yang sama."
            )
            st.caption(
                f"Persistensi GARCH (alpha+beta) = {persist:.3f}"
                + (f", paruh-waktu sekitar {half_life:.0f} hari kerja" if np.isfinite(half_life) else "")
                + ". Makin jauh horizonnya, prakiraan makin mendekati volatilitas jangka panjang, "
                "sehingga informasi dari kondisi pasar saat ini memudar."
            )
            st.caption(
                "Cara kerja: jalur return masa depan disimulasikan dari GARCH, fitur tiap jalur dihitung, lalu LSTM "
                "memprediksi realized volatility tiap hari. Rentang P5 sampai P95 adalah sebaran antar jalur simulasi, "
                "bukan interval kepercayaan atas error model. Model dilatih dan dievaluasi untuk horizon 1 hari; "
                "akurasi multi-hari perlu divalidasi (notebook Tahap 20)."
            )

# ---- Tab 3: evaluasi data test
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

# ---- Tab 4: tabel hasil
with tab_tbl:
    st.subheader("Tabel A. Ringkasan tiap model")
    st.dataframe(art["tables"]["results"])
    st.subheader("Tabel B. Perbandingan berpasangan (error dan efisiensi)")
    st.dataframe(art["tables"]["significance"])
    st.subheader("Efisiensi pencarian")
    st.dataframe(art["tables"]["efficiency"])
    st.subheader("Setelan terbaik tiap model")
    st.dataframe(pd.DataFrame({n: e["params"] for n, e in art["models"].items()}).T.astype(str))

# ---- Tab 5: konvergensi pencarian
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
