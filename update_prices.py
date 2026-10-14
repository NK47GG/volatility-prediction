import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "^JKLQ45"
START = "2026-01-01"          # sama dengan END di notebook (data setelah ini yang dibutuhkan app)
OUT = "harga_terbaru.csv"

close = pd.Series(dtype=float)
for attempt in range(1, 4):
    try:
        raw = yf.download(TICKER, start=START, progress=False)
        if raw is not None and len(raw) > 0:
            close = pd.Series(np.asarray(raw["Close"]).ravel(), index=pd.DatetimeIndex(raw.index)).dropna()
            if len(close):
                break
    except Exception as e:
        print(f"Percobaan {attempt} gagal: {e}")
    time.sleep(10 * attempt)

if close.empty:
    # keluar dengan error agar file lama TIDAK ditimpa dengan data kosong, dan GitHub menandai run gagal
    sys.exit("yfinance tidak mengembalikan data setelah 3 percobaan")

# buang bar hari ini jika bursa (WIB) belum tutup, supaya harga intraday tidak ikut tersimpan
now = pd.Timestamp.now(tz="Asia/Jakarta")
if close.index[-1].date() == now.date() and now.hour * 60 + now.minute < 16 * 60 + 30:
    close = close.iloc[:-1]

pd.DataFrame({"Date": close.index.strftime("%Y-%m-%d"), "Close": close.values}).to_csv(OUT, index=False)
print(f"{OUT}: {len(close)} hari, sampai {close.index[-1].date()}")
