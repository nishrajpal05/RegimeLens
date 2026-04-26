import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

NIFTY50_TICKER = "^NSEI"


def fetch_nifty50_daily(period_days: int = 500) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=period_days)

    tickers_to_try = ["^NSEI", "NSEI.NS", "^NSEBANK"]
    
    df = pd.DataFrame()
    for ticker_str in tickers_to_try:
        try:
            ticker = yf.Ticker(ticker_str)
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                timeout=30,
            )
            if not df.empty:
                print(f"[data] Using ticker: {ticker_str}")
                break
        except Exception as e:
            print(f"[data] {ticker_str} failed: {e}")
            continue

    if df.empty:
        try:
            df = yf.download(
                "^NSEI",
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            print(f"[data] yf.download also failed: {e}")

    if df.empty:
        raise ValueError(
            "Could not fetch Nifty 50 data. Try:\n"
            "  1. pip install --upgrade yfinance\n"
        )

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df.sort_index(inplace=True)
    df.dropna(subset=["Close"], inplace=True)

    print(f"[data] Fetched {len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def fetch_nifty50_intraday() -> pd.DataFrame:
    ticker = yf.Ticker(NIFTY50_TICKER)
    df = ticker.history(period="5d", interval="5m")

    if df.empty:
        print("[data] WARNING: Intraday data unavailable, returning empty DataFrame")
        return pd.DataFrame()

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Datetime"
    df.sort_index(inplace=True)
    df.dropna(subset=["Close"], inplace=True)

    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["minutes_from_open"] = (df["hour"] - 9) * 60 + df["minute"] - 15
    df["session_bucket"] = (df["minutes_from_open"] // 5).clip(0, 74)

    print(f"[data] Fetched {len(df)} intraday 5-min bars")
    return df
