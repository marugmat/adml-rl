from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    # Moving averages
    df["ma_10"] = close.rolling(10).mean()
    df["ema_10"] = close.ewm(span=10, adjust=False).mean()

    # RSI (14)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands (20)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (ma20 + 1e-9)

    # Volume indicators (if available)
    if "Volume" in df.columns:
        df["vol_ma_20"] = df["Volume"].rolling(20).mean()
        df["vol_ratio"] = df["Volume"] / (df["vol_ma_20"] + 1e-9)

    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df


def plot_technical_indicators(df: pd.DataFrame, title_prefix: str = "") -> None:
    plt.figure(figsize=(16, 12))
    n = 5
    plt.subplot(n, 1, 1)
    plt.plot(df["Close"], label="Close")
    if "ma_10" in df.columns:
        plt.plot(df["ma_10"], label="MA 10")
    if "ema_10" in df.columns:
        plt.plot(df["ema_10"], label="EMA 10")
    plt.title(f"{title_prefix}Price & Moving Averages")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(n, 1, 2)
    if "rsi_14" in df.columns:
        plt.plot(df["rsi_14"], label="RSI 14", color="purple")
        plt.axhline(70, color="red", linestyle="--", alpha=0.5)
        plt.axhline(30, color="green", linestyle="--", alpha=0.5)
        plt.title(f"{title_prefix}RSI (14)")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.axis("off")

    plt.subplot(n, 1, 3)
    if "macd" in df.columns and "macd_signal" in df.columns:
        plt.plot(df["macd"], label="MACD", color="blue")
        plt.plot(df["macd_signal"], label="Signal", color="orange")
        if "macd_hist" in df.columns:
            plt.bar(df.index, df["macd_hist"], label="Hist", color="gray", alpha=0.3)
        plt.title(f"{title_prefix}MACD")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.axis("off")

    plt.subplot(n, 1, 4)
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        plt.plot(df["Close"], label="Close", color="black")
        plt.plot(df["bb_upper"], label="BB Upper", color="red", linestyle="--")
        plt.plot(df["bb_lower"], label="BB Lower", color="blue", linestyle="--")
        plt.fill_between(df.index, df["bb_lower"], df["bb_upper"], color="gray", alpha=0.1)
        plt.title(f"{title_prefix}Bollinger Bands")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.axis("off")

    plt.subplot(n, 1, 5)
    if "Volume" in df.columns:
        plt.plot(df["Volume"], label="Volume", color="gray")
        if "vol_ma_20" in df.columns:
            plt.plot(df["vol_ma_20"], label="Vol MA 20", color="blue")
        if "vol_ratio" in df.columns:
            plt.plot(df["vol_ratio"], label="Vol Ratio", color="orange")
        plt.title(f"{title_prefix}Volume & Volume Indicators")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.axis("off")

    plt.tight_layout()
    plt.show()
