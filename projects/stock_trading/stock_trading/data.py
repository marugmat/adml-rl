from __future__ import annotations

from typing import Tuple

import pandas as pd
import yfinance as yf


DEFAULT_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and return a cleaned DataFrame."""
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    cols = columns or DEFAULT_COLUMNS
    df = df[cols].dropna()
    return df


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/test partitions."""
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df
