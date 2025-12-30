import datetime as dt
from pathlib import Path
import sys

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency in this environment
    pd = None


if pd is None:  # pragma: no cover - environment without pandas
    import unittest

    raise unittest.SkipTest("pandas is required for event study tests")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from event_study import backtest_event_strategy, compute_event_windows, summarize_car


def sample_prices():
    if pd is None:
        return None
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    base = pd.DataFrame({
        "date": dates.date,
        "close": [100 + i for i in range(len(dates))],
        "open": 0,
        "high": 0,
        "low": 0,
        "volume": 0,
    })
    return base


def test_event_window_alignment():
    if pd is None:
        return
    prices = {"BTC": sample_prices()}
    events = pd.DataFrame({
        "event_id": [1],
        "asset": ["BTC"],
        "event_type": ["spot_btc_etf_trading_start"],
        "event_date": [dt.date(2024, 1, 11)],
    })
    results = compute_event_windows(prices, events, window=(-2, 2))
    assert 1 in results
    res = results[1]
    assert len(res.ar) == 5


def test_backtest_costs_applied():
    if pd is None:
        return
    prices = {"BTC": sample_prices()}
    events = pd.DataFrame({
        "event_id": [1, 2],
        "asset": ["BTC", "BTC"],
        "event_type": ["spot_btc_etf_trading_start", "spot_btc_etf_trading_start"],
        "event_date": [dt.date(2024, 1, 11), dt.date(2024, 1, 20)],
    })
    trades = backtest_event_strategy(prices, events, hold_days=1, entry_offset=-1, volatility_widening=False)
    assert not trades.empty
    assert all(trades["net_return"] < trades["gross_return"])


def test_summary_stats():
    if pd is None:
        return
    prices = {"BTC": sample_prices()}
    events = pd.DataFrame({
        "event_id": [1],
        "asset": ["BTC"],
        "event_type": ["spot_btc_etf_trading_start"],
        "event_date": [dt.date(2024, 1, 11)],
    })
    results = compute_event_windows(prices, events, window=(-2, 2))
    summary = summarize_car(results.values())
    assert not summary.empty

