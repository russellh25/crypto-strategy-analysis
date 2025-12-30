import os
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# Ensure local src is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cmc_data import MissingAPIKeyError, get_ohlcv_daily  # noqa: E402
from event_study import (  # noqa: E402
    backtest_event_strategy,
    compute_event_windows,
    save_reports,
    summarize_car,
    trade_stats,
)


DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    return df


def load_prices(assets) -> Dict[str, pd.DataFrame]:
    prices: Dict[str, pd.DataFrame] = {}
    for asset in assets:
        sym = asset if asset != "MIX" else "BTC"
        try:
            prices[sym] = get_ohlcv_daily(sym)
        except MissingAPIKeyError as exc:
            cache_path = DATA_DIR / f"prices_{sym.lower()}_daily.csv"
            if cache_path.exists():
                cached = pd.read_csv(cache_path, parse_dates=["date"])
                cached["date"] = cached["date"].dt.date
                prices[sym] = cached
            else:
                raise MissingAPIKeyError(
                    f"{exc}. Provide cached prices at {cache_path} to proceed without downloads."
                )
    return prices


def main():
    events_path = DATA_DIR / "etf_events.csv"
    if not events_path.exists():
        raise SystemExit("Missing data/etf_events.csv. Please create it first.")
    events = load_events(events_path)
    assets = sorted(set(events["asset"].unique()) | {"BTC"})
    prices = load_prices(assets)

    results = compute_event_windows(prices, events, window=(-10, 10), market_symbol="BTC")
    summary = summarize_car(results.values())
    trades = backtest_event_strategy(prices, events, hold_days=1, entry_offset=-1, volatility_widening=True)

    REPORT_DIR.mkdir(exist_ok=True)
    save_reports(results, summary, trades, REPORT_DIR)

    stats = trade_stats(trades)
    stats_path = REPORT_DIR / "trade_stats.txt"
    stats_path.write_text("\n".join(f"{k}: {v}" for k, v in stats.items()))
    print("Event study complete. Reports in", REPORT_DIR)
    print(stats)


if __name__ == "__main__":
    main()

