from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency guard
    PANDAS_AVAILABLE = False

    class _PandasStub:  # pragma: no cover - typing aid
        DataFrame = object
        Series = object

    pd = _PandasStub()


@dataclass
class EventResult:
    event_id: int
    asset: str
    event_type: str
    event_date: pd.Timestamp
    window: Tuple[int, int]
    ar: pd.Series
    car: pd.Series


def _compute_log_returns(df: pd.DataFrame) -> pd.Series:
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas is required to compute event windows.")
    df = df.sort_values("date").drop_duplicates("date")
    closes = df["close"].astype(float)
    returns = (closes / closes.shift(1)).apply(lambda x: math.log(x) if pd.notnull(x) else None)
    returns.iloc[0] = 0.0
    returns.index = df["date"].values
    return returns


def _find_center_index(dates, event_date: pd.Timestamp) -> int:
    pos = dates.searchsorted(event_date.to_datetime64(), side="left")
    if pos >= len(dates):
        return -1
    return pos


def _window_indices(center: int, window: Tuple[int, int], length: int) -> Tuple[int, int]:
    start = center + window[0]
    end = center + window[1]
    if start < 0 or end >= length:
        return -1, -1
    return start, end


def compute_event_windows(
    prices: Dict[str, pd.DataFrame],
    events: pd.DataFrame,
    window: Tuple[int, int] = (-10, 10),
    market_symbol: str = "BTC",
) -> Dict[int, EventResult]:
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas is required to compute event windows.")
    results: Dict[int, EventResult] = {}
    log_returns = {sym: _compute_log_returns(df) for sym, df in prices.items()}
    date_arrays = {sym: df["date"].to_numpy() for sym, df in prices.items()}

    for row in events.itertuples():
        asset = row.asset
        price_sym = asset if asset in prices else market_symbol
        if price_sym not in prices:
            continue
        dates = date_arrays[price_sym]
        event_date = pd.to_datetime(row.event_date)
        center = _find_center_index(dates, event_date)
        if center < 0:
            continue
        start, end = _window_indices(center, window, len(dates))
        if start < 0:
            continue
        asset_lr = log_returns[price_sym].iloc[start : end + 1]
        if asset != market_symbol and market_symbol in log_returns:
            # align market to asset window dates
            market_lr = log_returns[market_symbol].reindex(asset_lr.index, method="nearest")
            ar = asset_lr - market_lr
        else:
            ar = asset_lr
        car = ar.cumsum()
        results[row.event_id] = EventResult(
            event_id=row.event_id,
            asset=asset,
            event_type=row.event_type,
            event_date=event_date,
            window=window,
            ar=ar,
            car=car,
        )
    return results


def summarize_car(results: Iterable[EventResult]) -> pd.DataFrame:
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas is required to summarize CAR.")
    records = []
    for res in results:
        for offset, val in res.car.items():
            records.append(
                {
                    "event_id": res.event_id,
                    "asset": res.asset,
                    "event_type": res.event_type,
                    "offset": offset,
                    "car": val,
                }
            )
    df = pd.DataFrame(records)
    if df.empty:
        return df
    grouped = df.groupby(["asset", "event_type", "offset"])  # type: ignore[arg-type]
    summary = grouped["car"].agg(["mean", "median", "count"]).reset_index()
    return summary


def backtest_event_strategy(
    prices: Dict[str, pd.DataFrame],
    events: pd.DataFrame,
    hold_days: int = 1,
    entry_offset: int = -1,
    volatility_widening: bool = False,
) -> pd.DataFrame:
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas is required to run the backtest.")
    trades = []
    per_asset_last_exit: Dict[str, pd.Timestamp] = {}
    cost_map = {"BTC": 0.0012, "ETH": 0.0012, "XRP": 0.003}

    for row in events.sort_values("event_date").itertuples():
        asset = row.asset
        price_sym = asset if asset in prices else "BTC"
        df = prices.get(price_sym)
        if df is None:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        dates = df["date"].to_numpy()
        event_date = pd.to_datetime(row.event_date)
        center_idx = _find_center_index(dates, event_date)
        if center_idx < 0:
            continue
        entry_idx = center_idx + entry_offset
        exit_idx = entry_idx + hold_days
        if entry_idx < 0 or exit_idx >= len(df):
            continue
        entry_date = pd.to_datetime(dates[entry_idx])
        exit_date = pd.to_datetime(dates[exit_idx])
        last_exit = per_asset_last_exit.get(price_sym)
        if last_exit is not None and entry_date <= last_exit:
            continue
        entry_price = float(df.loc[entry_idx, "close"])
        exit_price = float(df.loc[exit_idx, "close"])
        gross_ret = exit_price / entry_price - 1
        base_cost = cost_map.get(asset, 0.0015)
        if volatility_widening:
            event_day = pd.to_datetime(dates[center_idx])
            widen = 1.5 if entry_date <= event_day <= exit_date else 1.0
            cost = base_cost * widen
        else:
            cost = base_cost
        net_ret = gross_ret - cost
        trades.append(
            {
                "event_id": row.event_id,
                "asset": asset,
                "event_type": row.event_type,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_ret,
                "net_return": net_ret,
            }
        )
        per_asset_last_exit[price_sym] = exit_date
    return pd.DataFrame(trades)


def trade_stats(trades: pd.DataFrame) -> Dict[str, float]:
    if not PANDAS_AVAILABLE:
        return {"count": 0}
    if trades.empty:
        return {"count": 0}
    net = trades["net_return"].astype(float)
    return {
        "count": float(len(net)),
        "mean": net.mean(),
        "median": net.median(),
        "hit_rate": (net > 0).mean(),
        "best": net.max(),
        "worst": net.min(),
        "cumulative": (1 + net).prod() - 1,
    }


def save_reports(results: Dict[int, EventResult], summary: pd.DataFrame, trades: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save CAR per event
    car_records = []
    for res in results.values():
        for offset, val in res.car.items():
            car_records.append(
                {
                    "event_id": res.event_id,
                    "asset": res.asset,
                    "event_type": res.event_type,
                    "event_date": res.event_date.date(),
                    "offset": offset,
                    "car": val,
                }
            )
    pd.DataFrame(car_records).to_csv(out_dir / "car_by_event.csv", index=False)
    summary.to_csv(out_dir / "car_summary.csv", index=False)
    trades.to_csv(out_dir / "trades.csv", index=False)
    pd.DataFrame([trade_stats(trades)]).to_csv(out_dir / "trade_stats.csv", index=False)

