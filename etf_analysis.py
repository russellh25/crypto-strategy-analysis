import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request

# Constants
EVENT_TYPE_PRIORITY = {
    "approval": 3,
    "launch": 2,
    "major_signal": 1,
    "filing": 0,
}


def _download_json(url: str) -> dict:
    with urllib.request.urlopen(url) as response:  # nosec: B310 - trusted endpoint
        return json.loads(response.read().decode())


def download_ohlc(coin_id: str, vs_currency: str = "usd") -> pd.DataFrame:
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency={vs_currency}&days=max"
    )
    raw = _download_json(url)
    ohlc = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close"])
    ohlc["date"] = pd.to_datetime(ohlc["timestamp"], unit="ms").dt.date
    daily = (
        ohlc.groupby("date")[["open", "high", "low", "close"]]
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .reset_index()
    )
    return daily


def download_volumes(coin_id: str, vs_currency: str = "usd") -> pd.DataFrame:
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        f"?vs_currency={vs_currency}&days=max&interval=daily"
    )
    raw = _download_json(url)
    volumes = pd.DataFrame(raw.get("total_volumes", []), columns=["timestamp", "volume"])
    volumes["date"] = pd.to_datetime(volumes["timestamp"], unit="ms").dt.date
    daily = volumes.groupby("date")["volume"].sum().reset_index()
    return daily


def load_price_history(cache_path: Path = Path("data/btc_usd_daily.csv")) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["date"])

    ohlc = download_ohlc("bitcoin")
    volume = download_volumes("bitcoin")
    df = pd.merge(ohlc, volume, on="date", how="left")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= "2017-01-01"].sort_values("date").reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    return df


def load_events(path: Path = Path("data/etf_events.csv")) -> pd.DataFrame:
    events = pd.read_csv(path, parse_dates=["event_date"])
    events["event_type"] = events["event_type"].str.strip().str.lower()
    return events


def align_event_date(event_date: pd.Timestamp, price_index: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    position = price_index.searchsorted(event_date)
    if position >= len(price_index):
        return None
    return price_index[position]


@dataclass
class EventWindowResult:
    event_date: pd.Timestamp
    trading_date: pd.Timestamp
    asset: str
    event_type: str
    short_description: str
    returns: Dict[int, float]
    pre_vol: float
    post_vol: float
    cumulative_curve: pd.Series
    drawdown_curve: pd.Series


def compute_event_windows(prices: pd.DataFrame, events: pd.DataFrame, window: int = 30) -> List[EventWindowResult]:
    close = prices.set_index("date")["close"]
    daily_log_returns = np.log(close).diff()
    results: List[EventWindowResult] = []
    for _, event in events.iterrows():
        trading_date = align_event_date(event["event_date"], close.index)
        if trading_date is None:
            continue
        center_loc = close.index.get_loc(trading_date)
        if center_loc - window < 0 or center_loc + window >= len(close):
            continue
        returns: Dict[int, float] = {}
        for offset in [-30, -7, -1, 0, 1, 7, 30]:
            start_idx = center_loc
            end_idx = center_loc + offset
            start_px = close.iloc[min(start_idx, end_idx)]
            end_px = close.iloc[max(start_idx, end_idx)]
            returns[offset] = end_px / start_px - 1
        pre_slice = daily_log_returns.iloc[center_loc - window : center_loc]
        post_slice = daily_log_returns.iloc[center_loc + 1 : center_loc + window + 1]
        pre_vol = float(pre_slice.std() * np.sqrt(252))
        post_vol = float(post_slice.std() * np.sqrt(252))
        window_range = slice(center_loc - window, center_loc + window + 1)
        rel_index = np.arange(-window, window + 1)
        cumulative = (close.iloc[window_range] / close.iloc[center_loc]) - 1
        cumulative.index = rel_index
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / (1 + running_max)
        results.append(
            EventWindowResult(
                event_date=event["event_date"],
                trading_date=trading_date,
                asset=event["asset"],
                event_type=event["event_type"],
                short_description=event["short_description"],
                returns=returns,
                pre_vol=pre_vol,
                post_vol=post_vol,
                cumulative_curve=cumulative,
                drawdown_curve=drawdown,
            )
        )
    return results


def summarize_event_results(results: List[EventWindowResult]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for res in results:
        record = {
            "event_date": res.event_date,
            "trading_date": res.trading_date,
            "asset": res.asset,
            "event_type": res.event_type,
            "short_description": res.short_description,
            "pre_vol": res.pre_vol,
            "post_vol": res.post_vol,
        }
        for offset, value in res.returns.items():
            record[f"ret_{offset:+d}"] = value
        records.append(record)
    per_event = pd.DataFrame(records)

    summary_records = []
    for offset in [-30, -7, -1, 0, 1, 7, 30]:
        col = f"ret_{offset:+d}"
        summary_records.append(
            {
                "window": offset,
                "mean": per_event[col].mean(),
                "median": per_event[col].median(),
            }
        )
    summary = pd.DataFrame(summary_records)
    return per_event, summary


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))


def _welch_t_test(sample_a: pd.Series, sample_b: pd.Series) -> Tuple[float, float]:
    mean_a, mean_b = sample_a.mean(), sample_b.mean()
    var_a, var_b = sample_a.var(ddof=1), sample_b.var(ddof=1)
    n_a, n_b = len(sample_a), len(sample_b)
    denom = np.sqrt(var_a / n_a + var_b / n_b)
    if denom == 0:
        return np.nan, np.nan
    t_stat = (mean_a - mean_b) / denom
    # Normal approximation for two-sided p-value
    p_val = 2 * (1 - _normal_cdf(abs(t_stat)))
    return t_stat, p_val


def _mann_whitney(sample_a: pd.Series, sample_b: pd.Series) -> Tuple[float, float]:
    combined = pd.concat([sample_a, sample_b], ignore_index=True)
    ranks = combined.rank(method="average")
    ranks_a = ranks.iloc[: len(sample_a)]
    ranks_b = ranks.iloc[len(sample_a) :]
    u_a = ranks_a.sum() - len(sample_a) * (len(sample_a) + 1) / 2
    u_b = ranks_b.sum() - len(sample_b) * (len(sample_b) + 1) / 2
    u_stat = min(u_a, u_b)
    mean_u = len(sample_a) * len(sample_b) / 2
    std_u = np.sqrt(len(sample_a) * len(sample_b) * (len(sample_a) + len(sample_b) + 1) / 12)
    if std_u == 0:
        return u_stat, np.nan
    z = (u_stat - mean_u) / std_u
    p_val = 2 * (1 - _normal_cdf(abs(z)))
    return u_stat, p_val


def compute_t_tests(per_event: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in [1, 7, 30]:
        pre_col = "ret_-1" if window == 1 else "ret_-7" if window == 7 else "ret_-30"
        post_col = f"ret_+{window}"
        pre = per_event[pre_col].dropna()
        post = per_event[post_col].dropna()
        t_stat, p_val = _welch_t_test(post, pre)
        u_stat, u_p = _mann_whitney(post, pre)
        rows.append(
            {
                "window": window,
                "t_stat": t_stat,
                "t_p_value": p_val,
                "mann_whitney_u": u_stat,
                "mann_whitney_p": u_p,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    net_return: float
    event_type: str
    label: str


def _resolve_event_overlaps(events: pd.DataFrame, horizon: int, price_index: pd.DatetimeIndex) -> List[pd.Series]:
    resolved: List[pd.Series] = []
    for _, event in events.sort_values("event_date").iterrows():
        trading_date = align_event_date(event["event_date"], price_index)
        if trading_date is None:
            continue
        candidate_start = trading_date
        candidate_end = price_index[min(price_index.searchsorted(trading_date) + horizon, len(price_index) - 1)]
        while resolved and candidate_start <= resolved[-1]["exit_date"]:
            last = resolved[-1]
            last_priority = EVENT_TYPE_PRIORITY.get(last["event_type"], -1)
            new_priority = EVENT_TYPE_PRIORITY.get(event["event_type"], -1)
            if new_priority > last_priority:
                resolved.pop()
            else:
                break
        else:
            pass
        if resolved and candidate_start <= resolved[-1]["exit_date"]:
            continue
        event_copy = event.copy()
        event_copy["trading_date"] = trading_date
        event_copy["exit_date"] = candidate_end
        resolved.append(event_copy)
    return resolved


def backtest_strategy(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    horizon: int,
    entry_shift: int = 0,
    filter_types: Optional[List[str]] = None,
) -> Tuple[List[Trade], pd.Series]:
    close = prices.set_index("date")["close"]
    daily_returns = close.pct_change().fillna(0)
    selected = events if filter_types is None else events[events["event_type"].isin(filter_types)]
    resolved_events = _resolve_event_overlaps(selected, horizon, close.index)
    trades: List[Trade] = []
    position = pd.Series(0.0, index=daily_returns.index)

    for event in resolved_events:
        trading_date: pd.Timestamp = event["trading_date"]
        entry_idx = close.index.get_loc(trading_date) + entry_shift
        if entry_idx < 0 or entry_idx >= len(close):
            continue
        exit_idx = entry_idx + horizon
        if exit_idx >= len(close):
            continue
        entry_date = close.index[entry_idx]
        exit_date = close.index[exit_idx]
        entry_price = float(close.iloc[entry_idx])
        exit_price = float(close.iloc[exit_idx])
        gross_return = exit_price / entry_price - 1
        net_return = gross_return - 0.002
        trades.append(
            Trade(
                entry_date=entry_date,
                exit_date=exit_date,
                entry_price=entry_price,
                exit_price=exit_price,
                net_return=net_return,
                event_type=event["event_type"],
                label=event["short_description"],
            )
        )
        position.iloc[entry_idx + 1 : exit_idx + 1] = 1.0
        position.iloc[entry_idx + 1] -= 0.001
        position.iloc[exit_idx] -= 0.001

    strategy_returns = position * daily_returns
    return trades, strategy_returns


def performance_summary(trades: List[Trade], strategy_returns: pd.Series) -> Dict[str, float]:
    equity = (1 + strategy_returns).cumprod()
    total_periods = len(strategy_returns)
    cagr = float(equity.iloc[-1] ** (252 / total_periods) - 1)
    ann_vol = float(strategy_returns.std() * np.sqrt(252))
    sharpe = float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else np.nan
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    mdd = float(drawdown.min())
    trade_returns = pd.Series([t.net_return for t in trades])
    hit_rate = float((trade_returns > 0).mean()) if not trade_returns.empty else np.nan
    avg_trade_return = float(trade_returns.mean()) if not trade_returns.empty else np.nan
    return {
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "hit_rate": hit_rate,
        "avg_trade_return": avg_trade_return,
        "trade_count": len(trades),
        "final_value": float(equity.iloc[-1]),
    }


def plot_event_price(prices: pd.DataFrame, events: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prices["date"], prices["close"], label="Close")
    for _, event in events.iterrows():
        ax.axvline(event["event_date"], color="red", linestyle="--", alpha=0.6)
        ax.text(event["event_date"], ax.get_ylim()[1], event["event_type"], rotation=90, va="top", fontsize=8)
    ax.set_title("BTC price with ETF milestones")
    ax.set_ylabel("USD")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_equity(strategy_returns: pd.Series, path: Path, title: str) -> None:
    equity = (1 + strategy_returns).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(equity.index, equity, label="Equity Curve")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].set_title(title)
    axes[1].fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def describe_trades(trades: List[Trade]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "event_type": t.event_type,
                "label": t.label,
                "net_return": t.net_return,
            }
            for t in trades
        ]
    )
