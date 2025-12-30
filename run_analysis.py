import csv
import json
import math
import statistics
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

WINDOW = 30


def fetch_json(url: str):
    with urllib.request.urlopen(url) as resp:  # nosec: B310 - trusted endpoint
        return json.loads(resp.read().decode())


def fetch_price_data() -> List[Dict]:
    start = int(datetime(2016, 1, 1).timestamp())
    end = int(time.time())
    url = (
        "https://data.tradingview.com/history"
        f"?symbol=BITSTAMP:BTCUSD&resolution=D&from={start}&to={end}"
    )
    payload = fetch_json(url)
    if payload.get("s") != "ok":
        raise ValueError(f"TradingView history request failed with status: {payload.get('s')}")

    rows = []
    times = payload.get("t", [])
    volumes = payload.get("v") or [0] * len(times)
    if len(volumes) < len(times):
        volumes = list(volumes) + [0] * (len(times) - len(volumes))

    for ts, o, h, l, c, v in zip(
        times,
        payload.get("o", []),
        payload.get("h", []),
        payload.get("l", []),
        payload.get("c", []),
        volumes,
    ):
        date = datetime.utcfromtimestamp(ts).date()
        if date < datetime(2017, 1, 1).date():
            continue
        rows.append(
            {
                "date": date,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
        )
    return rows


def read_events(path: Path) -> List[Dict]:
    events = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["event_date"] = datetime.fromisoformat(row["event_date"]).date()
            events.append(row)
    return events


def save_price_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"date": r["date"].isoformat(), **{k: r[k] for k in ["open", "high", "low", "close", "volume"]}})


def compute_log_returns(rows: List[Dict]) -> List[float]:
    returns = [0.0]
    for i in range(1, len(rows)):
        prev = rows[i - 1]["close"]
        cur = rows[i]["close"]
        returns.append(math.log(cur / prev))
    return returns


def find_trading_date(rows: List[Dict], target) -> int:
    for idx, r in enumerate(rows):
        if r["date"] >= target:
            return idx
    return -1


def event_windows(rows: List[Dict], events: List[Dict]):
    log_returns = compute_log_returns(rows)
    per_event = []
    for ev in events:
        center = find_trading_date(rows, ev["event_date"])
        if center < WINDOW or center + WINDOW >= len(rows):
            continue
        entry_price = rows[center]["close"]
        window_returns = {}
        for offset in [-30, -7, -1, 0, 1, 7, 30]:
            idx = center + offset
            window_returns[offset] = rows[idx]["close"] / entry_price - 1
        pre_slice = log_returns[center - WINDOW : center]
        post_slice = log_returns[center + 1 : center + WINDOW + 1]
        pre_vol = statistics.pstdev(pre_slice) * math.sqrt(252)
        post_vol = statistics.pstdev(post_slice) * math.sqrt(252)
        per_event.append(
            {
                "event_date": ev["event_date"],
                "event_type": ev["event_type"],
                "short_description": ev["short_description"],
                "trading_date": rows[center]["date"],
                "returns": window_returns,
                "pre_vol": pre_vol,
                "post_vol": post_vol,
            }
        )
    return per_event


def summarize_events(per_event: List[Dict]):
    summary = {}
    for offset in [-30, -7, -1, 0, 1, 7, 30]:
        vals = [ev["returns"][offset] for ev in per_event]
        summary[offset] = {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
        }
    return summary


def equity_curve(rows: List[Dict], trades: List[Dict]):
    daily_returns = [0.0]
    for i in range(1, len(rows)):
        daily_returns.append(rows[i]["close"] / rows[i - 1]["close"] - 1)
    position = [0.0 for _ in rows]
    for tr in trades:
        for idx in range(tr["entry"] + 1, tr["exit"] + 1):
            position[idx] = 1.0
        position[tr["entry"] + 1] -= 0.001
        position[tr["exit"]] -= 0.001
    equity = [1.0]
    for ret, pos in zip(daily_returns[1:], position[1:]):
        equity.append(equity[-1] * (1 + pos * ret))
    peak = [max(equity[: i + 1]) for i in range(len(equity))]
    drawdown = [(eq - pk) / pk for eq, pk in zip(equity, peak)]
    return equity, drawdown


def backtest(rows: List[Dict], events: List[Dict], horizon: int, entry_shift: int = 0, filter_types=None):
    filtered = [ev for ev in events if filter_types is None or ev["event_type"] in filter_types]
    filtered.sort(key=lambda x: x["event_date"])
    trades = []
    last_exit = -1
    priority = {"approval": 3, "launch": 2, "major_signal": 1, "filing": 0}
    for ev in filtered:
        entry_idx = find_trading_date(rows, ev["event_date"]) + entry_shift
        if entry_idx < 0 or entry_idx >= len(rows):
            continue
        exit_idx = entry_idx + horizon
        if exit_idx >= len(rows):
            continue
        if entry_idx <= last_exit:
            if priority.get(ev["event_type"], -1) <= priority.get(trades[-1]["event_type"], -1):
                continue
            trades.pop()
        entry_price = rows[entry_idx]["close"]
        exit_price = rows[exit_idx]["close"]
        net_ret = exit_price / entry_price - 1 - 0.002
        trades.append(
            {
                "entry": entry_idx,
                "exit": exit_idx,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "net_return": net_ret,
                "event_type": ev["event_type"],
                "label": ev["short_description"],
            }
        )
        last_exit = exit_idx
    equity, drawdown = equity_curve(rows, trades)
    trade_returns = [t["net_return"] for t in trades]
    total_days = len(rows)
    cagr = (equity[-1]) ** (252 / total_days) - 1
    ann_vol = statistics.pstdev([equity[i] / equity[i - 1] - 1 for i in range(1, len(equity))]) * math.sqrt(252)
    sharpe = (cagr / ann_vol) if ann_vol else float("nan")
    max_dd = min(drawdown)
    hit_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) if trade_returns else float("nan")
    avg_trade = statistics.mean(trade_returns) if trade_returns else float("nan")
    stats = {
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "avg_trade_return": avg_trade,
        "trade_count": len(trades),
        "final_value": equity[-1],
    }
    return trades, equity, drawdown, stats


def save_svg_line(x_values, y_values, path: Path, title: str, y_label: str):
    width, height = 800, 400
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    y_range = max_y - min_y if max_y != min_y else 1
    def scale_x(x):
        return 50 + (x - min_x) / (max_x - min_x) * (width - 100) if max_x != min_x else width / 2
    def scale_y(y):
        return height - 50 - (y - min_y) / y_range * (height - 100)
    points = " ".join(f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in zip(x_values, y_values))
    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
  <rect width='100%' height='100%' fill='white' stroke='lightgray'/>
  <polyline fill='none' stroke='steelblue' stroke-width='2' points='{points}' />
  <text x='{width/2}' y='20' text-anchor='middle' font-size='16'>{title}</text>
  <text x='20' y='{height/2}' transform='rotate(-90 20,{height/2})' font-size='12'>{y_label}</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


def main():
    prices = fetch_price_data()
    save_price_csv(prices, Path('data/btc_usd_daily.csv'))
    events = read_events(Path('data/etf_events.csv'))
    per_event = event_windows(prices, events)
    summary = summarize_events(per_event)
    summary_path = Path('data/event_summary.csv')
    with summary_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['window', 'mean', 'median'])
        for k, v in summary.items():
            writer.writerow([k, v['mean'], v['median']])
    # Backtests
    trades_7, equity_7, dd_7, stats_7 = backtest(prices, events, horizon=7)
    trades_30, equity_30, dd_30, stats_30 = backtest(prices, events, horizon=30)
    approval_launch = [ev for ev in events if ev['event_type'] in {'approval', 'launch'}]
    trades_s2, equity_s2, dd_s2, stats_s2 = backtest(prices, approval_launch, horizon=7, entry_shift=-7)
    # Save stats
    def save_stats(path, stats):
        with path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(stats.keys())
            writer.writerow(stats.values())
    save_stats(Path('data/stats_s1_7.csv'), stats_7)
    save_stats(Path('data/stats_s1_30.csv'), stats_30)
    save_stats(Path('data/stats_s2.csv'), stats_s2)
    # Save equity svg
    save_svg_line(list(range(len(equity_7))), equity_7, Path('images/s1_7_equity.svg'), 'Strategy S1 - 7 day hold', 'Equity')
    save_svg_line(list(range(len(equity_30))), equity_30, Path('images/s1_30_equity.svg'), 'Strategy S1 - 30 day hold', 'Equity')
    save_svg_line(list(range(len(equity_s2))), equity_s2, Path('images/s2_equity.svg'), 'Strategy S2 - pre-event entry', 'Equity')
    # Price svg
    price_values = [row['close'] for row in prices]
    save_svg_line(list(range(len(price_values))), price_values, Path('images/btc_price_events.svg'), 'BTC price', 'USD')
    print('Summary', summary)
    print('S1-7', stats_7)
    print('S1-30', stats_30)
    print('S2', stats_s2)


if __name__ == '__main__':
    main()
