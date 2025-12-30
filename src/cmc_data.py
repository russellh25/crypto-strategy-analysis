import os
import time
import datetime as dt
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests


DEFAULT_STARTS = {
    "BTC": dt.date(2014, 1, 1),
    "ETH": dt.date(2015, 8, 1),
    "XRP": dt.date(2013, 1, 1),
}

API_URL = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical"


class MissingAPIKeyError(RuntimeError):
    pass


class CMCAPIError(RuntimeError):
    pass


def _require_key() -> str:
    key = os.getenv("CMC_API_KEY") or os.getenv("X-CMC_PRO_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "CMC_API_KEY is not set. Set it in your environment or .env file to download data."
        )
    return key


def _request_with_retry(params: dict, headers: dict, retries: int = 5, backoff: float = 1.0) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(API_URL, params=params, headers=headers, timeout=60)
            if resp.status_code == 429:
                time.sleep(backoff * (2 ** attempt))
                continue
            if resp.status_code >= 500:
                time.sleep(backoff * (2 ** attempt))
                last_error = CMCAPIError(
                    f"CMC server error {resp.status_code}: {resp.text[:200]}"
                )
                continue
            if resp.status_code == 401 or resp.status_code == 403:
                raise CMCAPIError(
                    "CMC key missing/invalid or plan does not permit this endpoint."
                )
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("status", {}).get("error_code") not in (0, None):
                raise CMCAPIError(
                    f"CMC returned error: {payload['status'].get('error_message')}"
                )
            return payload
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(backoff * (2 ** attempt))
    raise CMCAPIError(f"Failed to fetch data after retries: {last_error}")


def _chunk_ranges(start: dt.date, end: dt.date) -> List[Tuple[dt.date, dt.date]]:
    chunks = []
    cur = start
    while cur <= end:
        next_end = min(dt.date(cur.year, 12, 31), end)
        chunks.append((cur, next_end))
        cur = next_end + dt.timedelta(days=1)
    return chunks


def _parse_quotes(quotes: List[dict]) -> pd.DataFrame:
    rows = []
    for q in quotes:
        quote = q.get("quote", {}).get("USD", {})
        time_open = q.get("time_open")
        if not quote or not time_open:
            continue
        date = pd.to_datetime(time_open).date()
        rows.append(
            {
                "date": date,
                "open": float(quote.get("open")),
                "high": float(quote.get("high")),
                "low": float(quote.get("low")),
                "close": float(quote.get("close")),
                "volume": float(quote.get("volume", 0.0)),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


def fetch_ohlcv(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    key = _require_key()
    headers = {
        "X-CMC_PRO_API_KEY": key,
        "Accept": "application/json",
        "User-Agent": "crypto-etf-event-study/1.0",
    }
    all_rows: List[pd.DataFrame] = []
    for rng_start, rng_end in _chunk_ranges(start, end):
        params = {
            "symbol": symbol,
            "time_start": rng_start.isoformat(),
            "time_end": rng_end.isoformat(),
            "interval": "daily",
            "convert": "USD",
        }
        payload = _request_with_retry(params=params, headers=headers)
        quotes = payload.get("data", {}).get("quotes", [])
        all_rows.append(_parse_quotes(quotes))
        time.sleep(1.0)
    if not all_rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return combined


def get_ohlcv_daily(symbol: str, start: str | dt.date | None = None, end: str | dt.date | None = None, force_download: bool = False) -> pd.DataFrame:
    symbol = symbol.upper()
    start_date = (
        pd.to_datetime(start).date()
        if start is not None
        else DEFAULT_STARTS.get(symbol, dt.date(2014, 1, 1))
    )
    end_date = pd.to_datetime(end).date() if end is not None else dt.date.today()
    cache_path = Path("data") / f"prices_{symbol.lower()}_daily.csv"
    if cache_path.exists() and not force_download:
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        cached["date"] = cached["date"].dt.date
        cached = cached.sort_values("date").drop_duplicates("date")
        if cached["date"].min() <= start_date and cached["date"].max() >= end_date:
            return cached[(cached["date"] >= start_date) & (cached["date"] <= end_date)].reset_index(drop=True)
        start_date = min(start_date, cached["date"].min())
        end_date = max(end_date, cached["date"].max())
    df = fetch_ohlcv(symbol, start_date, end_date)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)

