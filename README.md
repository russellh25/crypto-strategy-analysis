# crypto-strategy-analysis

This repository builds a reproducible crypto ETF milestone event-study and backtest focused on Bitcoin spot ETF events. The workflow uses publicly available TradingView market data and a hand-curated ETF milestone table with cited sources.

## Deliverables
- **Notebook:** `notebooks/crypto_etf_event_study_and_backtest.ipynb`
- **Event table:** `data/etf_events.csv`
- **Utility code:** `etf_analysis.py` (pandas-based helpers) and `run_analysis.py` (standard-library fallback for data download and metrics)
- **Figures:** generated into `images/` after running the notebook or `run_analysis.py`

## Data
- **Prices:** daily OHLCV for BTC fetched from the TradingView history endpoint (`https://data.tradingview.com/history`). The download is triggered automatically by the notebook or by running `run_analysis.py` and cached to `data/btc_usd_daily.csv`.
- **ETF events:** manually curated, cited milestones stored in `data/etf_events.csv`.

| event_date | asset | event_type | short_description | source |
| --- | --- | --- | --- | --- |
| 2023-06-15 | BTC | filing | BlackRock files for iShares Bitcoin Trust spot BTC ETF | [Reuters](https://www.reuters.com/markets/us/blackrock-files-bitcoin-etf-coinbase-provide-custody-2023-06-15/) |
| 2023-08-29 | BTC | major_signal | DC Circuit court rules SEC must review Grayscale spot bitcoin ETF application | [Reuters](https://www.reuters.com/markets/us/court-sides-with-grayscale-bitcoin-etf-case-versus-sec-2023-08-29/) |
| 2024-01-10 | BTC | approval | SEC approves multiple spot bitcoin ETFs in the United States | [SEC Press Release](https://www.sec.gov/news/press-release/2024-4) |
| 2024-01-11 | BTC | launch | First U.S. spot bitcoin ETFs begin trading on exchanges | [Reuters](https://www.reuters.com/markets/us/spot-bitcoin-etf-fever-grows-debut-looms-2024-01-10/) |

## Methodology
1. **Event study**
   - Align each milestone date to the next available trading day in the BTC series.
   - Compute returns over windows `[-30, -7, -1, 0, +1, +7, +30]` relative to the event close, plus cumulative returns over `[-30, +30]` and realized volatility (annualized log-return standard deviation) for `[-30, -1]` vs. `[+1, +30]`.
   - Plot price with event markers, cumulative-return paths, and drawdowns per event.
   - Aggregate mean/median window returns and run simple t-test and Mann–Whitney U approximations to compare pre/post windows.

2. **Backtests**
   - **S1 (event momentum):** enter at event close, hold for `H ∈ {7, 30}` days, exit at close. Overlapping events keep the higher-priority type (approval > launch > major signal > filing). Transaction cost: 10 bps per side.
   - **S2 (rumor/sell-the-news):** enter 7 trading days before approval/launch events and exit on the event close. Same cost assumptions.
   - Performance metrics: CAGR, annualized volatility, Sharpe (mean/stdev * √252), max drawdown, hit rate, average trade return, trade count, and equity/drawdown curves.

## How to reproduce
1. Ensure Python 3.11+ with `numpy`, `pandas`, and `matplotlib` available.
2. With network access to TradingView:
   - Run `python run_analysis.py` for a standard-library-only end-to-end pull of data, statistics, and SVG equity/price plots.
   - Or open and execute `notebooks/crypto_etf_event_study_and_backtest.ipynb` for the pandas/matplotlib workflow and richer visualizations.
3. Outputs are cached to `data/` and `images/` for re-use.

## Caveats
- Internet access is required to pull price history from TradingView; offline runs will fail to refresh data.
- This codebase does not install dependencies automatically. If `numpy`/`pandas`/`matplotlib` are missing, install them or rely on the `run_analysis.py` fallback (which still needs network access).
- Statistical tests use normal approximations (no SciPy dependency). Results are indicative rather than definitive for small samples.
- Event timestamps use UTC news/report dates; intraday market reactions and jurisdictional time differences are not captured.
