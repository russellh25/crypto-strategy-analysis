# crypto-strategy-analysis

Event-study and simple backtest around major crypto ETF milestones using CoinMarketCap daily OHLCV data. Caches are stored locally so reruns avoid hitting the API once populated.

## Deliverables
- Notebook: `notebooks/03_etf_event_study.ipynb`
- Event table: `data/etf_events.csv` (30 ETF/ETP milestones for BTC, ETH, XRP, and mixed BTC/ETH futures)
- Analysis modules: `src/cmc_data.py` (CMC downloads + caching) and `src/event_study.py` (windows, CAR, backtest)
- CLI runner: `run_analysis.py` (end-to-end pipeline)
- Reports: `reports/` (CAR per event, summary CAR, trades, trade stats)

## Data sources
- Prices: CoinMarketCap `/v2/cryptocurrency/ohlcv/historical` (daily, USD). Cached to `data/prices_<symbol>_daily.csv`.
- Events: Hard-coded CSV with 30 dates (spot BTC ETF launches on 2024-01-11, spot ETH ETF launches on 2024-07-23, futures ETFs, and XRP/ETP milestones).

## Setup
1. Python 3.11+ with pandas installed (standard scientific stack).
2. Set your CoinMarketCap API key (required for first download):
   ```bash
   export CMC_API_KEY="YOUR_KEY"
   # or create a local .env (not tracked)
   echo "CMC_API_KEY=YOUR_KEY" > .env
   ```
   The pipeline will fall back to cached CSVs in `data/` if the key is missing but data already exists.

## Running the pipeline
```bash
python run_analysis.py
```
This will:
- Download BTC/ETH/XRP (plus BTC as proxy for MIX) via CMC with polite rate limiting and retries.
- Compute log-return windows and CAR for offsets -10..+10 versus BTC as market baseline for non-BTC assets.
- Run a simple pre-event entry/backtest (enter t=-1, exit t=+1, configurable) with transaction costs (BTC/ETH 12 bps round-trip, XRP 30 bps; optional 1.5x widening on event day).
- Save reports to `reports/`.

## Notebook workflow
Open `notebooks/03_etf_event_study.ipynb` to inspect CAR tables and trade stats interactively. The notebook mirrors the CLI flow and uses cached prices if available.

## GitHub/CI hygiene
- API keys are never committed. Use `.env` locally and GitHub Actions secrets (`CMC_API_KEY`) in CI.
- `.env` is already ignored via `.gitignore`; `.env.example` documents the required variable.

## Caveats
- CMC free tier may rate-limit; the code backs off with retries and sleeps between yearly chunks. Provide caches for deterministic offline runs.
- Event dates are fixed to public launch dates; adjust `data/etf_events.csv` if you want additional milestones or jurisdictions.
- No external dependencies beyond pandas/requests.
