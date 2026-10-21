# idx-stock-scoring

Ranks Indonesia Stock Exchange (IDX) stocks with a transparent multi-factor score (0-100) for research and screening. Not investment advice.

## Quick start

```bash
make setup      # create .venv and install dependencies (first time only)
make notebook   # fetch fresh data, then open the scoring notebook
```

Run `make` to list all commands:

| Command | What it does |
|---|---|
| `make setup` | Create `.venv` and install `requirements.txt` |
| `make fetch` | Fetch data, reusing caches that are still fresh |
| `make refresh` | Fetch data, ignoring all caches (about 90s) |
| `make coverage` | Show per-field data coverage from the last fetch |
| `make notebook` | Run `make fetch`, then open `idx_stock_scoring.ipynb` in JupyterLab |
| `make clean-cache` | Delete cached API responses |
| `make clean` | Delete the cache and all generated data files |

Pass options to the fetcher with `ARGS`, for example `make fetch ARGS="--min-mcap 1e12 --workers 2"`.

## How it works

### 1. Data (`fetch_data.py`)

1. **Pasardana** (one call, free): the full IDX universe with prices, returns, liquidity, sector, PER/PBR/ROE.
2. **Pre-filter** by market cap, daily value and trade count, so Yahoo is only called for stocks that could pass the screen (about 340 of 980).
3. **Yahoo Finance** (`yfinance`, free): `.info`, annual statements and dividend history for the shortlist. Fetched concurrently, with a shared pause on rate limits and a per-ticker cache (7 days). If a refresh fails, the stale cached row is used.
4. **Derived fields** computed from the statements fill Yahoo's gaps: 3-year revenue and net income CAGR, FCF and FCF margin, cash flow to net income, interest coverage, debt to equity, net debt to EBITDA, trailing dividend, dividend yield, years of dividends and payout ratio.

Outputs in `data/`:

| File | Content |
|---|---|
| `latest.csv` | Most recent metrics; the notebook reads this |
| `idx_metrics_YYYYMMDD.csv` | Dated snapshot, kept as history |
| `coverage_latest.csv` | Share of the shortlist with a value, per field |
| `failures.csv` | Tickers Yahoo could not return |
| `cache/` | Raw per-ticker and per-day caches (git-ignored) |

Fetcher options (`.venv/bin/python fetch_data.py --help`):

| Option | Default | Meaning |
|---|---|---|
| `--min-mcap` | `5e11` | Minimum market cap (IDR) |
| `--min-value` | `1e8` | Minimum daily traded value (IDR) |
| `--min-freq` | `200` | Minimum daily trade count |
| `--workers` | `3` | Concurrent Yahoo requests |
| `--ttl-days` | `7` | Yahoo cache lifetime |
| `--refresh` | off | Ignore all caches |

### 2. Scoring (`scoring.py`, driven from `idx_stock_scoring.ipynb`)

1. Load `data/latest.csv`.
2. **Score the whole fetched universe** (`scoring.score_universe`), so a score means "strong against the market" rather than "best of the few that survived the filters".
3. **Apply the eligibility screen afterwards** (`SCREEN` in the notebook): market cap, ROE, debt to equity (applies to every stock), liquidity, dividend yield and data coverage. A stock with an unknown value for a screened field does not pass.
4. Combine six factors, each an average of percentile-ranked inputs:

   | Factor | Weight | Inputs |
   |---|---|---|
   | Quality | 25% | ROE, ROA, operating and net margin, cash conversion, earnings consistency |
   | Growth | 25% | 3-year revenue and net income CAGR, latest revenue growth |
   | Income | 20% | Dividend yield, payout sustainability, 5-year dividend record |
   | Valuation | 15% | PE, PB, EV/EBITDA, FCF yield (ranked within sector) |
   | Risk | 10% | Beta, volatility, debt to equity, net debt to EBITDA, interest coverage |
   | Momentum | 5% | 12-1 month and 6-month return |

5. Rank stocks overall and by sector. `Rank` is among the screened stocks; `Universe Rank` is against everything fetched.
6. Build a portfolio from the top `top_n` stocks (or your own `manual_tickers`), sized by score, market cap and liquidity within min/max position limits, in whole 100-share lots.

Scoring rules worth knowing:

- **Missing data is not neutral.** A missing input counts as 30, so a factor with half its inputs missing is pulled down. Inputs that do not apply to banks (cash conversion, EV/EBITDA, net debt, interest coverage) are skipped instead of penalised.
- **No leverage exemption.** The debt-to-equity limit applies to every stock. For financials it is total liabilities / equity, because deposits are their debt (Yahoo's Total Debt leaves them out).
- **Payout is judged on sustainability.** A 20-70% payout scores 100, fading to 0 at 0% and at 100% or more. No dividend scores 0 on income.
- **Valuation and leverage are ranked within the IDX sector** (sectors under 8 stocks fall back to the whole market). Negative multiples earn no credit.
- **`Flags`** marks red flags: loss-making, payout above 100%, negative free cash flow, free float below 10%, thin data.
- The factor weights are judgement calls. They have not been tested against historical returns.

### 3. Charts (`viz.py`)

Section 7 of the notebook draws five interactive Plotly charts and saves them as one standalone page, `reports/idx_report.html` (open it in any browser; Plotly loads from a CDN):

| Chart | Question it answers |
|---|---|
| What drives each score | How many points each factor adds to a stock's Total Score |
| Factor profile | Where each top stock is strong or weak across the six factors |
| Income versus profitability | Dividend yield against ROE, bubble size = market cap, portfolio names labeled |
| Sector concentration | How many screened stocks sit in each sector |
| Portfolio allocation | Weight, amount and lots per stock |

Colors come from a validated categorical palette (one fixed color per factor). The charts use a light theme only.

## Notes

- Yahoo rate-limits bursts. If a run hits errors, rerun `make fetch`; tickers already cached are not refetched. Lower `--workers` if it persists.
- `Frequency` and `Value` are single-day snapshots, so names near a liquidity threshold can flip between runs.
- Data is only as good as its sources. Check `make coverage` before trusting a score that depends on a sparsely filled field.
