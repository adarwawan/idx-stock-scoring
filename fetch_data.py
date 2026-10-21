"""Fetch IDX stock data and write a dated metrics CSV.

Pipeline
  1. Pasardana (one call): universe, prices, returns, liquidity, sector, PER/PBR/ROE.
  2. Pre-filter the universe by market cap / liquidity so Yahoo is only called for
     names that could pass the screen.
  3. Yahoo Finance (concurrent, cached): .info, annual statements and dividends,
     reduced to a flat set of fundamentals (the old columns plus derived ones).
  4. Write data/idx_metrics_<date>.csv, data/latest.csv and a coverage report.

Usage
  .venv/bin/python fetch_data.py                  # use caches where fresh
  .venv/bin/python fetch_data.py --refresh        # ignore caches
  .venv/bin/python fetch_data.py --min-mcap 1e12 --workers 4
"""
import argparse
import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

PASARDANA_URL = 'https://pasardana.id/api/StockSearchResult/GetAll'
DATA_DIR = Path(__file__).parent / 'data'
CACHE_DIR = DATA_DIR / 'cache'
YF_CACHE_DIR = CACHE_DIR / 'yf'

# Pasardana columns worth keeping (the rest are ids and duplicate name columns).
PASARDANA_KEEP = [
    'Code', 'Name', 'SectorName', 'SubSectorName', 'NewSectorName', 'NewIndustryName',
    'NewSubSectorName', 'NewSubIndustryName', 'Last', 'PrevClosingPrice',
    'AdjustedClosingPrice', 'Volume', 'Frequency', 'Value', 'OneDay', 'OneWeek',
    'OneMonth', 'ThreeMonth', 'SixMonth', 'OneYear', 'ThreeYear', 'FiveYear', 'TenYear',
    'Mtd', 'Ytd', 'Per', 'Pbr', 'Capitalization', 'BetaOneYear', 'StdevOneYear',
    'PerAnnualized', 'PsrAnnualized', 'PcfrAnnualized', 'AdjustedAnnualHighPrice',
    'AdjustedAnnualLowPrice', 'LastDate', 'Roe', 'FreeFloatPct',
]

log = logging.getLogger('fetch_data')

# Yahoo answers bursts with 429s. When one worker sees it, every worker pauses.
_cooldown_until = 0.0
_cooldown_lock = threading.Lock()


def _wait_cooldown():
    while (wait := _cooldown_until - time.time()) > 0:
        time.sleep(min(wait, 5))


def _trigger_cooldown(seconds):
    global _cooldown_until
    with _cooldown_lock:
        _cooldown_until = max(_cooldown_until, time.time() + seconds)


# ---------------------------------------------------------------- Pasardana --

def fetch_pasardana(refresh=False):
    """Universe + market data. Cached per day since it is a daily snapshot."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f'pasardana_{date.today():%Y%m%d}.json'
    if path.exists() and not refresh:
        return pd.DataFrame(json.loads(path.read_text()))
    resp = requests.get(PASARDANA_URL, timeout=30)
    resp.raise_for_status()
    rows = resp.json()
    path.write_text(json.dumps(rows))
    return pd.DataFrame(rows)


def prefilter(df, min_mcap, min_value, min_freq):
    keep = (
        (df['Capitalization'] >= min_mcap)
        & (df['Value'] >= min_value)
        & (df['Frequency'] >= min_freq)
    )
    return df[keep.fillna(False)]


# -------------------------------------------------------------------- Yahoo --

def _row(frame, *names):
    """First matching line item from a yfinance statement as a date-sorted Series."""
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    for name in names:
        if name in frame.index:
            return frame.loc[name].dropna().sort_index()
    return pd.Series(dtype=float)


def _cagr(series, years=3):
    """CAGR over `years` annual periods; None if the endpoints are not positive."""
    if len(series) <= years:
        return None
    first, last = series.iloc[-1 - years], series.iloc[-1]
    if first <= 0 or last <= 0:
        return None
    return float((last / first) ** (1 / years) - 1)


def _div(a, b):
    return float(a / b) if a is not None and b not in (None, 0) and np.isfinite(b) else None


def _pct(x):
    return None if x is None else x * 100


def _last(series):
    return float(series.iloc[-1]) if len(series) else None


def _fetch_one_raw(ticker):
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    if info.get('currentPrice') is None:
        return None
    return {
        'info': info,
        'income': tk.income_stmt,
        'balance': tk.balance_sheet,
        'cashflow': tk.cashflow,
        'dividends': tk.dividends,
    }


def derive_metrics(ticker, raw):
    """Reduce raw yfinance objects to one flat row of fundamentals."""
    info, inc, bs, cf, divs = (raw[k] for k in ('info', 'income', 'balance', 'cashflow', 'dividends'))

    float_shares, shares_out = info.get('floatShares'), info.get('sharesOutstanding')
    price = info.get('currentPrice')

    revenue = _row(inc, 'Total Revenue', 'Operating Revenue')
    net_income = _row(inc, 'Net Income Common Stockholders', 'Net Income')
    ebit = _row(inc, 'EBIT', 'Operating Income')
    interest = _row(inc, 'Interest Expense', 'Interest Expense Non Operating').abs()
    equity = _row(bs, 'Stockholders Equity', 'Common Stock Equity')
    total_debt = _row(bs, 'Total Debt')
    total_liab = _row(bs, 'Total Liabilities Net Minority Interest')
    total_assets = _row(bs, 'Total Assets')
    ocf = _row(cf, 'Operating Cash Flow', 'Cash Flow From Continuing Operating Activities',
               'Cash Flowsfromusedin Operating Activities Direct')
    fcf = _row(cf, 'Free Cash Flow')
    capex = _row(cf, 'Capital Expenditure')  # negative = spend
    if ocf.empty and not fcf.empty and not capex.empty:
        ocf = (fcf - capex).dropna()  # FCF = OCF + capex, so OCF = FCF - capex
    cash_div = _row(cf, 'Cash Dividends Paid', 'Common Stock Dividend Paid').abs()

    # Dividends over the last 12 months and how many of the last 5 years paid one.
    ttm_div, div_years = 0.0, 0  # no dividend history means a non-payer, not missing data
    if divs is not None and len(divs):
        idx = divs.index.tz_localize(None) if divs.index.tz is not None else divs.index
        s = pd.Series(divs.values, index=idx)
        ttm_div = float(s[s.index > pd.Timestamp.today() - pd.DateOffset(years=1)].sum())
        div_years = int(len({d.year for d in s.index if d.year >= date.today().year - 5}))

    ni, rev = _last(net_income), _last(revenue)
    debt_equity = _div(_last(total_debt), _last(equity))

    return {
        'Ticker': ticker,
        'YF Name': info.get('shortName'),
        'YF Sector': info.get('sector'),
        'Current Price': price,
        'Market Cap': info.get('marketCap'),
        'PE Ratio': info.get('trailingPE'),
        'Forward PE': info.get('forwardPE'),
        'Dividend Yield': info.get('dividendYield'),
        'Debt to Equity': info.get('debtToEquity'),
        'Revenue Growth': info.get('revenueGrowth'),
        'EPS Growth': info.get('earningsQuarterlyGrowth'),
        'ROA': info.get('returnOnAssets'),
        'Operating Margin': info.get('operatingMargins'),
        'Net Margin': info.get('profitMargins'),
        'EV/EBITDA': info.get('enterpriseToEbitda'),
        'Beta': info.get('beta'),
        'Payout Ratio': info.get('payoutRatio'),
        'Current Ratio': info.get('currentRatio'),
        'Quick Ratio': info.get('quickRatio'),
        'Free Float Ratio': _div(float_shares, shares_out),
        'Currency': info.get('currency'),
        'Financial Currency': info.get('financialCurrency'),
        # Derived from the statements: fills Yahoo's gaps and gives multi-year views.
        'Revenue CAGR 3Y': _cagr(revenue),
        'Net Income CAGR 3Y': _cagr(net_income),
        'Net Income TTM': ni,
        'Net Income Years Positive': int((net_income.tail(4) > 0).sum()) if len(net_income) else None,
        'FCF': _last(fcf),
        'FCF Margin': _div(_last(fcf), rev),
        'CFO to Net Income': _div(_last(ocf), ni) if ni and ni > 0 else None,
        'Interest Coverage': _div(_last(ebit), _last(interest)),
        'Debt to Equity Calc': None if debt_equity is None else debt_equity * 100,
        # Yahoo's Total Debt leaves out bank deposits, so banks look debt-free on the line above.
        'Liabilities to Equity': _pct(_div(_last(total_liab), _last(equity))),
        'Debt to Assets': _pct(_div(_last(total_debt), _last(total_assets))),
        'Net Debt to EBITDA': _div(
            (_last(total_debt) or 0) - (info.get('totalCash') or 0), info.get('ebitda')),
        'Dividend TTM': ttm_div,
        'Dividend Yield Calc': None if not price or ttm_div is None else ttm_div / price * 100,
        'Dividend Years 5Y': div_years,
        'Payout Ratio Calc': _div(_last(cash_div), ni) if ni and ni > 0 else None,
        'Statements Latest FY': str(max(net_income.index).date()) if len(net_income) else None,
    }


def fetch_ticker(ticker, ttl_days, refresh, retries=5):
    """Return (row | None, status). Reads/writes a per-ticker JSON cache.

    If a refresh fails but an older cached row exists, the stale row is used.
    """
    path = YF_CACHE_DIR / f'{ticker}.json'
    cached = json.loads(path.read_text()) if path.exists() else None
    if cached and not refresh:
        age = (datetime.now() - datetime.fromisoformat(cached['fetched_at'])).days
        if age < ttl_days:
            return cached['row'], 'cache'

    last_err = None
    for attempt in range(retries):
        _wait_cooldown()
        try:
            raw = _fetch_one_raw(ticker)
            row = None if raw is None else derive_metrics(ticker, raw)
            path.write_text(json.dumps({'fetched_at': datetime.now().isoformat(), 'row': row}))
            return row, 'fetched' if row else 'no_data'
        except Exception as exc:  # network, rate limit, parse
            last_err = exc
            if 'Too Many Requests' in str(exc) or 'Rate limit' in str(exc):
                _trigger_cooldown(20 * (attempt + 1))
            else:
                time.sleep((2 ** attempt) + random.random())
    if cached and cached['row']:
        return cached['row'], f'stale: {last_err}'
    return None, f'error: {last_err}'


def fetch_yahoo(tickers, workers, ttl_days, refresh):
    YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows, status = [], {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_ticker, t, ttl_days, refresh): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            ticker = futures[fut]
            row, st = fut.result()
            status[ticker] = st
            if row:
                rows.append(row)
            if i % 50 == 0 or i == len(futures):
                log.info('yahoo %d/%d', i, len(futures))
    return pd.DataFrame(rows), pd.Series(status, name='status')


# ------------------------------------------------------------------ Output --

def coverage_report(df, cols):
    cov = df[cols].notna().mean().rename('coverage').to_frame()
    cov['missing'] = df[cols].isna().sum()
    return cov.sort_values('coverage')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--min-mcap', type=float, default=5e11, help='IDR, default 5e11')
    p.add_argument('--min-value', type=float, default=1e8, help='daily traded value IDR, default 1e8')
    p.add_argument('--min-freq', type=float, default=200, help='daily trade count, default 200')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--ttl-days', type=int, default=7, help='Yahoo cache lifetime')
    p.add_argument('--refresh', action='store_true', help='ignore all caches')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

    start = time.time()
    pas = fetch_pasardana(args.refresh)
    pas['Ticker'] = pas['Code'].astype(str) + '.JK'
    pas = pas[[c for c in PASARDANA_KEEP if c in pas.columns] + ['Ticker']]
    log.info('pasardana: %d stocks', len(pas))

    shortlist = prefilter(pas, args.min_mcap, args.min_value, args.min_freq)
    log.info('prefilter: %d stocks go to Yahoo', len(shortlist))

    yf_df, status = fetch_yahoo(shortlist['Ticker'].tolist(), args.workers, args.ttl_days, args.refresh)
    log.info('yahoo status: %s', status.str.split(':').str[0].value_counts().to_dict())

    # Left-join onto the shortlist so names Yahoo lacks stay visible (with NaNs).
    out = shortlist.merge(yf_df, on='Ticker', how='left')
    # Yahoo's sector is coarser than IDX's; keep the old 'Sector' name for the notebook.
    out = out.rename(columns={'YF Sector': 'Sector'})
    out['Fetched On'] = date.today().isoformat()

    DATA_DIR.mkdir(exist_ok=True)
    stamped = DATA_DIR / f'idx_metrics_{date.today():%Y%m%d}.csv'
    out.to_csv(stamped, index=False)
    out.to_csv(DATA_DIR / 'latest.csv', index=False)
    status[status.str.startswith('error') | (status == 'no_data')].to_csv(DATA_DIR / 'failures.csv')

    skip = ('Ticker', 'YF Name', 'YF Sector', 'Currency', 'Financial Currency', 'Statements Latest FY')
    yahoo_cols = [c for c in yf_df.columns if c not in skip]
    cov = coverage_report(out, yahoo_cols)
    cov.to_csv(DATA_DIR / 'coverage_latest.csv')
    print('\nYahoo field coverage over the shortlist (%d rows):' % len(out))
    print(cov.round(2).to_string())
    print(f'\nWrote {stamped} and data/latest.csv in {time.time() - start:.0f}s')


if __name__ == '__main__':
    main()
