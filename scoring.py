"""Multi-factor scoring for IDX stocks.

Design rules (each fixes a problem in the first version of the notebook):

* Scores are computed on the whole fetched universe and the screen is applied
  afterwards. A score of 80 therefore means "strong against the market", not
  "best of the 30 that survived the filters".
* A missing input is not a free 50. Factors average the inputs they have and
  are pulled toward UNKNOWN_SCORE in proportion to what is missing.
* Valuation and leverage are ranked within the IDX sector (banks carry far more
  debt than miners), falling back to the whole market for small sectors.
* Debt to equity applies to every stock, banks included. For financials it is
  measured as total liabilities / equity, because deposits are their debt and
  Yahoo's Total Debt leaves them out (BBCA would look 1% geared).
* Growth uses 3-year CAGRs instead of one noisy quarter.
* Risk is volatility and balance-sheet strength. Liquidity is a screen input.
* Dividend payout is judged on sustainability, so a 150% payout is a red flag.
"""
import numpy as np
import pandas as pd

WEIGHTS = {
    'Quality Score': 0.25,
    'Growth Score': 0.25,
    'Income Score': 0.20,
    'Valuation Score': 0.15,
    'Risk Score': 0.10,
    'Momentum Score': 0.05,
}

UNKNOWN_SCORE = 30        # what a missing input counts as (below the neutral 50)
MIN_GROUP = 8             # smaller sectors fall back to market-wide ranks
SECTOR_COL = 'NewSectorName'
FINANCIAL_SECTOR = 'Keuangan'

# Eligibility thresholds you can edit in the notebook; applied AFTER scoring.
DEFAULT_SCREEN = {
    'min_market_cap': 1e12,
    'min_roe': 0.10,
    'max_debt_to_equity': 100,   # applies to every stock, financials included
    'min_frequency': 1000,
    'min_dividend_yield': 5,     # percent
    'min_coverage': 0.6,         # share of scoring inputs that must be present
    'min_pe': None,              # e.g. 5; None = off. Loss-makers (PER <= 0) fail when either bound is set
    'max_pe': None,              # e.g. 20
}

NUMERIC = [
    'Per', 'Pbr', 'Roe', 'ROA', 'Operating Margin', 'Net Margin', 'EV/EBITDA', 'Beta',
    'BetaOneYear', 'StdevOneYear', 'OneMonth', 'ThreeMonth', 'SixMonth', 'OneYear',
    'Dividend Yield', 'Dividend Yield Calc', 'Payout Ratio', 'Payout Ratio Calc',
    'Dividend Years 5Y', 'Revenue Growth', 'EPS Growth', 'Revenue CAGR 3Y',
    'Net Income CAGR 3Y', 'Net Income TTM', 'Net Income Years Positive', 'FCF',
    'FCF Margin', 'CFO to Net Income', 'Interest Coverage', 'Debt to Equity',
    'Debt to Equity Calc', 'Liabilities to Equity', 'Net Debt to EBITDA', 'Market Cap', 'Frequency', 'Value',
    'Free Float Ratio', 'FreeFloatPct',
]


def _rank(s, higher_is_better=True, groups=None):
    """Percentile rank 0-100 (best = 100); NaN stays NaN. Ranks within `groups` when big enough."""
    s = pd.to_numeric(s, errors='coerce')
    asc = higher_is_better
    pct = s.rank(pct=True, ascending=asc)
    if groups is not None:
        sizes = s.groupby(groups).transform('count')
        within = s.groupby(groups).rank(pct=True, ascending=asc)
        pct = within.where(sizes >= MIN_GROUP, pct)
    return pct * 100


def _payout_sustainability(payout, yield_):
    """100 for a 20-70% payout, fading to 0 at 0% and at 100%+. Non-payers score 0."""
    p = pd.to_numeric(payout, errors='coerce')
    score = np.select(
        [p <= 0, p < 0.2, p <= 0.7, p <= 1.0],
        [0, p / 0.2 * 100, 100, (1 - (p - 0.7) / 0.3) * 100],
        default=0,
    )
    out = pd.Series(score, index=p.index, dtype=float).where(p.notna())
    return out.where(~(yield_ <= 0), 0)  # no dividend, no income


def _factor(components):
    """Combine (score, applicable) pairs into one 0-100 factor score and a coverage ratio.

    `applicable` is a boolean Series: False means the input does not apply to that
    stock (e.g. leverage for a bank), so it is dropped from the average and from coverage.
    """
    scores = pd.DataFrame({k: v[0] for k, v in components.items()})
    applicable = pd.DataFrame({k: v[1] for k, v in components.items()})
    present = scores.notna() & applicable
    n_applicable = applicable.sum(axis=1)
    coverage = (present.sum(axis=1) / n_applicable.replace(0, np.nan))
    mean_present = scores.where(present).mean(axis=1)
    blended = coverage * mean_present + (1 - coverage) * UNKNOWN_SCORE
    return blended.fillna(50), coverage.fillna(1.0)


def score_universe(raw):
    """Add factor scores, Total Score, coverage and flags to every stock in `raw`."""
    df = raw.copy()
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    sector = df[SECTOR_COL]
    is_fin = sector.eq(FINANCIAL_SECTOR)
    yes = pd.Series(True, index=df.index)
    non_fin = ~is_fin

    # Derived inputs
    df['Leverage'] = df['Debt to Equity Calc'].where(non_fin, df['Liabilities to Equity'])
    df['Dividend Yield Best'] = df['Dividend Yield Calc'].fillna(df['Dividend Yield'])
    payout = df['Payout Ratio Calc'].fillna(df['Payout Ratio'])
    df['Momentum 12-1'] = (1 + df['OneYear']) / (1 + df['OneMonth']) - 1
    df['FCF Yield'] = df['FCF'] / df['Market Cap']
    pos = lambda s: s.where(s > 0)  # a negative multiple is a loss, not a cheap stock

    quality = {
        'ROE': (_rank(df['Roe']), yes),
        'ROA': (_rank(df['ROA']), yes),
        'Operating Margin': (_rank(df['Operating Margin']), yes),
        'Net Margin': (_rank(df['Net Margin']), yes),
        'Cash conversion': (_rank(df['CFO to Net Income'].clip(upper=3)), non_fin),
        'Earnings consistency': (_rank(df['Net Income Years Positive']), yes),
    }
    growth = {
        'Revenue CAGR 3Y': (_rank(df['Revenue CAGR 3Y']), yes),
        'Net income CAGR 3Y': (_rank(df['Net Income CAGR 3Y']), yes),
        'Revenue growth': (_rank(df['Revenue Growth']), yes),
    }
    income = {
        'Dividend yield': (_rank(df['Dividend Yield Best']), yes),
        'Payout sustainability': (_payout_sustainability(payout, df['Dividend Yield Best']), yes),
        'Dividend record': (_rank(df['Dividend Years 5Y']), yes),
    }
    valuation = {
        'PE': (_rank(pos(df['Per']), False, sector), yes),
        'PB': (_rank(pos(df['Pbr']), False, sector), yes),
        'EV/EBITDA': (_rank(pos(df['EV/EBITDA']), False, sector), non_fin),
        'FCF yield': (_rank(df['FCF Yield'], True, sector), non_fin),
    }
    risk = {
        'Beta': (_rank(df['BetaOneYear'], False), yes),
        'Volatility': (_rank(df['StdevOneYear'], False), yes),
        'Debt to equity': (_rank(df['Leverage'], False, sector), yes),
        # A ratio is only meaningful when earnings are positive.
        'Net debt to EBITDA': (_rank(df['Net Debt to EBITDA'].where(df['Net Income TTM'] > 0), False, sector), non_fin),
        'Interest coverage': (_rank(df['Interest Coverage'].clip(upper=50), True, sector), non_fin),
    }
    momentum = {
        '12-1 month return': (_rank(df['Momentum 12-1']), yes),
        'Six month return': (_rank(df['SixMonth']), yes),
    }

    cov = {}
    for name, comps in [('Quality', quality), ('Growth', growth), ('Income', income),
                        ('Valuation', valuation), ('Risk', risk), ('Momentum', momentum)]:
        df[f'{name} Score'], cov[name] = _factor(comps)

    df['Total Score'] = sum(df[k] * w for k, w in WEIGHTS.items())
    df['Data Coverage'] = sum(cov[k.replace(' Score', '')] * w for k, w in WEIGHTS.items())
    df['Universe Rank'] = df['Total Score'].rank(ascending=False, method='min').astype(int)
    df['Flags'] = _flags(df, payout)
    return df


def _flags(df, payout):
    rules = {
        'loss-making': (df['Net Income TTM'] <= 0) | (df['Per'] < 0),
        'payout>100%': payout > 1,
        'negative FCF': df['FCF'] < 0,
        'free float<10%': df['FreeFloatPct'] < 10,
        'thin data': df['Data Coverage'] < 0.6,
    }
    out = pd.Series('', index=df.index)
    for label, mask in rules.items():
        out = out.where(~mask.fillna(False), out + ', ' + label)
    return out.str.lstrip(', ')


def apply_screen(scored, screen=None):
    """Return the scored stocks that pass the eligibility screen (unknown values fail)."""
    s = {**DEFAULT_SCREEN, **(screen or {})}
    keep = (
        scored['Current Price'].notna()
        & (scored['Market Cap'] >= s['min_market_cap'])
        & (scored['Roe'] >= s['min_roe'])
        & (scored['Leverage'] < s['max_debt_to_equity'])
        & (scored['Frequency'] > s['min_frequency'])
        & (scored['Dividend Yield Best'] > s['min_dividend_yield'])
        & (scored['Data Coverage'] >= s['min_coverage'])
    )
    if s['min_pe'] is not None or s['max_pe'] is not None:
        pe = scored['Per'].fillna(pd.to_numeric(scored['PE Ratio'], errors='coerce'))  # Yahoo fills Pasardana gaps
        keep &= pe > 0
        if s['min_pe'] is not None:
            keep &= pe >= s['min_pe']
        if s['max_pe'] is not None:
            keep &= pe <= s['max_pe']
    return scored[keep.fillna(False)].copy()


RANK_COLUMNS = [
    'Rank', 'Universe Rank', 'Ticker', 'Sector', 'NewSubSectorName', 'NewSubIndustryName',
    'Total Score', 'Quality Score', 'Valuation Score', 'Growth Score', 'Momentum Score',
    'Risk Score', 'Income Score', 'Data Coverage', 'Flags',
    'Current Price', 'Market Cap', 'Per', 'Pbr', 'Dividend Yield Best', 'Roe',
    'Leverage', 'Revenue CAGR 3Y', 'Net Income CAGR 3Y', 'FCF Margin',
    'Net Debt to EBITDA', 'OneMonth', 'ThreeMonth', 'SixMonth', 'OneYear',
    'BetaOneYear', 'StdevOneYear', 'Frequency', 'Payout Ratio', 'Free Float Ratio',
]


def rank(screened):
    """Order the screened stocks by Total Score; `Rank` is among the screened ones."""
    out = screened.sort_values(['Total Score', 'Market Cap'], ascending=[False, False]).reset_index(drop=True)
    out['Rank'] = np.arange(1, len(out) + 1)
    out['Dividend Yield'] = out['Dividend Yield Best']  # the name the portfolio and charts use
    cols = [c for c in RANK_COLUMNS if c in out.columns] + ['Dividend Yield']
    return out[cols]
