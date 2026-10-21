"""Charts for the IDX scoring notebook (Plotly).

Colors follow the validated reference palette (categorical slots 1-6 pass the
adjacent CVD / lightness checks in light mode). Three of the six sit below 3:1
contrast on the surface, so every chart has a legend or direct labels plus hover
text carrying the values.

Each function takes the notebook's DataFrames and returns a plotly Figure.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e8e7e3'
MUTED = '#b9b8b2'

# Categorical slots 1-6, in fixed order, assigned per factor (not per rank).
FACTOR_COLORS = {
    'Quality Score': '#2a78d6',
    'Growth Score': '#eb6834',
    'Income Score': '#1baf7a',
    'Valuation Score': '#eda100',
    'Risk Score': '#e87ba4',
    'Momentum Score': '#008300',
}
SEQUENTIAL_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
BAR_COLOR = '#2a78d6'

FONT = dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            color=TEXT_SECONDARY, size=12)


def _layout(fig, title, subtitle=None, height=480, margin=None):
    text = f'<b>{title}</b>' + (f'<br><span style="font-size:12px;color:{TEXT_SECONDARY}">{subtitle}</span>'
                               if subtitle else '')
    fig.update_layout(
        title=dict(text=text, x=0, xanchor='left', font=dict(size=16, color=TEXT_PRIMARY)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=FONT, height=height,
        margin=margin or dict(l=80, r=40, t=80, b=60),
        hoverlabel=dict(bgcolor=SURFACE, font=dict(color=TEXT_PRIMARY), bordercolor=GRID),
    )
    fig.update_xaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=GRID, ticks='')
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=GRID, ticks='')
    return fig


def _short(ticker):
    return ticker.replace('.JK', '')


def score_breakdown(ranked, weights, n=15):
    """Stacked bars: how much each factor adds to a stock's Total Score."""
    top = ranked.sort_values('Total Score', ascending=False).head(n).iloc[::-1]
    labels = [_short(t) for t in top['Ticker']]
    fig = go.Figure()
    for factor, color in FACTOR_COLORS.items():
        contrib = top[factor] * weights[factor]
        fig.add_bar(
            y=labels, x=contrib, orientation='h', name=factor.replace(' Score', ''),
            marker=dict(color=color, line=dict(color=SURFACE, width=2)),  # 2px surface gap
            customdata=np.column_stack([top[factor], [f'{weights[factor]:.0%}'] * len(top)]),
            hovertemplate=('<b>%{y}</b><br>' + factor.replace(' Score', '') +
                           ': %{customdata[0]:.0f}/100 x %{customdata[1]} weight'
                           '<br>adds %{x:.1f} points<extra></extra>'),
        )
    fig.add_trace(go.Scatter(  # invisible trace carrying the total at the bar tip
        y=labels, x=top['Total Score'], mode='text', text=[f'{v:.1f}' for v in top['Total Score']],
        textposition='middle right', textfont=dict(color=TEXT_PRIMARY, size=12),
        showlegend=False, hoverinfo='skip', cliponaxis=False))
    fig.update_layout(barmode='stack', bargap=0.45,
                      legend=dict(orientation='h', x=0, y=1.0, yanchor='bottom', title_text='', traceorder='normal'))
    fig.update_xaxes(title_text='Total Score (weighted points)', range=[0, top['Total Score'].max() * 1.12])
    fig.update_yaxes(showgrid=False)
    return _layout(fig, 'What drives each score',
                   f'Top {len(top)} stocks. Segment = factor score x factor weight.',
                   height=max(360, 34 * len(top) + 190), margin=dict(l=70, r=50, t=140, b=60))


def factor_heatmap(ranked, n=15):
    """Heatmap of the six factor scores for the top stocks."""
    top = ranked.sort_values('Total Score', ascending=False).head(n)
    factors = list(FACTOR_COLORS)
    z = top[factors].to_numpy()
    fig = go.Figure(go.Heatmap(
        z=z, x=[f.replace(' Score', '') for f in factors], y=[_short(t) for t in top['Ticker']],
        colorscale=[[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)],
        zmin=0, zmax=100, xgap=2, ygap=2,
        colorbar=dict(title='Score', thickness=12, outlinewidth=0, tickfont=dict(color=TEXT_SECONDARY)),
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.0f}/100<extra></extra>',
    ))
    fig.update_yaxes(autorange='reversed', showgrid=False)
    fig.update_xaxes(showgrid=False, side='top')
    return _layout(fig, 'Factor profile', 'Darker = stronger. Stocks sorted by Total Score.',
                   height=max(360, 30 * len(top) + 150), margin=dict(l=70, r=40, t=110, b=30))


def yield_vs_roe(ranked, portfolio=None, label_n=8):
    """Dividend yield against ROE; bubble area = market cap.

    With a portfolio, only its stocks are drawn (all labeled). Without one, the top
    `label_n` stocks by Total Score are drawn.
    """
    pool = portfolio if portfolio is not None else ranked.nlargest(label_n, 'Total Score')
    df = pool.dropna(subset=['Dividend Yield', 'Roe', 'Market Cap']).copy()
    size = np.sqrt(df['Market Cap'] / df['Market Cap'].max()) * 34 + 12  # diameter px, >= 8
    name = 'Portfolio' if portfolio is not None else f'Top {len(df)} by score'

    # Alternate label sides so neighbors at similar yield/ROE do not collide.
    order = df.sort_values('Dividend Yield').index
    side = {idx: ['top center', 'bottom center'][i % 2] for i, idx in enumerate(order)}

    fig = go.Figure(go.Scatter(
        x=df['Dividend Yield'], y=df['Roe'] * 100, mode='markers+text', name=name,
        text=[_short(t) for t in df['Ticker']], textposition=[side[i] for i in df.index],
        textfont=dict(color=TEXT_PRIMARY, size=12),
        marker=dict(size=size, color=BAR_COLOR, opacity=0.85, sizemode='diameter',
                    line=dict(color=SURFACE, width=2)),  # 2px surface ring
        customdata=np.column_stack([df['Ticker'].map(_short), df['Total Score'], df['Market Cap'] / 1e12]),
        hovertemplate=('<b>%{customdata[0]}</b><br>Dividend yield: %{x:.1f}%<br>ROE: %{y:.1f}%'
                       '<br>Total Score: %{customdata[1]:.1f}<br>Market cap: %{customdata[2]:.1f}T IDR'
                       '<extra></extra>'),
    ))
    pad_x = (df['Dividend Yield'].max() - df['Dividend Yield'].min()) * 0.12 + 0.3
    pad_y = (df['Roe'].max() - df['Roe'].min()) * 100 * 0.18 + 2
    fig.update_xaxes(title_text='Dividend yield (%)',
                     range=[df['Dividend Yield'].min() - pad_x, df['Dividend Yield'].max() + pad_x])
    fig.update_yaxes(title_text='Return on equity (%)',
                     range=[df['Roe'].min() * 100 - pad_y, df['Roe'].max() * 100 + pad_y])
    fig.update_layout(showlegend=False)
    return _layout(fig, 'Income versus profitability', f'{name}. Bubble area = market cap.', height=520,
                   margin=dict(l=80, r=40, t=80, b=60))


def sector_mix(ranked, by='Sector'):
    """How many screened stocks sit in each sector."""
    counts = ranked[by].fillna('Unknown').value_counts().sort_values()
    fig = go.Figure(go.Bar(
        y=counts.index, x=counts.values, orientation='h', marker=dict(color=BAR_COLOR, cornerradius=4),
        text=counts.values, textposition='outside', textfont=dict(color=TEXT_PRIMARY),
        cliponaxis=False, hovertemplate='<b>%{y}</b><br>%{x} stocks<extra></extra>'))
    fig.update_layout(bargap=0.5)
    fig.update_xaxes(title_text='Stocks', dtick=1, range=[0, counts.max() * 1.15])
    fig.update_yaxes(showgrid=False)
    return _layout(fig, 'Sector concentration', f'{int(counts.sum())} screened stocks by {by.lower()}.',
                   height=max(300, 40 * len(counts) + 140), margin=dict(l=150, r=50, t=80, b=60))


def allocation(portfolio):
    """Portfolio weights, with IDR amount and lots in the tooltip."""
    p = portfolio.sort_values('Portfolio Percentage')
    fig = go.Figure(go.Bar(
        y=[_short(t) for t in p['Ticker']], x=p['Portfolio Percentage'], orientation='h',
        marker=dict(color=BAR_COLOR, cornerradius=4),
        text=[f'{v:.1f}%' for v in p['Portfolio Percentage']], textposition='outside',
        textfont=dict(color=TEXT_PRIMARY), cliponaxis=False,
        customdata=np.column_stack([p['Sector'], p['Value'] / 1e6, p['Lots']]),
        hovertemplate=('<b>%{y}</b> (%{customdata[0]})<br>Weight: %{x:.1f}%'
                       '<br>Invested: %{customdata[1]:.1f}M IDR<br>Lots: %{customdata[2]:.0f}<extra></extra>')))
    fig.update_layout(bargap=0.45)
    fig.update_xaxes(title_text='Portfolio weight (%)', range=[0, p['Portfolio Percentage'].max() * 1.18])
    fig.update_yaxes(showgrid=False)
    return _layout(fig, 'Portfolio allocation',
                   f'{len(p)} stocks, {p["Value"].sum() / 1e6:,.1f}M IDR invested.',
                   height=max(320, 40 * len(p) + 140), margin=dict(l=70, r=50, t=80, b=60))


def save_report(figures, path='reports/idx_report.html', title='IDX stock scoring'):
    """Write the figures into one standalone HTML page (Plotly JS loaded from the CDN)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = ''.join(
        f'<section>{fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)}</section>'
        for i, fig in enumerate(figures))
    out.write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>body{{margin:0;background:{SURFACE};color:{TEXT_PRIMARY};font-family:{FONT["family"]}}}
main{{max-width:960px;margin:0 auto;padding:24px 16px}}h1{{font-size:22px;margin:0 0 16px}}
section{{margin-bottom:32px}}</style></head><body><main><h1>{title}</h1>{body}</main></body></html>''')
    return out
