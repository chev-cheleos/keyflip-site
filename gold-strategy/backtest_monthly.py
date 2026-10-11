"""Monthly trend-filter test on gold vs buy-and-hold.

Data: World Bank monthly average gold prices (USD/oz), via
https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv

Usage:
    python3 backtest_monthly.py monthly.csv

Method (deliberately conservative):
- A signal uses prices through month t-1 and is held over the return from the
  month-t average to the month-(t+1) average. The extra month of lag removes
  the spurious lag-1 autocorrelation that monthly averaging creates (the
  Working effect), which would otherwise flatter trend-following results.
- Cash earns 0% (T-bill income is ignored, which understates the filters).
- Each position change costs 0.2% (as in Bartsch, Baur, Dichtl, Drobetz).
"""
import sys

import numpy as np
import pandas as pd

COST = 0.002

df = pd.read_csv(sys.argv[1])
df["Date"] = pd.to_datetime(df["Date"])
s = df.set_index("Date")["Price"].astype(float)
s = s[s.index >= "1970-01-01"]
ret_next = s.shift(-1) / s - 1


def above_sma(n):
    return (s > s.rolling(n).mean()).astype(float).shift(1)


def momentum(n):
    return (s / s.shift(n) - 1 > 0).astype(float).shift(1)


def run(pos, start, end):
    m = (s.index >= start) & (s.index <= end)
    p, r = pos[m], ret_next[m]
    ok = p.notna() & r.notna()
    p, r = p[ok], r[ok]
    strat = p * r - p.diff().abs().fillna(p.abs()) * COST
    eq = (1 + strat).cumprod()
    cagr = eq.iloc[-1] ** (12 / len(strat)) - 1
    vol = strat.std() * np.sqrt(12)
    sharpe = strat.mean() * 12 / vol
    max_dd = (eq / eq.cummax() - 1).min()
    return cagr, vol, sharpe, max_dd, (p != 0).mean()


# Explicit comparisons: NaN warm-up months and exact ties stay flat (0).
sma10, mom12 = s.rolling(10).mean(), s / s.shift(12) - 1
bull = (s > sma10) & (mom12 > 0)
bear = (s < sma10) & (mom12 < 0)
strategies = {
    "Buy & hold": pd.Series(1.0, index=s.index),
    "Long if > SMA10": above_sma(10),
    "Long if 12m mom > 0": momentum(12),
    "Long if both": bull.astype(float).shift(1),
    "Long/short if both": pd.Series(
        np.where(bull, 1.0, np.where(bear, -1.0, 0.0)), index=s.index).shift(1),
}

periods = [
    ("1972-01-01", "2026-08-01", "1972-2026"),
    ("1980-02-01", "1999-12-01", "1980-1999 (bear/sideways)"),
    ("2000-01-01", "2026-08-01", "2000-2026"),
    ("2011-09-01", "2015-12-01", "2011-2015 (bear)"),
]

for start, end, label in periods:
    print(f"\n=== {label} ===")
    print(f"{'strategy':22s} {'CAGR':>7s} {'Vol':>6s} {'Sharpe':>6s} {'MaxDD':>7s} {'In mkt':>6s}")
    for name, pos in strategies.items():
        cagr, vol, sharpe, max_dd, invested = run(pos, start, end)
        print(f"{name:22s} {cagr:7.1%} {vol:6.1%} {sharpe:6.2f} {max_dd:7.1%} {invested:6.0%}")

print("\n=== Robustness: other lookbacks ===")
for start, end in [("1972-01-01", "2026-08-01"), ("2000-01-01", "2026-08-01")]:
    print(f"{start[:4]}-{end[:4]}")
    for name, pos in [(f"SMA{n}", above_sma(n)) for n in (6, 8, 10, 12)] + \
                     [(f"MOM{n}", momentum(n)) for n in (6, 9, 12)]:
        cagr, _, sharpe, max_dd, _ = run(pos, start, end)
        print(f"  {name:6s} CAGR {cagr:5.1%}  Sharpe {sharpe:.2f}  MaxDD {max_dd:6.1%}")

print("\n=== Latest months ===")
latest = pd.DataFrame({
    "avg_price": s,
    "sma10": s.rolling(10).mean().round(0),
    "mom12": (s / s.shift(12) - 1).round(3),
}).tail(6)
print(latest.to_string())
