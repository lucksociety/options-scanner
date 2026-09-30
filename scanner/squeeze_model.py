#!/usr/bin/env python3
"""
Squeeze model: a measured probability that a $1-10 stock reaches +50% (and +30%) within 20 sessions, for EVERY name
on EVERY day, fitted on two years of history and tested out of sample. The live scanner ranks by this number.

Why: the tracker showed the point-sheet ranking (fuel + bottoming) picked names with a median best move of +2%
over two weeks and zero +50% hits, while the trigger backtest showed the edge lives in the spike -- volume ignition
and relative strength on a small float. So instead of guessing weights, every feature is bucketed, each bucket's
measured hit-rate becomes a log-odds contribution (shrunk toward the base rate for thin buckets), the contributions
are summed, and the result is calibrated back to a probability on held-out data. Naive-Bayes-with-shrinkage: crude,
transparent, and every card can show which buckets pushed it up.

Features (all computed from daily bars the scanner already downloads; static fuel from key stats):
  rvol      today's volume / 63-day average           vr3      3-day avg volume / 20-day
  ret1 ret5 ret20                                     rs5      5-day return minus SPY's
  brk10 brk20   close above the prior 10 / 20-day high (fresh breakout)
  h20 lo20  distance from the 20-day high / low        lo52     distance above the 52-week low
  rsi14     clpos (close position in the day's range)  gap      open vs prior close
  s20 s50   price vs 20 / 50-day average               atr      ATR(14) as % of price
  dv        average daily dollar volume ($M)            px       price
  sf sr flt sharesShort/float, days to cover, float (M) -- today's key stats (no history, a known compromise)
Target: best high over the next 20 sessions from the NEXT day's open >= +50% (hit50) and >= +30% (hit30).
Outputs data/backtest/squeeze_model.json (the table + held-out lift by decile) and squeeze_rows.csv (sample).
"""
import json, logging, math, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery as EQ

log = logging.getLogger("squeeze_model"); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "data" / "backtest"

# bucket edges per feature (right-open); the last bucket is >= last edge
EDGES = {
    "rvol": [0.5, 0.8, 1.2, 1.7, 2.5, 4, 7], "vr3": [0.6, 0.9, 1.3, 2, 3, 5],
    "ret1": [-10, -4, -1, 1, 4, 10, 20], "ret5": [-20, -8, -2, 3, 10, 25, 50], "ret20": [-30, -15, -5, 5, 15, 35, 70],
    "rs5": [-15, -5, 0, 5, 12, 25], "h20": [-40, -25, -15, -8, -3, 0], "lo20": [3, 8, 15, 30, 60],
    "lo52": [5, 15, 30, 60, 120], "rsi": [30, 40, 50, 60, 70, 80], "clpos": [0.25, 0.5, 0.75, 0.9],
    "gap": [-5, -1, 1, 4, 10, 20], "s20": [-20, -10, -3, 3, 10, 25], "s50": [-30, -15, -5, 5, 15, 35],
    "atr": [4, 6, 8, 11, 15, 20], "dv": [0.5, 1, 2, 5, 12, 30], "px": [1.5, 2.5, 4, 6, 8],
    "sf": [3, 8, 15, 25, 40], "sr": [1, 2, 4, 7, 12], "flt": [10, 25, 50, 100, 300],
}
FLAGS = ["brk10", "brk20", "above20", "stack"]          # boolean features, two buckets each
INTER = {"rvol_x_flt": (("rvol", [1.7, 4]), ("flt", [25, 100])),          # a few interactions the folklore insists on
         "rvol_x_sf": (("rvol", [1.7, 4]), ("sf", [8, 20])),
         "brk_x_rvol": (("brk10", None), ("rvol", [1.7, 4]))}


def universe():
    q = EQ("and", [EQ("btwn", ["intradayprice", 1.0, 10.0]), EQ("gt", ["avgdailyvol3m", 300_000]), EQ("eq", ["region", "us"])])
    out, offset = {}, 0
    while True:
        res = yf.screen(q, offset=offset, size=250, sortField="avgdailyvol3m", sortAsc=False)
        quotes = res.get("quotes", []) if res else []
        for x in quotes:
            if x.get("symbol"): out[x["symbol"]] = x
        if len(quotes) < 250 or offset > 4000: break
        offset += 250; time.sleep(0.4)
    return out


def rsi_s(c, n=14):
    d = c.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1 / n, adjust=False).mean(); al = l.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))


def features(h, spy):
    """Per-session feature frame for one name. h = OHLCV DataFrame, spy = SPY close aligned to h.index."""
    c, hi, lo, op, v = h["Close"], h["High"], h["Low"], h["Open"], h["Volume"]
    F = pd.DataFrame(index=h.index)
    av63 = v.rolling(63).mean(); av20 = v.rolling(20).mean()
    F["rvol"] = v / av63; F["vr3"] = v.rolling(3).mean() / av20
    F["ret1"] = (c / c.shift(1) - 1) * 100; F["ret5"] = (c / c.shift(5) - 1) * 100; F["ret20"] = (c / c.shift(20) - 1) * 100
    F["rs5"] = F["ret5"] - (spy / spy.shift(5) - 1) * 100
    ph10 = hi.shift(1).rolling(10).max(); ph20 = hi.shift(1).rolling(20).max()
    F["brk10"] = (c > ph10).astype(int); F["brk20"] = (c > ph20).astype(int)
    F["h20"] = (c / hi.rolling(20).max() - 1) * 100; F["lo20"] = (c / lo.rolling(20).min() - 1) * 100
    F["lo52"] = (c / c.rolling(252, min_periods=120).min() - 1) * 100
    F["rsi"] = rsi_s(c); rng = (hi - lo); F["clpos"] = ((c - lo) / rng.replace(0, np.nan)).fillna(0.5)
    F["gap"] = (op / c.shift(1) - 1) * 100
    s20 = c.rolling(20).mean(); s50 = c.rolling(50).mean()
    F["s20"] = (c / s20 - 1) * 100; F["s50"] = (c / s50 - 1) * 100
    F["above20"] = (c > s20).astype(int); F["stack"] = ((c > s20) & (s20 > s50)).astype(int)
    tr = pd.concat([hi - lo, (hi - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    F["atr"] = tr.rolling(14).mean() / c * 100; F["dv"] = av63 * c / 1e6; F["px"] = c
    # targets: from the NEXT day's open, best high over the following 20 sessions
    entry = op.shift(-1)
    fh = hi.shift(-1).rolling(20).max().shift(-19)                     # max of high[t+1 .. t+20]
    F["best20"] = (fh / entry - 1) * 100
    F["hit50"] = (F["best20"] >= 50).astype(float); F["hit30"] = (F["best20"] >= 30).astype(float)
    F.loc[F["best20"].isna(), ["hit50", "hit30"]] = np.nan
    return F


def bucket(name, v):
    if name in FLAGS: return "1" if v else "0"
    e = EDGES[name]
    if v is None or (isinstance(v, float) and v != v): return "na"
    for i, x in enumerate(e):
        if v < x: return f"<{x}" if i == 0 else f"{e[i-1]}..{x}"
    return f">={e[-1]}"


def bucket_series(name, series):
    """Vectorised bucket() for a whole column."""
    if name in FLAGS: return series.fillna(0).astype(bool).map({True: "1", False: "0"})
    e = EDGES[name]; labels = [f"<{e[0]}"] + [f"{e[i-1]}..{e[i]}" for i in range(1, len(e))] + [f">={e[-1]}"]
    idx = np.searchsorted(np.asarray(e, dtype=float), series.to_numpy(dtype=float), side="right")
    out = pd.Series(np.array(labels)[idx], index=series.index); out[series.isna()] = "na"
    return out


def _cut(series, edges):
    """Vectorised bucket index as strings: None edges = boolean flag, else np.searchsorted; NaN -> 'na'."""
    if edges is None: return series.fillna(0).astype(bool).map({True: "1", False: "0"})
    idx = np.searchsorted(np.asarray(edges, dtype=float), series.to_numpy(dtype=float), side="right")
    out = pd.Series(idx.astype(str), index=series.index); out[series.isna()] = "na"
    return out


def inter_keys(name, D):
    (a, ea), (b, eb) = INTER[name]
    return _cut(D[a], ea) + "|" + _cut(D[b], eb)


def inter_key(name, row):
    """Single-row version (the live scanner scores one name at a time)."""
    return inter_keys(name, pd.DataFrame([row])).iloc[0]


def fit(D, target="hit50", shrink=150):
    """Bucket log-odds table. contribution = log(p_bucket/p_base) shrunk by n/(n+shrink)."""
    base = float(D[target].mean()); T = {"base": round(base, 5), "target": target, "feats": {}, "inter": {}}
    def lo(p, n):
        p = min(max(p, 1e-4), 1 - 1e-4); return math.log(p / (1 - p)) - math.log(base / (1 - base)), n
    for f in list(EDGES) + FLAGS:
        keys = bucket_series(f, D[f]); t = {}
        for k, g in D.groupby(keys)[target]:
            n = int(g.count()); p = float(g.mean()) if n else base
            l, _ = lo(p, n); t[k] = dict(n=n, p=round(p, 4), w=round(l * n / (n + shrink), 4))
        T["feats"][f] = t
    for name in INTER:
        keys = inter_keys(name, D); t = {}
        for k, g in D.groupby(keys)[target]:
            n = int(g.count()); p = float(g.mean()) if n else base
            l, _ = lo(p, n); t[k] = dict(n=n, p=round(p, 4), w=round(l * n / (n + shrink), 4))
        T["inter"][name] = t
    return T


def score(D, T, damp=0.55):
    """Sum of bucket log-odds (damped, because the features are correlated), back to a probability."""
    base = T["base"]; s = pd.Series(0.0, index=D.index)
    for f, t in T["feats"].items():
        s += bucket_series(f, D[f]).map(lambda k: t.get(k, {}).get("w", 0.0))
    for name, t in T["inter"].items():
        s += inter_keys(name, D).map(lambda k: t.get(k, {}).get("w", 0.0))
    z = math.log(base / (1 - base)) + damp * s
    return 1 / (1 + np.exp(-z)), s


def calib_map(z, y, bins=20):
    """Held-out mapping from raw damped log-odds to the ACTUAL hit-rate, by quantile bin: [[z_mean, actual], ...].
    The live scanner interpolates on this, so the probability on the card is what that score really produced."""
    q = pd.qcut(z.rank(method="first"), bins, labels=False); out = []
    for d in range(bins):
        m = q == d; out.append([round(float(z[m].mean()), 4), round(float(y[m].mean()), 4)])
    return out


def apply_calib(z, cmap):
    zs = np.array([a for a, _ in cmap]); ys = np.array([b for _, b in cmap])
    return np.interp(np.asarray(z, dtype=float), zs, ys)


def calibrate(p, y, bins=10):
    """Held-out reliability by decile of predicted probability: n, predicted mean, actual hit-rate, avg best20."""
    q = pd.qcut(p.rank(method="first"), bins, labels=False)
    out = []
    for d in range(bins):
        m = q == d
        out.append(dict(decile=int(d + 1), n=int(m.sum()), pred=round(float(p[m].mean()) * 100, 2), actual=round(float(y[m].mean()) * 100, 2)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    U = universe(); syms = sorted(U); log.info("universe: %d names", len(syms))
    hist = yf.download(syms + ["SPY"], period="2y", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    spy = hist["SPY"]["Close"].dropna()
    stat = {}
    for t in syms:                                                   # today's key stats: static fuel (compromise, stated)
        try:
            info = yf.Ticker(t).info
            sf = float(info.get("shortPercentOfFloat") or 0) * 100; ss = info.get("sharesShort"); fl = info.get("floatShares")
            if not sf and ss and fl: sf = ss / fl * 100
            stat[t] = dict(sf=sf, sr=float(info.get("shortRatio") or 0), flt=(fl / 1e6) if fl else np.nan)
        except Exception: stat[t] = dict(sf=np.nan, sr=np.nan, flt=np.nan)
        time.sleep(0.1)
    rows = []
    for t in syms:
        try:
            h = hist[t].dropna(subset=["Close"])
            if len(h) < 120: continue
            F = features(h, spy.reindex(h.index).ffill()); F["t"] = t
            for k, v in stat[t].items(): F[k] = v
            rows.append(F.iloc[70:])
        except Exception as e: log.debug("%s: %s", t, e)
    D = pd.concat(rows); D = D[D["px"].between(0.8, 12)]           # keep the band the scanner actually trades
    D["date"] = D.index
    known = D[D["hit50"].notna()].copy()
    log.info("%d name-days, %d with a known outcome, base hit50 %.2f%% hit30 %.2f%%", len(D), len(known), known["hit50"].mean() * 100, known["hit30"].mean() * 100)
    # time split: fit on the first 70% of dates, test on the last 30%
    dates = sorted(known["date"].unique()); cut = dates[int(len(dates) * 0.7)]
    train, test = known[known["date"] < cut], known[known["date"] >= cut]
    T50 = fit(train, "hit50"); T30 = fit(train, "hit30")
    p50, z50 = score(test, T50); p30, z30 = score(test, T30)
    rel50 = calibrate(p50, test["hit50"]); rel30 = calibrate(p30, test["hit30"])
    cmap50 = calib_map(z50, test["hit50"]); cmap30 = calib_map(z30, test["hit30"])
    c50 = pd.Series(apply_calib(z50, cmap50), index=test.index)
    relc = calibrate(c50, test["hit50"])                                   # reliability AFTER calibration (should sit on the diagonal)
    top = test.assign(p=p50).sort_values("p", ascending=False)
    top_n = {k: dict(n=k, hit50=round(float(top.head(k)["hit50"].mean()) * 100, 1), hit30=round(float(top.head(k)["hit30"].mean()) * 100, 1),
                     med_best=round(float(top.head(k)["best20"].median()), 1)) for k in (100, 300, 1000, 3000)}
    # refit on everything for the live table
    A50 = fit(known, "hit50"); A30 = fit(known, "hit30")
    model = dict(asof=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                 names=int(D["t"].nunique()), rows=int(len(known)), first=str(dates[0])[:10], last=str(dates[-1])[:10], split=str(cut)[:10],
                 base50=A50["base"], base30=A30["base"], edges=EDGES, flags=FLAGS, inter={k: [list(v[0]), list(v[1])] for k, v in INTER.items()},
                 t50=A50, t30=A30, calib50=cmap50, calib30=cmap30,
                 heldout=dict(n=int(len(test)), base50=round(float(test["hit50"].mean()), 5), base30=round(float(test["hit30"].mean()), 5),
                              rel50=rel50, rel30=rel30, relc50=relc, top=top_n),
                 caveat="Today's universe replayed backwards (survivorship-biased) with today's short interest / float applied to every past day (no free SI history). Entry = next day's open.")
    json.dump(model, open(OUT / "squeeze_model.json", "w"), separators=(",", ":"))
    known.sample(min(20000, len(known)), random_state=1).round(3).to_csv(OUT / "squeeze_rows.csv", index=False)
    log.info("held-out top-100 hit50 %s%% (base %.1f%%) · deciles: %s", top_n[100]["hit50"], test["hit50"].mean() * 100, [(r["decile"], r["pred"], r["actual"]) for r in rel50])
    return 0


if __name__ == "__main__":
    sys.exit(main())
