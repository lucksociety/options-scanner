#!/usr/bin/env python3
"""
3-4 week market outlook for the Luck Society Option Scanner: should we be buying calls or puts?

The squeeze regime (market.py) asks "will the tape reward a heavily shorted stock today". This asks a different
question: over the LIFE OF A CONTRACT (15-25 sessions) is the broad market more likely to be higher or lower?
It is computed as a full daily HISTORY (one row per session) so that exactly the same arithmetic can be replayed
over years of data and scored against what SPY actually did 15 sessions later — the banner cites those measured
odds rather than an opinion.

Score: bias in roughly -100..+100. Five groups, each explainable on the card:
  trend        (+/-25)  SPY/QQQ vs their 20 & 50-day, 20 over 50, 50-day slope
  breadth      (+/-20)  share of the 11 sector ETFs above their 50-day; equal-weight and small caps vs SPY
  volatility   (+/-20)  VIX level, VIX vs its 20-day average, VIX term structure (spot vs 3-month)
  appetite     (+/-15)  high yield vs treasuries, cyclicals vs defensives, 10-year yield trend
  reversion    (+/-20)  what mean-reverts inside 3-4 weeks: SPY RSI extremes, stretch from the 50-day, post-panic
                        (VIX spiked above 25 and has since fallen 20%+), drawdown-and-turn
Bands: >= +30 favor calls · +10..+30 lean calls · -10..+10 mixed · -30..-10 lean puts · <= -30 favor puts

CALIBRATION. The first backtest (2019-2026, 1,734 sessions) showed the raw bias points the WRONG way at this horizon:
the most bearish readings (panic: VIX spiking, everything under the 50-day) were the best 15-session buys, and the worst
bucket was a trend that had just rolled over with no panic yet. Trend, breadth and volatility all mean-reverted; only
risk appetite and the reversion group scored in the intuitive direction. So the live verdict is not the raw bias: each
component's bucket is mapped to the EXCESS 15-session SPY return that bucket actually produced (shrunk toward zero for
thin buckets), the five excesses are summed into an expected excess return, and the band comes from that. The raw bias
is still shown as a descriptor. outlook_backtest.py fits the table on the first ~70% of history and reports the bands'
hit-rates on the held-out last ~30%, so the banner's odds are not pure in-sample flattery.
"""
import numpy as np
import pandas as pd

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
TICKERS = ["SPY", "QQQ", "IWM", "RSP", "^VIX", "^VIX3M", "HYG", "IEF", "^TNX"] + SECTORS
WEIGHTS = dict(trend=25, breadth=20, vol=20, appetite=15, reversion=20)
BANDS = [(30, "favor_calls", "Favor calls"), (10, "lean_calls", "Lean calls"), (-10, "mixed", "Mixed"),
         (-30, "lean_puts", "Lean puts"), (-999, "favor_puts", "Favor puts")]


PARTS = ("trend", "breadth", "vol", "appetite", "reversion")
BUCKET = 10                       # component buckets are 10 points wide
EXP_BANDS = [(0.6, "favor_calls", "Favor calls"), (0.2, "lean_calls", "Lean calls"), (-0.2, "mixed", "Mixed"),
             (-0.6, "lean_puts", "Lean puts"), (-999, "favor_puts", "Favor puts")]


def bucket_key(v):
    lo = int(np.floor(v / BUCKET) * BUCKET)
    return f"{lo}..{lo + BUCKET}"


def calibrate(F, fwd, shrink=60):
    """Lookup table {part: {bucket: excess}}: mean forward return of the bucket minus the overall mean, shrunk toward 0
    by n/(n+shrink) so a 12-session bucket cannot dominate. fwd = forward SPY return series aligned to F."""
    base = float(fwd.mean()); table = {"base": round(base, 3), "parts": {}}
    for p in PARTS:
        keys = F[p].map(bucket_key); t = {}
        for k in sorted(set(keys), key=lambda x: int(x.split("..")[0])):
            m = keys == k; n = int(m.sum())
            if n == 0: continue
            ex = float(fwd[m].mean()) - base
            t[k] = dict(n=n, excess=round(ex * n / (n + shrink), 3), raw=round(ex, 3), up=round(float((fwd[m] > 0).mean() * 100), 1))
        table["parts"][p] = t
    return table


def expected(F, table):
    """Expected 15-session SPY excess return per session from the calibration table (0 for unseen buckets)."""
    out = pd.Series(0.0, index=F.index)
    for p in PARTS:
        t = table["parts"].get(p, {})
        out += F[p].map(lambda v: t.get(bucket_key(v), {}).get("excess", 0.0))
    return out


def exp_band_of(x):
    if x is None or x != x: return "unknown", "Unknown"
    for lo, key, label in EXP_BANDS:
        if x >= lo: return key, label
    return "favor_puts", "Favor puts"


def band_of(bias):
    if bias is None or bias != bias: return "unknown", "Unknown"
    for lo, key, label in BANDS:
        if bias >= lo: return key, label
    return "favor_puts", "Favor puts"


def _closes(hist, tickers):
    """{ticker: close Series} from a yfinance group_by='ticker' download; missing tickers are simply absent."""
    out = {}
    for t in tickers:
        try:
            s = hist[t]["Close"].dropna() if isinstance(hist.columns, pd.MultiIndex) else hist["Close"].dropna()
            if len(s): out[t] = s.astype(float)
        except Exception:
            pass
    return out


def _rsi(s, n=14):
    d = s.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1 / n, adjust=False).mean(); al = l.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))


def _step(x, cuts):
    """Piecewise score: cuts = [(threshold, value), ...] descending; first threshold x >= wins, else last value."""
    out = pd.Series(np.nan, index=x.index)
    rem = pd.Series(True, index=x.index)
    for thr, val in cuts:
        m = rem & (x >= thr); out[m] = val; rem &= ~m
    return out.fillna(cuts[-1][1]) if cuts else out


def outlook_frame(hist):
    """One row per session: every component, its note inputs, the bias and the band. Needs ~260 sessions of warm-up."""
    C = _closes(hist, TICKERS)
    spy = C.get("SPY")
    if spy is None or len(spy) < 80: raise RuntimeError("SPY history missing")
    idx = spy.index
    A = lambda t: C[t].reindex(idx).ffill() if t in C else None
    F = pd.DataFrame(index=idx)

    # ---- trend (+/-25)
    s20, s50 = spy.rolling(20).mean(), spy.rolling(50).mean()
    tr = pd.Series(0.0, index=idx)
    tr += np.where(spy > s50, 7, -7); tr += np.where(spy > s20, 5, -5); tr += np.where(s20 > s50, 5, -5)
    slope = s50 / s50.shift(10) - 1
    tr += np.where(slope > 0.002, 4, np.where(slope < -0.002, -4, 0))
    qqq = A("QQQ")
    if qqq is not None: tr += np.where(qqq > qqq.rolling(50).mean(), 4, -4)
    F["trend"] = tr.clip(-25, 25); F["spy_vs50"] = (spy / s50 - 1) * 100; F["spy_vs20"] = (spy / s20 - 1) * 100

    # ---- breadth (+/-20)
    above = []
    for t in SECTORS:
        s = A(t)
        if s is not None: above.append((s > s.rolling(50).mean()).astype(float))
    br = pd.Series(0.0, index=idx)
    if above:
        pct = pd.concat(above, axis=1).mean(axis=1) * 100
        F["sect_pct"] = pct
        br += _step(pct, [(70, 10), (55, 5), (45, 0), (30, -5), (-1, -10)])
    rsp, iwm = A("RSP"), A("IWM")
    if rsp is not None:
        b20 = (rsp / rsp.shift(20) - 1 - (spy / spy.shift(20) - 1)) * 100; F["rsp20"] = b20
        br += _step(b20, [(0.5, 5), (-0.5, 0), (-99, -5)])
    if iwm is not None:
        i20 = (iwm / iwm.shift(20) - 1 - (spy / spy.shift(20) - 1)) * 100; F["iwm20"] = i20
        br += _step(i20, [(1.0, 5), (-1.0, 0), (-99, -5)])
    F["breadth"] = br.clip(-20, 20)

    # ---- volatility (+/-20)
    vix, v3 = A("^VIX"), A("^VIX3M")
    vo = pd.Series(0.0, index=idx)
    if vix is not None:
        F["vix"] = vix
        vo += _step(vix, [(30, -10), (25, -6), (20, -2), (15, 3), (-1, 6)])
        va = vix / vix.rolling(20).mean() - 1; F["vix_vs20"] = va * 100
        vo += _step(va, [(0.15, -6), (-0.10, 0), (-99, 6)])
        if v3 is not None:
            ts = vix / v3; F["vix_ts"] = ts
            vo += _step(ts, [(1.0, -8), (0.9, 3), (-99, 8)])
    F["vol"] = vo.clip(-20, 20)

    # ---- risk appetite (+/-15)
    ap = pd.Series(0.0, index=idx)
    hyg, ief = A("HYG"), A("IEF")
    if hyg is not None and ief is not None:
        cr = ((hyg / ief) / (hyg / ief).shift(20) - 1) * 100; F["credit20"] = cr
        ap += _step(cr, [(1.0, 6), (-1.0, 0), (-99, -6)])
    xly, xlk, xlu, xlp = A("XLY"), A("XLK"), A("XLU"), A("XLP")
    if all(x is not None for x in (xly, xlk, xlu, xlp)):
        cyc = (xly / xly.shift(20) - 1 + xlk / xlk.shift(20) - 1) / 2; dfn = (xlu / xlu.shift(20) - 1 + xlp / xlp.shift(20) - 1) / 2
        cd = (cyc - dfn) * 100; F["cyc_def20"] = cd
        ap += _step(cd, [(2.0, 5), (-2.0, 0), (-99, -5)])
    tnx = A("^TNX")
    if tnx is not None:
        y20 = (tnx / tnx.shift(20) - 1) * 100; F["tnx20"] = y20
        ap += _step(y20, [(5.0, -4), (-3.0, 0), (-99, 4)])
    F["appetite"] = ap.clip(-15, 15)

    # ---- 3-4 week mean reversion (+/-20)
    rv = pd.Series(0.0, index=idx)
    rsi = _rsi(spy); F["rsi"] = rsi
    rv += _step(rsi, [(75, -8), (65, -3), (40, 0), (30, 4), (-1, 8)])
    st = (spy / s50 - 1) * 100
    rv += _step(st, [(7, -6), (-7, 0), (-99, 6)])
    if vix is not None:
        pk = vix.rolling(20).max()
        post_panic = (pk > 25) & (vix < pk * 0.8); F["post_panic"] = post_panic
        rv += np.where(post_panic, 8, 0)
    hi60 = spy.rolling(60).max(); lo20 = spy.rolling(20).min()
    dd = (lo20 / hi60 - 1) * 100; off = (spy / lo20 - 1) * 100; F["dd60"] = dd; F["off_low"] = off
    rv += np.where((dd <= -6) & (off >= 2), 5, 0)
    F["reversion"] = rv.clip(-20, 20)

    F["bias"] = F[["trend", "breadth", "vol", "appetite", "reversion"]].sum(axis=1).clip(-100, 100)
    F["band"] = [band_of(b)[0] for b in F["bias"]]
    return F


def outlook_today(hist, table=None):
    """The live reading: last row of outlook_frame plus human notes for the banner. With a calibration table the
    verdict comes from the expected excess return; without one it falls back to the raw bias and says so."""
    F = outlook_frame(hist)
    r = F.iloc[-1]
    def g(k, nd=1):
        v = r.get(k)
        return None if v is None or (isinstance(v, float) and v != v) else (round(float(v), nd) if isinstance(v, (float, int, np.floating, np.integer)) else v)
    bias = g("bias"); raw_key, raw_label = band_of(bias)
    exp_ = None; contrib = {}
    if table:
        exp_ = round(float(expected(F.iloc[[-1]], table).iloc[0]), 2)
        for p in PARTS:
            b = table["parts"].get(p, {}).get(bucket_key(float(r[p])), {})
            contrib[p] = dict(bucket=bucket_key(float(r[p])), excess=b.get("excess"), up=b.get("up"), n=b.get("n"))
        key, label = exp_band_of(exp_)
    else:
        key, label = raw_key, raw_label
    notes = []
    if g("spy_vs50") is not None: notes.append(f"SPY {g('spy_vs50'):+.1f}% vs 50-day, {g('spy_vs20'):+.1f}% vs 20-day")
    if g("sect_pct") is not None: notes.append(f"{g('sect_pct', 0):.0f}% of sectors above their 50-day")
    if g("iwm20") is not None: notes.append(f"IWM {g('iwm20'):+.1f}pp vs SPY / 20d")
    if g("vix") is not None:
        s = f"VIX {g('vix'):.1f} ({g('vix_vs20'):+.0f}% vs its 20-day avg)"
        if g("vix_ts", 2) is not None: s += ", " + ("backwardation — stress" if g("vix_ts", 2) > 1.0 else f"term structure {g('vix_ts', 2)}")
        notes.append(s)
    if g("credit20") is not None: notes.append(f"high yield vs treasuries {g('credit20'):+.1f}% / 20d")
    if g("cyc_def20") is not None: notes.append(f"cyclicals vs defensives {g('cyc_def20'):+.1f}pp / 20d")
    if g("rsi") is not None: notes.append(f"SPY RSI {g('rsi', 0):.0f}")
    if bool(r.get("post_panic")): notes.append("post-panic: VIX spiked above 25 and has fallen 20%+")
    verdict = {"favor_calls": "Tape favors the long side over the next 3–4 weeks: lean on the Calls and Breakout boards.",
               "lean_calls": "Mild upward tilt over the next 3–4 weeks: calls have the edge, keep puts to the strongest breakdowns.",
               "mixed": "No directional edge over the next 3–4 weeks: trade the setup, not the market; size smaller both ways.",
               "lean_puts": "Mild downward tilt over the next 3–4 weeks: puts have the edge, keep calls to fired triggers only.",
               "favor_puts": "Tape favors the short side over the next 3–4 weeks: lean on the Bear board.",
               "unknown": "Outlook data unavailable."}[key]
    parts = {k: g(k, 1) for k in PARTS}
    # score for the dial: calibrated expectation mapped so 0 = -1.5% excess, 50 = 0, 100 = +1.5%; raw bias if uncalibrated
    score = (round(float(np.clip(50 + exp_ / 1.5 * 50, 0, 100))) if exp_ is not None
             else (round((bias + 100) / 2) if bias is not None else None))
    return dict(bias=bias, raw_band=raw_key, raw_label=raw_label, exp=exp_, calibrated=bool(table), contrib=contrib,
                score=score, band=key, label=label, verdict=verdict,
                parts=parts, weights=WEIGHTS, notes=notes, asof=str(F.index[-1].date()))
